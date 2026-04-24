import numpy as np
import time
import torch
import os
import json
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

# --- LỚP BỔ TRỢ THEO DÕI CHỈ SỐ ---
class AverageMeter(object):
    """Tính toán và lưu trữ giá trị trung bình/hiện tại"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- CÁC HÀM TIỆN ÍCH HỖ TRỢ TRAIN ---

def clip_gradient(optimizer, grad_clip):
    """Giới hạn độ dốc (Gradient Clipping) để tránh bùng nổ gradient."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy_top_k(scores, targets, k):
    """Tính toán Top-K accuracy."""
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)

def adjust_learning_rate(optimizer, shrink_factor):
    """Giảm tốc độ học (Learning Rate)."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print(f"\n[*] Đã giảm Learning Rate. LR mới: {optimizer.param_groups[0]['lr']}")

def save_checkpoint_caption(data_name, epoch, epochs_since_improvement, encoder, decoder, 
                            encoder_optimizer, decoder_optimizer, bleu4, is_best):
    """Lưu trữ mô hình (Checkpoint)."""
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict() if encoder_optimizer is not None else None,
        'decoder_optimizer': decoder_optimizer.state_dict()
    }
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    filename = f'models/checkpoint_{data_name}.pth.tar'
    torch.save(state, filename)
    
    if is_best:
        torch.save(state, f'models/BEST_checkpoint_{data_name}.pth.tar')
        print(f"🌟 Đã cập nhật mô hình tốt nhất tại Epoch {epoch}")

# --- CÁC HÀM THỰC THI CHÍNH ---

def train_caption(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, cfg, device):
    decoder.train() 
    encoder.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    alpha_c = cfg['training']['alpha_c']
    grad_clip = cfg['training']['grad_clip']
    print_freq = cfg['training']['print_freq']
    top_k = cfg['evaluation']['top_k']

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        # Chuyển dữ liệu sang GPU
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]

        # Nén chuỗi để tính loss
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Backward
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Metrics
        top5 = accuracy_top_k(scores, targets, top_k)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    loss=losses, top5=top5accs))
    return losses.avg, top5accs.avg

def validate_caption(val_loader, encoder, decoder, criterion, word_map, cfg, device):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    losses = AverageMeter()
    top5accs = AverageMeter()

    references = list()
    hypotheses = list()

    alpha_c = cfg['training']['alpha_c']
    top_k = cfg['evaluation']['top_k']

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy_top_k(scores, targets, top_k)
            top5accs.update(top5, sum(decode_lengths))

            # --- SỬA LỖI GPU/CPU Ở ĐÂY ---
            sort_ind = sort_ind.cpu()
            allcaps = allcaps.cpu()[sort_ind] 

            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                # Loại bỏ <start>, <pad>, <end> để tính BLEU chính xác
                img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}], img_caps))
                references.append(img_captions)

            # Dự đoán (Hypotheses)
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.cpu().tolist() # Phải đưa về CPU trước khi tolist
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            hypotheses.extend(temp_preds)

        bleu4 = corpus_bleu(references, hypotheses)
        print(f'\n * VALIDATION - LOSS {losses.avg:.3f}, TOP-5 ACC {top5accs.avg:.3f}, BLEU-4 {bleu4:.4f}\n')

    return bleu4, losses.avg