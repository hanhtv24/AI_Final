import json
import os
import yaml
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import datetime

# --- IMPORT TỪ SRC ---
from src.data.data_loader import CaptionDataset
from src.models.Decoder import DecoderWithAttention
from src.models.Resnet101 import Encoder
from src.utils.util import *

def main():
    # 1. ĐỌC CẤU HÌNH TỪ FILE YAML
    with open('configs/default.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    data_folder = cfg['dataset']['data_folder']
    data_name = cfg['dataset']['data_name']
    
    # 2. KHỞI TẠO TENSORBOARD
    log_dir = 'logs/run_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[*] TensorBoard initialized. Run: `tensorboard --logdir=logs` to view.")

    # 3. THIẾT LẬP THIẾT BỊ (Ưu tiên GPU RTX 3060)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 4. ĐỌC TỪ ĐIỂN (WORD MAP)
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # 5. KHỞI TẠO MÔ HÌNH
    encoder = Encoder()
    decoder = DecoderWithAttention(
        attention_dim=cfg['model']['attention_dim'],
        embed_dim=cfg['model']['emb_dim'],
        decoder_dim=cfg['model']['decoder_dim'],
        vocab_size=len(word_map),
        dropout=cfg['model']['dropout']
    )

    # Khởi tạo Optimizer cho Decoder
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=float(cfg['training']['decoder_lr'])
    )
    
    # Thiết lập Fine-tune cho Encoder (ResNet101)
    fine_tune_encoder = cfg['model']['fine_tune_encoder']
    encoder.fine_tune(fine_tune_encoder)
    
    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=float(cfg['training']['encoder_lr']), 
        weight_decay=float(cfg['training']['encode_weight_decay'])
    ) if fine_tune_encoder else None

    # ĐẨY MÔ HÌNH LÊN GPU
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 6. CHUẨN BỊ DỮ LIỆU (DATALOADER)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Loader cho Training
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['training']['workers'], 
        pin_memory=True  # Tối ưu truyền dữ liệu lên GPU
    )

    # Loader cho Validation (Lưu ý: CaptionDataset trả về 4 giá trị ở chế độ VAL)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=cfg['training']['batch_size'], 
        shuffle=False, # Thường để False cho Validation 
        num_workers=cfg['training']['workers'], 
        pin_memory=True
    )

    # 7. VÒNG LẶP HUẤN LUYỆN
    epochs = cfg['training']['epochs']
    best_bleu4 = 0.
    epochs_since_improvement = 0

    for epoch in range(epochs):
        # Kiểm tra điều kiện dừng sớm
        if epochs_since_improvement == 20:
            print("[!] Dừng sớm (Early Stopping) vì mô hình không cải thiện.")
            break
            
        # Điều chỉnh LR nếu cần
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # --- TRAIN EPOCH ---
        train_loss, train_acc = train_caption(
            train_loader=train_loader, 
            encoder=encoder, 
            decoder=decoder,
            criterion=criterion, 
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer, 
            epoch=epoch, 
            cfg=cfg, 
            device=device
        )
        
        # Ghi log Train
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train_Top5', train_acc, epoch)

        # --- VALIDATION ---
        recent_bleu4, val_loss = validate_caption(
            val_loader=val_loader, 
            encoder=encoder, 
            decoder=decoder,
            criterion=criterion, 
            word_map=word_map, 
            cfg=cfg, 
            device=device
        )
        
        # Ghi log Val
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/BLEU-4', recent_bleu4, epoch)

        # --- LƯU CHECKPOINT ---
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        
        if not is_best:
            epochs_since_improvement += 1
            print(f"[*] Epochs since improvement: {epochs_since_improvement}")
        else:
            epochs_since_improvement = 0

        save_checkpoint_caption(
            data_name, epoch, epochs_since_improvement, encoder, decoder, 
            encoder_optimizer, decoder_optimizer, recent_bleu4, is_best
        )

    writer.close()

if __name__ == '__main__':
    main()