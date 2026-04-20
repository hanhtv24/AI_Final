import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import yaml
import h5py
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

# Import kiến trúc mạng (khung gỗ để lắp ráp)
from src.models.Resnet101 import Encoder
from src.models.Decoder import DecoderWithAttention

# Đọc cấu hình từ file yaml
with open('configs/default.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Lấy các đường dẫn từ cấu hình
data_folder = cfg['dataset']['data_folder']
data_name = cfg['dataset']['data_name']
checkpoint_path = cfg['evaluation']['checkpoint_path']
beam_size = cfg['evaluation']['top_k']

word_map_file = os.path.join(data_folder, f'WORDMAP_{data_name}.json')
test_image_path = os.path.join(data_folder, f'TEST_IMAGES_{data_name}.hdf5')
test_caps_path = os.path.join(data_folder, f'TEST_CAPTIONS_{data_name}.json')
test_caplens_path = os.path.join(data_folder, f'TEST_CAPLENS_{data_name}.json')

# CÁC THÔNG SỐ KIẾN TRÚC MẶC ĐỊNH
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5

def evaluate(beam_size):
    # 1. Đọc Word Map trước để biết kích thước từ vựng (vocab_size)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    vocab_size = len(word_map)
    rev_word_map = {v: k for k, v in word_map.items()}

    print(f"[*] Đang tải trọng số từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Lắp ráp "Khung" Encoder và nhồi trọng số vào
    encoder = Encoder()
    encoder = encoder.to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    # 3. Lắp ráp "Khung" Decoder và nhồi trọng số vào
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=vocab_size,
                                   dropout=dropout)
    decoder = decoder.to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()

    # Load dữ liệu bài Test
    h = h5py.File(test_image_path, 'r')
    images = h['images']

    with open(test_caps_path, 'r') as j:
        captions = json.load(j)

    with open(test_caplens_path, 'r') as j:
        caplens = json.load(j)

    references = []
    hypotheses = []

    print(f"[*] Đang chạy Beam Search (Beam size = {beam_size}) trên tập Test...")
    for i in tqdm(range(len(images))):
        k = beam_size # <--- ĐÃ SỬA LỖI Ở ĐÂY: Reset lại kích thước chùm k = 5 cho mỗi bức ảnh

        image = torch.FloatTensor(images[i]).unsqueeze(0).to(device)
        image = image / 255.

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        encoder_out = encoder(image)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)

        h_state, c_state = decoder.init_hidden_state(encoder_out)

        step = 1
        complete_seqs = list()
        complete_seqs_scores = list()

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)

            awe, _ = decoder.attention(encoder_out, h_state)
            gate = decoder.sigmoid(decoder.f_beta(h_state))
            awe = gate * awe

            h_state, c_state = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h_state, c_state))
            scores = decoder.fc(h_state)
            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)

            if k == 0 or step > 50:
                break

            seqs = seqs[incomplete_inds]
            h_state = h_state[prev_word_inds[incomplete_inds]]
            c_state = c_state[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1

        if len(complete_seqs_scores) == 0:
            seq = seqs[0].tolist()
        else:
            i_max = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i_max]

        caps = captions[i * 5:(i * 5) + 5]
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], caps)
        )
        references.append(img_captions)

        pred = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(pred)

    print("[*] Đang tính toán BLEU scores...")
    bleu4 = corpus_bleu(references, hypotheses)

    print("\n" + "="*45)
    print(f"🎉 KẾT QUẢ MÔ HÌNH ATTENTION:")
    print(f"➤ BLEU-4 Score: {bleu4:.4f} ({(bleu4*100):.2f}%)")
    print("="*45 + "\n")

if __name__ == '__main__':
    evaluate(beam_size)