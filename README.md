# Hệ thống Sinh Chú thích Ảnh Tự động (Automatic Image Captioning)

> **Đồ án môn học** | Kiến trúc Encoder–Decoder với cơ chế Soft Attention  
> Ngôn ngữ: Python 3.9+ · Framework: PyTorch · Tập dữ liệu: MS COCO 2014

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Luồng thực thi](#5-luồng-thực-thi)
   - [Bước 1 – Tiền xử lý dữ liệu](#bước-1--tiền-xử-lý-dữ-liệu)
   - [Bước 2 – Huấn luyện mô hình](#bước-2--huấn-luyện-mô-hình)
   - [Bước 3 – Đánh giá định lượng](#bước-3--đánh-giá-định-lượng)
   - [Bước 4 – Trực quan hóa kết quả](#bước-4--trực-quan-hóa-kết-quả)
   - [Bước 5 – Đánh giá trên tập tự thu thập](#bước-5--đánh-giá-trên-tập-tự-thu-thập)
   - [Bước 5 – Đánh giá Encoder](#bước-5--đánh-giá-encoder)
6. [Kết quả thực nghiệm](#6-kết-quả-thực-nghiệm)
7. [Tham khảo](#7-tham-khảo)

---

## 1. Giới thiệu

Đồ án triển khai và so sánh hai phương pháp sinh chú thích ảnh (Image Captioning) dựa trên kiến trúc **Encoder–Decoder**:

| Mô hình | Encoder | Decoder | Đặc điểm |
|---|---|---|---|
| **Baseline** | ResNet-101 | LSTM | Trích xuất đặc trưng toàn cục (global feature) |
| **Proposed** | ResNet-101 | LSTM + Soft Attention | Tập trung vào vùng không gian cụ thể khi sinh từng token |

Cơ chế **Soft Attention** (Bahdanau, 2015) cho phép decoder "nhìn" vào các vị trí khác nhau trên feature map của encoder tại mỗi bước giải mã, giúp câu mô tả sinh ra bám sát nội dung hình ảnh hơn so với phương pháp baseline.

---

## 2. Kiến trúc hệ thống

```
Luồng dữ liệu tổng quan
═══════════════════════════════════════════════════════════════

  ┌─────────────────┐     data_reader.py       ┌─────────────┐
  │  MS COCO 2014   │ ──────────────────────►  │  HDF5 File  │
  │  Images + JSON  │                          │ WORDMAP.json│
  └─────────────────┘                          └──────┬──────┘
                                                      │
                                               main.py + default.yaml
                                                      │
                                                      ▼
                                           ┌─────────────────────┐
                                           │   Encoder–Decoder   │
                                           │  ResNet-101 + LSTM  │
                                           │  (± Soft Attention) │
                                           └──────────┬──────────┘
                                                      │ Checkpoint tốt nhất
                                                      ▼
                                   ┌──────────────────────────────┐
                                   │   BEST_checkpoint.pth.tar    │
                                   └───────┬──────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                              ▼
          ┌─────────────────┐                        ┌────────────────────┐
          │  evaluate.py    │                        │  02_visualization  │
          │  BLEU-1 ~ 4     │                        │  Attention Heatmap │
          └─────────────────┘                        └────────────────────┘
```

---

## 3. Cấu trúc thư mục

```
AI_Final/
│
├── configs/
│   └── default.yaml                    # Toàn bộ cấu hình: đường dẫn, hyperparameter train/test
│
├── data/
│   └── coco/                           # Ảnh gốc + file HDF5 và WORDMAP.json đã tiền xử lý
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Phân tích và trực quan hóa tập dữ liệu
│   ├── 02_model_visualization.ipynb    # Attention Heatmap & bảng tổng kết điểm BLEU
│   ├── 03_encoder_evaluation.ipynb     # So sánh định lượng HOG vs. ResNet-101 Encoder
│   └── 04_self_collect_evaluation.ipynb # Đánh giá trên tập tự thu thập (VTV24)
│
├── self_collect/                       # Dữ liệu tự thu thập để demo thực tế
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # PyTorch Dataset – nạp dữ liệu vào mô hình
│   │   └── data_reader.py             # Tiền xử lý: tách từ, tạo WORDMAP, đóng gói HDF5
│   ├── models/
│   │   ├── __init__.py
│   │   ├── Resnet101.py               # Encoder (ResNet-101)
│   │   ├── Decoder.py                 # Decoder với Soft Attention
│   │   └── DecodeNoAttention.py       # Decoder Baseline (không có Attention)
│   └── utils/
│       ├── __init__.py
│       └── util.py                    # Hàm phụ trợ: tính toán, lưu/nạp checkpoint
│
├── weights/
│   ├── attention/                      # BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar
│   └── no_attention/                   # BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar
│
├── evaluate.ipynb                      # Beam Search + tính BLEU (Attention)
├── evaluate_baseline.ipynb             # Beam Search + tính BLEU (Baseline)
├── main.py                             # Điểm vào: khởi chạy toàn bộ quá trình huấn luyện
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 4. Cài đặt môi trường

**Yêu cầu:** Python ≥ 3.9, CUDA ≥ 11.3 (khuyến nghị GPU ≥ 8 GB VRAM)

```bash
# 1. Clone repository
git clone https://github.com/hanhtv24/AI_Final.git
cd AI_Final

# 2. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

Các thư viện chính được sử dụng:

| Thư viện | Mục đích |
|---|---|
| `torch`, `torchvision` | Xây dựng và huấn luyện mô hình |
| `h5py` | Đọc/ghi file HDF5 |
| `nltk` | Tính toán điểm BLEU |
| `Pillow` | Xử lý ảnh |
| `tqdm` | Thanh tiến trình huấn luyện |
| `PyYAML` | Đọc file cấu hình |

---

## 5. Luồng thực thi

### Bước 1 – Tiền xử lý dữ liệu

**File phụ trách:** `src/data/data_reader.py`  
**Notebook phân tích:** `notebooks/01_data_exploration.ipynb`

Script thực hiện ba nhiệm vụ chính:

1. Đọc file `dataset_coco.json` (chuẩn phân chia Karpathy Split).
2. Lọc các từ hiếm với ngưỡng `min_word_freq = 5`, xây dựng từ điển và xuất ra `WORDMAP.json`.
3. Đóng gói toàn bộ ảnh thành file `.hdf5` nhằm tối ưu hóa tốc độ I/O trong quá trình huấn luyện.

```bash
python src/data/data_reader.py \
    --data_folder  /path/to/coco \
    --output_folder ./data/coco \
    --min_word_freq 5
```

> **Đầu ra:** `data/coco/TRAIN_IMAGES.hdf5`, `data/coco/WORDMAP.json`

Sau bước này, `src/data/data_loader.py` sẽ bọc các file trên thành `torch.utils.data.Dataset` để truyền vào DataLoader trong quá trình huấn luyện.

---

### Bước 2 – Huấn luyện mô hình

**File phụ trách:** `main.py`  
**Cấu hình tập trung:** `configs/default.yaml`

```yaml
# configs/default.yaml (ví dụ)
model:
  encoder: resnet101
  attention: soft          # soft | none (baseline)
  embed_dim: 512
  decoder_dim: 512
  attention_dim: 512
  dropout: 0.5

training:
  batch_size: 32
  epochs: 20
  encoder_lr: 1.0e-4
  decoder_lr: 4.0e-4
  grad_clip: 5.0
  checkpoint_dir: weights/
```

```bash
python main.py --config configs/default.yaml
```

**Quá trình huấn luyện:**
- Hàm mất mát: `CrossEntropyLoss` (áp dụng trên chuỗi token đầu ra).
- Bộ tối ưu: `Adam` với learning rate riêng biệt cho Encoder và Decoder.
- Sau mỗi epoch, hàm `validate_caption()` tính điểm BLEU-4 trên tập Validation. Checkpoint đạt điểm cao nhất sẽ được lưu vào `weights/attention/` hoặc `weights/no_attention/` với tên file `BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar`.

---

### Bước 3 – Đánh giá định lượng

**File phụ trách:** `evaluate.py` (Attention) · `evaluate_baseline.py` (Baseline)

Bộ đánh giá sử dụng thuật toán **Beam Search** với `beam_size = 5` trên toàn bộ **5.000 ảnh** của tập Test, sau đó tính các chỉ số BLEU-1 đến BLEU-4 bằng thư viện `nltk`.

```bash
# Đánh giá mô hình Attention
python evaluate.py \
    --checkpoint weights/attention/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar \
    --word_map   data/coco/WORDMAP.json \
    --beam_size  5

# Đánh giá mô hình Baseline
python evaluate_baseline.py \
    --checkpoint weights/no_attention/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar \
    --word_map   data/coco/WORDMAP.json \
    --beam_size  5
```

---

### Bước 4 – Trực quan hóa kết quả

**File phụ trách:** `notebooks/02_model_visualization.ipynb`

Notebook thực hiện:
- Nạp checkpoint tốt nhất từ `weights/attention/` và `weights/no_attention/`.
- Sinh câu mô tả cho ảnh bất kỳ từ tập Test.
- Vẽ **Attention Heatmap** – bản đồ nhiệt thể hiện vùng ảnh mà mô hình "chú ý" tại từng bước giải mã.
- Tổng hợp và trình bày bảng điểm số cuối cùng của hai mô hình.

---

### Bước 5 – Đánh giá trên tập tự thu thập

**File phụ trách:** `notebooks/04_self_collect_evaluation.ipynb`

Notebook thực hiện đánh giá định tính và định lượng trên tập ảnh thực tế tự thu thập từ **VTV24** (500 ảnh, có caption ground truth tự đánh nhãn):
- Sinh caption bằng cả hai mô hình (Baseline và Attention) song song.
- Tính **BLEU-4 per ảnh** (smoothed) và **Corpus BLEU-4** toàn tập.
- Xuất bảng so sánh 2 cột kèm caption và điểm số ra file `results/self_collect_results.png`.

```bash
# Chạy từ thư mục notebooks/
jupyter notebook 04_self_collect_evaluation.ipynb
```

---

### Bước 5 – Đánh giá Encoder

**File phụ trách:** `notebooks/03_encoder_evaluation.ipynb`

Notebook thực hiện phân tích so sánh định lượng và định tính giữa hai phương pháp trích xuất đặc trưng ảnh:

- **HOG (Histogram of Oriented Gradients):** Trích xuất đặc trưng hình thái học bề mặt (góc cạnh, gradient cường độ pixel).
- **ResNet-101:** Trích xuất đặc trưng ngữ nghĩa sâu (deep semantic features) từ lớp `conv5_x`.

Kết quả thực nghiệm cho thấy ResNet-101 có biên độ phân tách ngữ nghĩa vượt trội, là cơ sở khoa học để chọn làm Encoder cho toàn bộ hệ thống Image Captioning.

---

## 6. Kết quả thực nghiệm

Đánh giá thực hiện trên tập **Test MS COCO (5.000 ảnh)** · Beam Size = 5 · GPU: NVIDIA GeForce RTX 3060.

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|:---:|:---:|:---:|:---:|
| Baseline — ResNet-101 + LSTM | 66.87% | 49.33% | 36.49% | 26.80% |
| **Proposed — ResNet-101 + LSTM + Soft Attention** | **70.93%** | **53.69%** | **40.54%** | **30.27%** |
| Show, Attend & Tell — Xu et al. (2015) *(tham khảo)* | 70.7% | 49.2% | 34.4% | 24.3% |

**Nhận xét:** Mô hình Proposed vượt Baseline trên toàn bộ 4 chỉ số BLEU, đồng thời vượt kết quả công bố của paper gốc *"Show, Attend and Tell"* (Xu et al., ICML 2015) ở tất cả các chỉ số. BLEU-4 cải thiện **+3.47%** so với Baseline và **+5.97%** so với paper gốc.

Mẫu Attention Heatmap minh họa khả năng của mô hình trong việc định vị đúng vùng đối tượng khi sinh từng từ được trình bày chi tiết trong `notebooks/02_model_visualization.ipynb`.

---

## 7. Tham khảo

1. Xu, K. et al. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML 2015.
2. Vinyals, O. et al. (2015). *Show and Tell: A Neural Image Caption Generator.* CVPR 2015.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
4. Karpathy, A. & Fei-Fei, L. (2015). *Deep Visual-Semantic Alignments for Image Description.* CVPR 2015.
5. Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV 2014.

---

*Đồ án thuộc chương trình đào tạo – Khoa Công nghệ Thông tin.*