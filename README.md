# LLM Lite BETA

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: GPU & CPU](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-orange.svg)](https://pytorch.org/)

**LLM Lite** adalah implementasi *Decoder-Only Transformer Language Model* (arsitektur GPT-style) dari nol menggunakan **PyTorch murni**. Dirancang dengan fokus pada kesederhanaan (*minimalist*), skalabilitas arsitektur yang fleksibel (25M hingga 128M parameter), fitur pelatihan modern, serta mode pengujian interaktif yang intuitif.

---

## 🌟 Fitur Utama

- **Arsitektur Transformer Modern**:
  - *Causal Multi-Head Self-Attention* dengan mask kausal triangular.
  - *Feed-Forward Network (FFN)* dengan aktivasi **GELU**.
  - *Pre-Layer Normalization* dan *Residual Connections* untuk stabilitas gradien.
  - *Weight Tying* antara token embedding dan linear projection head.
  - *Learnable Positional Embeddings*.

- **Skalabilitas Arsitektur Dinamis (Preset 25M – 128M)**:
  - Tersedia preset bawaan dari **25M** (ringan untuk eksperimen CPU lokal) hingga **128M** (skala GPT-2 Small penuh).
  - Fleksibilitas override parameter arsitektur langsung via argumen CLI (`--d_model`, `--num_layers`, dll).

- **Pipeline Pelatihan Canggih (*Advanced Training Suite*)**:
  - **Automatic Mixed Precision (AMP)** via `torch.amp.autocast` dan `GradScaler` (akselerasi CUDA bfloat16/float16, MPS, dan fallback aman CPU).
  - **Gradient Accumulation** (`--grad-accum-steps`) untuk mensimulasikan batch size efektif besar tanpa lonjakan konsumsi VRAM.
  - **Learning Rate Warmup + Cosine Annealing Scheduler** (`--warmup-epochs`) untuk konvergensi pelatihan yang mulus.

- **Inferensi & Pembangkitan Teks Cerdas**:
  - Pembangkitan dengan **Top-k + Top-p (Nucleus) Sampling** dan **Temperature Scaling**.
  - Evaluasi inferensi tanpa komputasi gradien via `torch.inference_mode()`.
  - Fallback aman untuk karakter tak dikenal (*Out-of-Vocabulary / OOV*).

- **Mode Interaktif (CLI REPL)**:
  - Antarmuka terminal interaktif untuk menguji prompt teks berulang kali tanpa memuat ulang model dari disk.
  - Kontrol konfigurasi dinamis saat runtime (`:temp`, `:len`, `:top_k`, `:top_p`, `:info`).

- **Modul Tokenizer Mandiri & Dataset Eksternal**:
  - Kelas `CharTokenizer` mandiri dengan kemampuan ekspor/impor kosakata format JSON (`--save-vocab`, `--load-vocab`).
  - Dukungan langsung untuk melatih model menggunakan file teks eksternal (`--data path/to/dataset.txt`).

- **Agen Pengumpul Pengetahuan Terisolasi (`knowledge_agent.py`)**:
  - Modul independen *zero-dependency* untuk mengekstrak dan mengumpulkan data pengetahuan dari *Teacher LLM API* (OpenAI-compatible / Groq / Gemini / OpenRouter / Ollama) ke dalam format dataset lokal.

---

## 📐 Spesifikasi Preset Arsitektur

| Preset | Total Parameter | $d_{model}$ | Heads | Layers | $d_{ffn}$ | Max Seq Len | Rekomendasi Target |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`25m`** | **25.499.136** (~25.5M) | 512 | 8 | 8 | 2048 | 512 | Eksperimen lokal cepat / CPU |
| **`50m`** | **49.585.280** (~49.6M) | 640 | 10 | 10 | 2560 | 512 | Dataset skala menengah |
| **`85m`** | **85.474.560** (~85.5M) | 768 | 12 | 12 | 3072 | 512 | Arsitektur standar GPT-2 |
| **`128m`** *(default)* | **128.395.008** (~128.4M) | 768 | 12 | 18 | 3072 | 1024 | Kapasitas penuh (GPU recommended) |

---

## 🛠️ Instalasi & Persyaratan

### Persyaratan Sistem
- Python 3.10 atau versi lebih baru
- PyTorch 2.0+ (disarankan versi dengan dukungan CUDA jika menggunakan GPU)

### Langkah Instalasi
```bash
# 1. Clone repository
git clone https://github.com/Iky969/llm_lite.git
cd llm_lite

# 2. Pasang dependensi PyTorch
pip install torch
```

---

## 🚀 Panduan Penggunaan

### 1. Uji Coba Cepat (*Smoke Test*)
Memeriksa integritas model dan memuat checkpoint untuk memastikan arsitektur berjalan lancar:
```bash
python llm_lite.py --smoke-test
```
*Untuk menguji preset lain:*
```bash
python llm_lite.py --smoke-test --size 25m --checkpoint checkpoint_25m.pt
```

### 2. Mode Interaktif (CLI REPL)
Mengobrol atau menguji prompt langsung di terminal:
```bash
python llm_lite.py --interactive
```
*Perintah di dalam REPL:*
- Ketik prompt apa pun lalu tekan **Enter**.
- `:temp 0.8` $\rightarrow$ Mengubah suhu sampling.
- `:len 300` $\rightarrow$ Mengubah panjang karakter generasi.
- `:info` $\rightarrow$ Menampilkan konfigurasi yang sedang aktif.
- `exit` / `quit` $\rightarrow$ Keluar dari mode interaktif.

### 3. Melatih Model (*Training*)
```bash
# Melatih model default (128M) dengan Mixed Precision dan Gradient Accumulation
python llm_lite.py --train --epochs 120 --batch_size 16 --grad-accum-steps 2 --amp

# Melatih menggunakan dataset eksternal kustom (.txt)
python llm_lite.py --train --data path/to/dataset.txt --epochs 50 --amp
```

### 4. Menghasilkan Teks (*Inference / Generate*)
```bash
python llm_lite.py --generate --prompt "Kecerdasan buatan di masa depan" --length 300 --temperature 0.7
```

### 5. Mengumpulkan Pengetahuan Tambahan (`knowledge_agent.py`)
Skrip terisolasi untuk mengumpulkan wawasan dari API model guru eksternal dan menyimpannya menjadi dataset:
```bash
# Mode interaktif pengumpul pengetahuan (kompatibel Groq, OpenRouter, OpenAI, dll.)
python knowledge_agent.py --url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile

# Menanyakan satu topik langsung
python knowledge_agent.py --ask "Jelaskan prinsip kerja jaringan saraf tiruan"
```
Semua jawaban akan otomatis tersimpan rapi ke dalam `knowledge_dataset.txt` yang siap dilatihkan kembali ke `llm_lite.py`.

---

## 💾 Model Weights & Checkpoints

> [!NOTE]
> File bobot model pre-trained (*checkpoints*) berukuran besar (~100 MB hingga ~500 MB) dan diabaikan dari repository Git utama untuk efisiensi penyimpanan. Tautan unduhan resmi bobot checkpoint akan dipublikasikan secara terpisah saat rilis stabil tersedia.

---

## 🗺️ Roadmap Pengembangan

- [x] Arsitektur Transformer GPT-style dari nol
- [x] Sistem preset ukuran dinamis (25M, 50M, 85M, 128M)
- [x] Training lanjutan: Automatic Mixed Precision (AMP), Warmup, Grad Accumulation
- [x] CLI REPL interaktif dengan runtime controls
- [x] Modul mandiri Knowledge Distillation Agent
- [ ] Implementasi Subword Tokenizer (Byte-Pair Encoding / BPE)
- [ ] Template notebook resmi untuk pelatihan di Google Colab GPU (T4/A100)
- [ ] KV-Cache untuk inferensi generasi teks super cepat

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](LICENSE). Bebas digunakan, dimodifikasi, dan didistribusikan untuk keperluan pembelajaran, riset, maupun komersial.
