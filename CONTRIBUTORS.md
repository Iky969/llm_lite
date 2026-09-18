# Kontributor & Catatan Pengembangan (Contributors) 👥

Proyek **LLM Lite** dikembangkan secara kolaboratif melalui pendekatan *Human-in-the-Loop AI Pair Programming*, memadukan arahan serta dataset awal dari kreator proyek dengan serangkaian iterasi teknis dari berbagai model AI terdepan.

Dokumen ini mencatat peran serta kontribusi spesifik dari setiap pihak yang terlibat dalam perjalanan pengembangan LLM Lite.

---

## 🧑‍💻 Project Creator & Lead

### **Rzy ([@Iky969](https://github.com/Iky969))**
- **Peran**: Inisiator, Kurator Proyek, & Pengarah Pengembangan.
- **Kontribusi**:
  - Ideasi awal dan fondasi kode implementasi pertama (Mini Language Model / LSTM).
  - Penyiapan korpus dataset sampel teks pengenalan AI & komputasi.
  - Pengarahan arsitektur, penetapan visi model, pengujian lingkungan, dan kurasi dataset.
  - Pengambilan keputusan desain (skalabilitas 128M parameter, fitur isolasi API, lisensi, dan publikasi).

---

## 🤖 AI Model Collaborators

Pengembangan arsitektur dan fitur LLM Lite telah melalui beberapa fase iterasi yang dibantu oleh model AI berikut:

### 1. **Anthropic Claude (Claude 3.5 Sonnet & Claude 3 Opus)**
*Fase Fondasi Arsitektur & Modernisasi Transformer*
- **Kontribusi**:
  - Merombak arsitektur awal (LSTM/RNN) menjadi **Decoder-Only Transformer modern (GPT-style)** murni PyTorch.
  - Menetapkan skala parameter awal ke **~25.5M parameter** (`d_model=512`, `num_layers=8`, `nhead=8`, `dim_feedforward=2048`).
  - Mengimplementasikan mekanisme *Causal Multi-Head Self-Attention*, *Pre-LayerNorm*, *Residual Connections*, dan *Weight Tying*.
  - Mengintegrasikan algoritma sampling teks **Top-k + Top-p (Nucleus) Sampling** dengan *Temperature Scaling*.
  - Menerapkan evaluasi inferensi berbasis `torch.inference_mode()` dan sistem penanganan karakter *Out-of-Vocabulary (OOV) fallback*.
  - Menyiapkan sistem penyimpanan checkpoint dengan validasi kompatibilitas arsitektur dasar dan smoke-test awal.
  - Menyusun panduan konteks proyek awal di `p.md`.

---

### 2. **Google DeepMind Antigravity (Gemini)**
*Fase Peningkatan Skala, Fitur Training Modern, Modularitas, & Ekosistem CLI*
- **Kontribusi**:
  - **Fitur Pelatihan Lanjutan (*Advanced Training Suite*)**:
    - Integrasi **Automatic Mixed Precision (AMP)** (`torch.amp.autocast` + `torch.amp.GradScaler`) untuk efisiensi komputasi GPU (CUDA/MPS) dan fallback aman CPU.
    - Menambahkan mekanisme **Gradient Accumulation** (`--grad-accum-steps`) guna mensimulasikan batch size efektif yang lebih besar tanpa lonjakan memori VRAM.
    - Mengintegrasikan **Linear Warmup LR Scheduler** yang disambung dengan *Cosine Annealing* (`--warmup-epochs`).
  - **Modularitas Tokenizer & Dataset Eksternal**:
    - Memisahkan dan merefaktor tokenizer menjadi kelas mandiri [`CharTokenizer`](llm_lite.py) dengan fitur ekspor/impor kosakata format JSON (`--save-vocab`, `--load-vocab`).
    - Menambahkan dukungan pelatihan menggunakan file dataset teks eksternal kustom via flag CLI `--data <path>`.
  - **Mode Interaktif Dinamis (CLI REPL)**:
    - Membangun antarmuka interaktif terminal (`--interactive` / `-i`) untuk pengujian prompt berkelanjutan tanpa memuat ulang model dari disk.
    - Menambahkan perintah konfigurasi dinamis saat runtime (`:temp`, `:len`, `:top_k`, `:top_p`, `:info`).
  - **Peningkatan Skala Model ke ~128M Parameter**:
    - Menaikkan parameter default ke **128.395.008 parameter (~128.4M)** dengan arsitektur penuh setara GPT-2 Small (`d_model=768`, `layers=18`, `heads=12`, `dim_feedforward=3072`, `max_seq_len=1024`).
    - Merancang sistem preset model dinamis (`--size {25m, 50m, 85m, 128m}`) beserta argumen override dimensi individu.
  - **Agen Pengetahuan Terisolasi ([`knowledge_agent.py`](knowledge_agent.py))**:
    - Menciptakan modul mandiri *zero-dependency* (berbasis library standar `urllib`) untuk mengambil wawasan dari Teacher LLM API (OpenAI-compatible / Groq / Gemini / OpenRouter) dan otomatis mencatatnya menjadi dataset pelatihan lokal, menjaga `llm_lite.py` tetap bersih dan aman dari kebocoran kredensial.
  - **Repositori & Dokumentasi Standar Industri**:
    - Penataan repositori Git, pembuatan `.gitignore` komprehensif, penyusunan Lisensi MIT resmi, dan penulisan dokumentasi [README.md](README.md) profesional.
    - Penyiapan alur sinkronisasi cloud ke Google Drive menggunakan `rclone`.

---

## 📜 Ringkasan Jejak Pengembangan (Timeline)

```
[Inisiasi]
  └── Rzy: Konsep awal, ideasi mini LLM, penyiapan dataset sampel & implementasi dasar (LSTM).
        │
[Fase 1 - Rekayasa Arsitektur Transformer]
  └── Claude (Sonnet / Opus): Transisi ke Decoder-Only Transformer, bobot 25.5M, Top-k/Top-p, OOV fallback.
        │
[Fase 2 - Skalabilitas, Fitur Training, & Ekosistem CLI]
  └── Antigravity (Gemini): Skala 128M parameter, sistem preset dinamis, AMP & Grad Accumulation,
      Tokenizer modular, REPL interaktif, Knowledge Agent terisolasi, README & Lisensi MIT.
```
