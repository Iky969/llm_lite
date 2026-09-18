# Konteks Proyek: llm_lite

File utama: llm_lite.py — language model Decoder-Only Transformer (GPT-style), sudah melalui beberapa
iterasi pengembangan oleh berbagai model AI (Opus, Sonnet, Gemini/Antigravity). Kode BUKAN raw,
sudah punya arsitektur & fitur yang mapan. Baca dulu isi file sebelum mengubah apa pun.

## Status Terkini (jangan diulang/dirombak tanpa alasan kuat)
- **Parameter Default**: **128.395.008 (~128.4M)** — skala penuh arsitektur GPT-2 Small.
- **Sistem Preset Model Dinamis**:
  - `128m` (default): `d_model=768, nhead=12, num_layers=18, dim_feedforward=3072, max_seq_len=1024` (~128.4M params)
  - `85m`: `d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, max_seq_len=512` (~85.5M params)
  - `50m`: `d_model=640, nhead=10, num_layers=10, dim_feedforward=2560, max_seq_len=512` (~49.6M params)
  - `25m`: `d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, max_seq_len=512` (~25.5M params)
  - Pengguna dapat mengganti preset dengan flag `--size {25m,50m,85m,128m}` atau melakukan override spesifik (`--d_model`, `--num_layers`, dll).
- **Arsitektur**: Decoder-Only Transformer (Causal Multihead Self-Attention, GELU Feed-Forward, LayerNorm, Residual Connections, Learnable Positional Embedding, dan Weight Tying antara embedding dan lm_head).
- **Fitur Sampling & Inferensi**: Top-k + Top-p (Nucleus) sampling, temperature scaling, `torch.inference_mode()`, OOV (out-of-vocabulary) fallback aman.
- **Mode Interaktif**: CLI REPL (`--interactive` / `-i`) untuk prompt-and-response berkelanjutan tanpa reload model, dengan kontrol runtime (`:temp`, `:len`, `:top_k`, `:top_p`, `:info`).
- **Dataset & Tokenizer Modular**:
  - Kelas `CharTokenizer` mandiri dengan ekspor/impor JSON (`--save-vocab`, `--load-vocab`).
  - Dukungan dataset teks eksternal (`--data path/to/file.txt`), fallback ke dataset bawaan jika tidak diberikan.
- **Fitur Training Lanjutan**:
  - Gradient Accumulation (`--grad-accum-steps`) untuk simulasi batch size efektif lebih besar.
  - Linear Warmup + Cosine Annealing LR Scheduler (`--warmup-epochs`).
  - Automatic Mixed Precision (`torch.amp.autocast` + `GradScaler`, `--amp` / `--no-amp`) untuk GPU (CUDA/MPS) dan fallback aman di CPU.
- **Checkpoint & Metadata**: Simpan/muat metadata lengkap (`epoch`, `val_loss`, `best_val_loss`, `vocab_size`, `chars`, `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `max_seq_len`, dan `scaler_state_dict`), serta validasi kompatibilitas arsitektur saat load.
- **Smoke Test**: Terverifikasi lolos via CLI `--smoke-test` untuk preset 128M dan 25M.
- **Status Checkpoint Tersedia**:
  - `checkpoint.pt`: Checkpoint model 128.4M parameter.
  - `checkpoint_25m.pt`: Checkpoint cadangan model 25.5M parameter.

## Environment
- Codespace ini hanya untuk build/development, BUKAN target eksekusi final.
- Resource terbatas (2 CPU core, tanpa GPU) — training lambat di CPU itu normal, BUKAN alasan mengurangi parameter.
- Target eksekusi/inference: GPU (utama), CPU (fallback). Training penuh: 120 epoch di GPU (misal via Google Colab).

## Perubahan Terakhir (Selesai)
1. **Peningkatan Skala Model ke ~128M Parameter**:
   - Menetapkan preset arsitektur GPT-2 Small (`d_model=768, num_layers=18, nhead=12, dim_feedforward=3072, max_seq_len=1024`) dengan total parameter **128.395.008**.
   - Menambahkan sistem `--size / --preset` dinamis (25m, 50m, 85m, 128m) dan opsi override dimensi.
2. **Fitur Training Lanjutan**: Gradient accumulation, Linear LR warmup, dan Mixed Precision (`torch.amp`).
3. **Dataset & Tokenizer Modular**: Dukungan dataset eksternal (`--data`) dan kelas `CharTokenizer` dengan simpan/muat kosakata (`--save-vocab`, `--load-vocab`).
4. **Mode Interaktif / REPL CLI**: `--interactive` / `-i` dengan perintah runtime.
5. **Git & Repo Hygiene**: File usang dibersihkan, `.gitignore` dipasang, commit terstruktur.

## Tugas Sekarang
- Model 128M dan sistem preset telah selesai diimplementasikan, diverifikasi, dan siap digunakan.
