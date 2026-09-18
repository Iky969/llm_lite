# Konteks Proyek: llm_lite

File utama: llm_lite.py — language model Decoder-Only Transformer (GPT-style), sudah melalui beberapa
iterasi pengembangan oleh berbagai model AI (Opus, Sonnet, Gemini/Antigravity). Kode BUKAN raw,
sudah punya arsitektur & fitur yang mapan. Baca dulu isi file sebelum mengubah apa pun.

## Status Terkini (jangan diulang/dirombak tanpa alasan kuat)
- **Parameter**: 25.499.136 (~25.5M) — JANGAN dikurangi karena alasan compute environment ini (lihat bagian Environment).
- **Arsitektur**: Decoder-Only Transformer (`d_model=512`, `nhead=8`, `num_layers=8`, `dim_feedforward=2048`, `max_seq_len=512`, `dropout=0.1`, weight tying).
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
- **Smoke Test**: Terverifikasi lolos via CLI `--smoke-test`.
- **Status Training Bobot**: Checkpoint tersimpan di `checkpoint.pt` (~25.5M parameter). Training penuh: 120 epoch di GPU.

## Environment
- Codespace ini hanya untuk build/development, BUKAN target eksekusi final.
- Resource terbatas (2 CPU core, tanpa GPU) — training lambat di CPU itu normal, BUKAN alasan mengurangi parameter.
- Target eksekusi/inference: GPU (utama), CPU (fallback).

## Perubahan Terakhir (Selesai)
1. **Opsi 4**: Fitur training lanjutan — Gradient accumulation, Linear LR warmup, dan Mixed Precision (`torch.amp`).
2. **Opsi 2**: Dukungan dataset eksternal (`--data`) dan pemisahan kelas modular `CharTokenizer` dengan penyimpanan/pemuatan kosakata (`--save-vocab`, `--load-vocab`).
3. **Opsi 3**: Mode interaktif / CLI REPL (`--interactive` / `-i`) dengan runtime controls.
4. **Opsi 5**: Merapikan repository — menghapus file usang `llm_lite`, menambahkan `.gitignore` (mengabaikan `checkpoint.pt` 306MB agar tidak ditolak limit GitHub 100MB), dan menyiapkan commit yang rapi.

## Tugas Sekarang
- Seluruh tugas terkini (Opsi 2, Opsi 3, Opsi 4, Opsi 5) telah selesai dikerjakan dan terverifikasi.
- [isi: fitur/perbaikan apa yang mau dikerjakan berikutnya bila ada]
