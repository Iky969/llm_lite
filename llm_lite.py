"""
LLM Lite - Decoder-Only Transformer Language Model
Arsitektur modern dengan Multi-Head Self-Attention, FFN, LayerNorm, dan Residual Connections.
Mendukung CPU / GPU (CUDA / MPS) secara otomatis.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ==========================================
# 1. PERSIAPAN DATASET DAN TOKENISASI
# ==========================================

# Dataset teks sampel: cuplikan ensiklopedia / pengenalan AI & komputasi
raw_text = """\
Kecerdasan buatan atau AI adalah simulasi kecerdasan manusia yang dimodelkan dalam mesin.
Mesin ini diprogram untuk berpikir seperti manusia dan meniru tindakan mereka.
Karakteristik ideal dari kecerdasan buatan adalah kemampuannya untuk merasionalisasi dan mengambil tindakan
yang memiliki peluang terbaik untuk mencapai tujuan tertentu.
Subkumpulan dari kecerdasan buatan adalah pembelajaran mesin atau machine learning.
Pembelajaran mesin berfokus pada pengembangan program komputer yang dapat mengakses data dan menggunakannya
untuk belajar mandiri tanpa bantuan aturan manusia secara terus menerus.
Proses pembelajaran dimulai dengan pengamatan atau data, seperti contoh, pengalaman langsung, atau instruksi,
untuk mencari pola dalam data dan membuat keputusan yang lebih baik di masa depan.
Jaringan saraf tiruan terinspirasi oleh jaringan saraf biologis otak manusia.
Dengan memperhatikan relasi antar karakter dan kata, model bahasa dapat memprediksi kata berikutnya secara tepat.
"""


class CharTokenizer:
    """
    Tokenizer berbasis karakter dengan dukungan:
    - OOV (out-of-vocabulary) fallback aman
    - Encode & Decode
    - Ekspor & Impor kosakata ke/dari file JSON (--save-vocab / --load-vocab)
    """

    def __init__(self, chars: Optional[List[str]] = None) -> None:
        self.chars: List[str] = sorted(list(set(chars))) if chars is not None else []
        self._build_mappings()

    def _build_mappings(self) -> None:
        self.char_to_idx: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def train_from_text(self, text: str) -> None:
        """Membangun kosakata karakter unik dari teks input."""
        self.chars = sorted(list(set(text)))
        self._build_mappings()

    def encode(
        self,
        text: str,
        report_oov: bool = True,
    ) -> Tuple[List[int], List[str]]:
        """
        Mengonversi teks menjadi daftar indeks token dengan OOV fallback.
        Mengembalikan (indices, oov_chars).
        """
        indices: List[int] = []
        oov_chars: List[str] = []
        for ch in text:
            if ch in self.char_to_idx:
                indices.append(self.char_to_idx[ch])
            else:
                if ch not in oov_chars:
                    oov_chars.append(ch)
        if oov_chars and report_oov:
            print(f"  [⚠] Karakter OOV diabaikan: {oov_chars}")
        return indices, oov_chars

    def decode(self, indices: List[int]) -> str:
        """Mengonversi daftar indeks token kembali menjadi string teks."""
        return "".join(self.idx_to_char.get(i, "") for i in indices)

    def save(self, filepath: str) -> None:
        """Menyimpan konfigurasi kosakata ke file JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "chars": self.chars,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [✓] Vocab tersimpan ke: {filepath} ({self.vocab_size} karakter)")

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Memuat konfigurasi kosakata dari file JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File vocab '{filepath}' tidak ditemukan.")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data.get("chars", [])
        tokenizer = cls(chars)
        print(f"  [✓] Vocab dimuat dari: {filepath} ({tokenizer.vocab_size} karakter)")
        return tokenizer


def load_dataset(file_path: Optional[str] = None) -> str:
    """
    Memuat dataset teks dari file eksternal (UTF-8).
    Jika file_path tidak dispesifikasikan, menggunakan dataset default (raw_text).
    """
    if file_path is None:
        return raw_text

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File dataset eksternal tidak ditemukan: '{file_path}'")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        raise ValueError(f"File dataset eksternal '{file_path}' kosong.")

    print(f"  [✓] Dataset eksternal dimuat: '{file_path}' ({len(content):,} karakter)")
    return content


# Tokenizer bawaan berdasarkan dataset sampel
default_tokenizer = CharTokenizer(sorted(list(set(raw_text))))
chars: List[str] = default_tokenizer.chars
vocab_size: int = default_tokenizer.vocab_size
char_to_idx: Dict[str, int] = default_tokenizer.char_to_idx
idx_to_char: Dict[int, str] = default_tokenizer.idx_to_char

# Konversi seluruh teks menjadi indeks angka (kompatibilitas default)
text_as_int: List[int] = default_tokenizer.encode(raw_text, report_oov=False)[0]


def print_vocab_info(
    tokenizer: Optional[CharTokenizer] = None,
    text_len: Optional[int] = None,
) -> None:
    """Mencetak informasi dataset dan kosakata."""
    tok = tokenizer or default_tokenizer
    total_len = text_len if text_len is not None else len(raw_text)
    print(f"Total Karakter dalam Teks  : {total_len:,}")
    print(f"Jumlah Karakter Unik (Vocab): {tok.vocab_size}")


def encode_with_oov_fallback(
    text: str,
    char_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[int], List[str]]:
    """
    Mengubah teks menjadi daftar indeks token, dengan penanganan
    karakter di luar kosakata (out-of-vocabulary / OOV).
    Fungsi utilitas untuk kompatibilitas ke belakang.
    """
    if char_map is None or char_map == char_to_idx:
        return default_tokenizer.encode(text)

    indices: List[int] = []
    oov_chars: List[str] = []
    for ch in text:
        if ch in char_map:
            indices.append(char_map[ch])
        else:
            if ch not in oov_chars:
                oov_chars.append(ch)
    if oov_chars:
        print(f"  [⚠] Karakter OOV diabaikan: {oov_chars}")
    return indices, oov_chars


def count_parameters(model: nn.Module) -> int:
    """Menghitung total parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def print_model_summary(model: nn.Module) -> None:
    """Mencetak ringkasan arsitektur dan jumlah parameter model."""
    total = count_parameters(model)
    print("\n" + "=" * 55)
    print("  MODEL SUMMARY")
    print("=" * 55)
    print(f"  Arsitektur     : Decoder-Only Transformer")
    print(f"  d_model        : {model.d_model}")
    print(f"  nhead          : {model.nhead}")
    print(f"  num_layers     : {model.num_layers}")
    print(f"  dim_feedforward: {model.dim_feedforward}")
    print(f"  max_seq_len    : {model.max_seq_len}")
    print(f"  Vocab Size     : {model.vocab_size}")
    print(f"  Trainable Params: {total:,} ({total / 1e6:.3f} M)")
    print("=" * 55 + "\n")


def create_sequences(
    data: List[int],
    seq_length: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mengkonversi list token menjadi pasangan (input, target) tensor.
    Target adalah pergeseran 1 token ke depan dari input.
    """
    inputs, targets = [], []
    for i in range(len(data) - seq_length):
        inputs.append(data[i : i + seq_length])
        targets.append(data[i + 1 : i + seq_length + 1])

    X = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return X, y


def create_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 16,
    shuffle: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Membagi tensor (X, y) menjadi list of batches."""
    n = len(X)
    indices = list(range(n))
    if shuffle:
        random.seed(42)
        random.shuffle(indices)

    n_batches = n // batch_size
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(n_batches):
        idx = indices[i * batch_size : (i + 1) * batch_size]
        batches.append((X[idx], y[idx]))
    return batches


def split_train_val(
    data: List[int],
    seq_length: int = 64,
    batch_size: int = 16,
    val_ratio: float = 0.2,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Membagi data menjadi set pelatihan (train) dan validasi (val) dengan rasio 80/20.
    Mengembalikan tuple (train_batches, val_batches).
    """
    X, y = create_sequences(data, seq_length)
    n = len(X)
    split = int(n * (1 - val_ratio))

    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_batches = create_batches(X_train, y_train, batch_size, shuffle=True)
    val_batches = create_batches(X_val, y_val, batch_size, shuffle=False)
    return train_batches, val_batches


# ==========================================
# 2. ARSITEKTUR MODEL: DECODER-ONLY TRANSFORMER
# ==========================================

class TransformerBlock(nn.Module):
    """
    Satu blok Decoder-Only Transformer:
    - Causal Multi-Head Self-Attention
    - Residual Connection + LayerNorm
    - Feed-Forward Network (FFN)
    - Residual Connection + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # --- Causal Self-Attention dengan Residual Connection ---
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = residual + self.dropout(attn_out)

        # --- Feed-Forward Network dengan Residual Connection ---
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class LLMLite(nn.Module):
    """
    Decoder-Only Transformer Language Model (GPT-style).
    Parameter default: d_model=512, nhead=8, num_layers=8, dim_feedforward=2048.
    Target utama GPU (CUDA/MPS), CPU sebagai fallback.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional Embedding (learnable)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.drop_emb = nn.Dropout(dropout)

        # Stack Transformer Blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        # Final LayerNorm sebelum head
        self.norm_final = nn.LayerNorm(d_model)

        # Language Model Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: lm_head berbagi bobot dengan token_emb
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Inisialisasi bobot model agar pelatihan lebih stabil."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Membuat causal (upper-triangular) attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor [batch_size, seq_len]
        Returns:
            logits: FloatTensor [batch_size, seq_len, vocab_size]
        """
        B, T = x.shape
        assert T <= self.max_seq_len, f"Sequence length {T} melebihi max_seq_len {self.max_seq_len}"

        device = x.device
        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        # Gabungkan token embedding dan positional embedding
        x = self.drop_emb(self.token_emb(x) + self.pos_emb(positions))

        # Causal attention mask
        causal_mask = self._make_causal_mask(T, device)

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits


# ==========================================
# 3. CHECKPOINT: SIMPAN & MUAT MODEL TERBAIK
#    (dengan metadata lengkap untuk resume tanpa konflik)
# ==========================================

CHECKPOINT_PATH = "checkpoint.pt"


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str = CHECKPOINT_PATH,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    """
    Menyimpan checkpoint model saat mencapai validation loss terbaik.
    Menyertakan metadata lengkap: arsitektur, vocab, statistik pelatihan,
    dan state GradScaler opsional agar model dapat di-resume tanpa konflik.
    """
    ckpt_data = {
        # --- Statistik Pelatihan ---
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": val_loss,
        # --- State Dicts ---
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # --- Konfigurasi Vocab ---
        "vocab_size": model.vocab_size,
        "chars": chars,
        # --- Konfigurasi Arsitektur ---
        "d_model": model.d_model,
        "nhead": model.nhead,
        "num_layers": model.num_layers,
        "dim_feedforward": model.dim_feedforward,
        "max_seq_len": model.max_seq_len,
    }
    if scaler is not None and scaler.is_enabled():
        ckpt_data["scaler_state_dict"] = scaler.state_dict()

    torch.save(ckpt_data, path)
    print(f"  [✓] Checkpoint tersimpan (epoch {epoch}, val_loss={val_loss:.4f})")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    path: str = CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    """
    Memuat checkpoint dari disk ke model (dan optimizer opsional).
    Melakukan validasi kompatibilitas arsitektur dan vocab sebelum memuat.
    Mengembalikan epoch terakhir yang tersimpan.
    """
    if not os.path.exists(path):
        print(f"[!] Checkpoint '{path}' tidak ditemukan. Mulai dari awal.")
        return 0

    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)

    # --- Validasi kompatibilitas arsitektur ---
    ckpt_vocab = ckpt.get("vocab_size")
    ckpt_d_model = ckpt.get("d_model")
    ckpt_nhead = ckpt.get("nhead")
    ckpt_num_layers = ckpt.get("num_layers")

    mismatches: List[str] = []
    if ckpt_vocab is not None and ckpt_vocab != model.vocab_size:
        mismatches.append(f"vocab_size: checkpoint={ckpt_vocab}, model={model.vocab_size}")
    if ckpt_d_model is not None and ckpt_d_model != model.d_model:
        mismatches.append(f"d_model: checkpoint={ckpt_d_model}, model={model.d_model}")
    if ckpt_nhead is not None and ckpt_nhead != model.nhead:
        mismatches.append(f"nhead: checkpoint={ckpt_nhead}, model={model.nhead}")
    if ckpt_num_layers is not None and ckpt_num_layers != model.num_layers:
        mismatches.append(f"num_layers: checkpoint={ckpt_num_layers}, model={model.num_layers}")

    if mismatches:
        print("[✗] Checkpoint TIDAK KOMPATIBEL dengan model saat ini:")
        for m in mismatches:
            print(f"    - {m}")
        print("[!] Silakan latih ulang model atau gunakan checkpoint yang sesuai.")
        return 0

    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        print(f"[✗] Gagal memuat state_dict: {e}")
        print("[!] Checkpoint tidak kompatibel dengan arsitektur model saat ini.")
        print("    Silakan latih ulang model atau gunakan checkpoint yang sesuai.")
        return 0

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (RuntimeError, ValueError):
            print("  [⚠] Optimizer state tidak kompatibel, direset.")

    if scaler is not None and scaler.is_enabled() and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except (RuntimeError, ValueError):
            print("  [⚠] Scaler state tidak kompatibel, direset.")

    epoch = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    best_val = ckpt.get("best_val_loss", val_loss)
    print(f"[✓] Checkpoint dimuat: epoch={epoch}, val_loss={val_loss:.4f}, best_val_loss={best_val:.4f}")
    return epoch


# ==========================================
# 4. PROSEDUR PELATIHAN (TRAINING LOOP)
# ==========================================

def setup_amp(
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[bool, torch.dtype, torch.amp.GradScaler]:
    """
    Menyiapkan konfigurasi Automatic Mixed Precision (AMP) berdasarkan perangkat.
    - CUDA: bfloat16 jika didukung Ampere+, float16 untuk arsitektur sebelumnya dengan GradScaler.
    - CPU: bfloat16 (jika diaktifkan), GradScaler dinonaktifkan.
    - MPS: float16, GradScaler dinonaktifkan.
    """
    device_type = device.type
    if not use_amp:
        return False, torch.float32, torch.amp.GradScaler(device_type, enabled=False)

    if device_type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        return True, amp_dtype, scaler
    elif device_type == "cpu":
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return True, amp_dtype, scaler
    elif device_type == "mps":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("mps", enabled=False)
        return True, amp_dtype, scaler
    else:
        return False, torch.float32, torch.amp.GradScaler("cpu", enabled=False)


def evaluate(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """Menghitung rata-rata validation loss tanpa gradient (inference_mode)."""
    model.eval()
    total_loss = 0.0
    device_type = device.type
    with torch.inference_mode():
        for X_batch, y_batch in batches:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(X_batch)
                loss = criterion(logits.view(-1, model.vocab_size), y_batch.view(-1))
            total_loss += loss.item()
    return total_loss / len(batches)


def train_model(
    model: nn.Module,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 120,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    val_interval: int = 10,
    grad_accum_steps: int = 1,
    warmup_epochs: int = 5,
    use_amp: bool = True,
) -> None:
    """
    Loop pelatihan utama dengan:
    - AdamW optimizer
    - Linear Warmup + Cosine Annealing LR Scheduler
    - Automatic Mixed Precision (torch.amp) & GradScaler
    - Gradient Accumulation untuk batch size efektif lebih besar
    - Evaluasi validation loss berkala
    - Auto-save checkpoint saat val_loss terbaik
    """
    if device is None:
        device = torch.device("cpu")

    if grad_accum_steps < 1:
        grad_accum_steps = 1

    model.to(device)

    # Inisialisasi optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # Inisialisasi Learning Rate Scheduler dengan Linear Warmup
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=lr * 0.1,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    elif warmup_epochs >= epochs and epochs > 0:
        scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=epochs,
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.1)

    # Konfigurasi Automatic Mixed Precision (AMP)
    amp_enabled, amp_dtype, scaler = setup_amp(device, use_amp=use_amp)
    amp_desc = f"Aktif ({amp_dtype})" if amp_enabled else "Nonaktif"

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    batch_size_base = len(train_batches[0][0]) if train_batches else 0
    effective_batch_size = batch_size_base * grad_accum_steps

    print("\n" + "=" * 60)
    print("  MEMULAI PROSES PELATIHAN MODEL AI (Transformer)...")
    print(f"  Perangkat komputasi  : {device}")
    print(f"  Mixed Precision (AMP): {amp_desc}")
    print(f"  Grad Accum Steps     : {grad_accum_steps} (Effective Batch: {effective_batch_size})")
    print(f"  Warmup Epochs        : {warmup_epochs}")
    print(f"  Total Epochs         : {epochs}")
    print(f"  Learning Rate        : {lr}")
    print("=" * 60)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        num_batches = len(train_batches)
        for step_idx, (X_batch, y_batch) in enumerate(train_batches):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(X_batch)
                loss = criterion(logits.view(-1, model.vocab_size), y_batch.view(-1))
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            total_loss += loss.item()

            is_accum_step = ((step_idx + 1) % grad_accum_steps == 0) or ((step_idx + 1) == num_batches)
            if is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_train_loss = total_loss / num_batches
        scheduler.step()

        # Evaluasi validasi setiap `val_interval` epoch
        if epoch % val_interval == 0 or epoch == 1:
            avg_val_loss = evaluate(
                model,
                val_batches,
                criterion,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"LR: {lr_now:.2e} | "
                f"Waktu: {elapsed:.1f}s"
            )

            # Simpan checkpoint jika val_loss membaik
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, best_val_loss, scaler=scaler)

    print("=" * 60)
    print("  PELATIHAN SELESAI!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60 + "\n")


# ==========================================
# 5. PEMBANGKITAN TEKS (INFERENCE / SAMPLING)
#    Top-k + Top-p (Nucleus) Sampling
# ==========================================

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """
    Menerapkan Top-k dan/atau Top-p (Nucleus) filtering pada logits.

    - top_k > 0: hanya menyimpan k token dengan probabilitas tertinggi.
    - top_p > 0.0: hanya menyimpan token terkecil yang jumlah
      kumulatif probabilitasnya >= top_p (Nucleus Sampling).

    Logits di luar ambang batas diatur ke -inf agar tidak tersampling.
    """
    # --- Top-k filtering ---
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Ambil threshold: nilai logit ke-k terbesar
        threshold = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # --- Top-p (Nucleus) filtering ---
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Tandai token yang kumulatif probabilitasnya > top_p
        # Geser ke kanan agar token pertama yang melewati threshold tetap disimpan
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Kembalikan ke urutan asli
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    return logits


def generate_text(
    model: nn.Module,
    start_text: str = "Kecerdasan",
    length: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    tokenizer: Optional[CharTokenizer] = None,
) -> str:
    """
    Menghasilkan teks baru berdasarkan prompt awal menggunakan
    Top-k + Top-p (Nucleus) Sampling dengan temperature scaling.

    Args:
        model: Model LLMLite yang sudah dilatih.
        start_text: Teks prompt awal.
        length: Jumlah karakter yang akan dibangkitkan.
        temperature: Suhu sampling (rendah=deterministik, tinggi=kreatif).
        top_k: Jumlah token teratas yang dipertimbangkan (0=nonaktif).
        top_p: Ambang kumulatif probabilitas untuk nucleus sampling (0.0=nonaktif).
        device: Perangkat komputasi (cpu/cuda/mps).
        tokenizer: Objek CharTokenizer (default: default_tokenizer).

    Returns:
        String teks yang dihasilkan.
    """
    if device is None:
        device = torch.device("cpu")

    tok = tokenizer or default_tokenizer

    model.eval()
    model.to(device)

    # Validasi karakter awal dengan OOV fallback
    input_indices, oov_chars = tok.encode(start_text)
    if not input_indices:
        print("  [⚠] Semua karakter prompt adalah OOV, menggunakan token acak sebagai seed.")
        input_indices = [random.randint(0, model.vocab_size - 1)]

    # Bangun teks awal hanya dari karakter yang valid
    generated_text = tok.decode(input_indices)

    with torch.inference_mode():
        context = input_indices[:]

        for _ in range(length):
            # Potong konteks agar tidak melebihi max_seq_len
            ctx = context[-model.max_seq_len :]
            input_tensor = torch.tensor([ctx], dtype=torch.long, device=device)

            logits = model(input_tensor)          # [1, T, V]
            last_logits = logits[0, -1, :]        # [V] — hanya prediksi token terakhir

            # Terapkan temperature scaling
            last_logits = last_logits / temperature

            # Terapkan Top-k dan Top-p filtering
            last_logits = top_k_top_p_filtering(last_logits, top_k=top_k, top_p=top_p)

            probabilities = torch.softmax(last_logits, dim=-1)

            # Sampling karakter berikutnya
            next_idx = torch.multinomial(probabilities, num_samples=1).item()

            generated_text += tok.idx_to_char.get(next_idx, "")
            context.append(next_idx)

    return generated_text


def run_interactive(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: Optional[CharTokenizer] = None,
) -> None:
    """
    Menjalankan REPL interaktif untuk generasi teks berkelanjutan
    tanpa perlu memuat ulang model dari disk di setiap pergantian prompt.
    """
    optimizer = optim.AdamW(model.parameters())
    loaded_epoch = load_checkpoint(model, optimizer, path=CHECKPOINT_PATH, device=device)
    if loaded_epoch == 0:
        print("[!] Berjalan dalam mode interaktif tanpa checkpoint terlatih.")

    tok = tokenizer or default_tokenizer

    print("\n" + "=" * 60)
    print("  💬 MODE INTERAKTIF (LLM Lite REPL)")
    print("=" * 60)
    print("  Ketik prompt awal lalu tekan Enter.")
    print("  Perintah khusus:")
    print("    :temp <nilai>   -> Ubah suhu sampling (cth: :temp 0.8)")
    print("    :len <nilai>    -> Ubah panjang generasi (cth: :len 300)")
    print("    :top_k <nilai>  -> Ubah top_k (cth: :top_k 50)")
    print("    :top_p <nilai>  -> Ubah top_p (cth: :top_p 0.95)")
    print("    :info           -> Tampilkan parameter saat ini")
    print("    quit / exit     -> Keluar dari mode interaktif")
    print("=" * 60 + "\n")

    current_temp = args.temperature
    current_length = args.length
    current_top_k = args.top_k
    current_top_p = args.top_p

    while True:
        try:
            prompt = input("llm_lite> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  [✓] Keluar dari mode interaktif. Sampai jumpa!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", ":q"):
            print("  [✓] Keluar dari mode interaktif. Sampai jumpa!")
            break

        # Penanganan perintah runtime
        if prompt.startswith(":temp "):
            try:
                current_temp = float(prompt.split()[1])
                print(f"  [✓] Temperature diubah ke: {current_temp}")
            except ValueError:
                print("  [✗] Format salah. Contoh: :temp 0.7")
            continue
        elif prompt.startswith(":len "):
            try:
                current_length = int(prompt.split()[1])
                print(f"  [✓] Length diubah ke: {current_length}")
            except ValueError:
                print("  [✗] Format salah. Contoh: :len 200")
            continue
        elif prompt.startswith(":top_k "):
            try:
                current_top_k = int(prompt.split()[1])
                print(f"  [✓] Top-k diubah ke: {current_top_k}")
            except ValueError:
                print("  [✗] Format salah. Contoh: :top_k 40")
            continue
        elif prompt.startswith(":top_p "):
            try:
                current_top_p = float(prompt.split()[1])
                print(f"  [✓] Top-p diubah ke: {current_top_p}")
            except ValueError:
                print("  [✗] Format salah. Contoh: :top_p 0.9")
            continue
        elif prompt == ":info":
            print(f"  - Temperature : {current_temp}")
            print(f"  - Length      : {current_length}")
            print(f"  - Top-k       : {current_top_k}")
            print(f"  - Top-p       : {current_top_p}")
            print(f"  - Vocab Size  : {tok.vocab_size}")
            print(f"  - Perangkat   : {device}")
            continue

        # Generasi teks
        hasil = generate_text(
            model,
            start_text=prompt,
            length=current_length,
            temperature=current_temp,
            top_k=current_top_k,
            top_p=current_top_p,
            device=device,
            tokenizer=tok,
        )
        print("-" * 50)
        print(hasil)
        print("-" * 50 + "\n")


# ==========================================
# 6. ARGUMENT PARSER & MAIN RUNNER
# ==========================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Lite — Decoder-Only Transformer Language Model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Jalankan loop pelatihan model.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Bangkitkan teks menggunakan checkpoint yang tersimpan.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Jalankan smoke test: muat checkpoint dan bangkitkan 1 sampel kalimat pendek.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Kecerdasan",
        help="Teks awal untuk pembangkitan (default: 'Kecerdasan').",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Suhu sampling (0.1–2.0). Rendah=deterministik, Tinggi=kreatif (default: 0.7).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling: jumlah token teratas (0=nonaktif, default: 40).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (Nucleus) sampling: ambang kumulatif probabilitas (0.0=nonaktif, default: 0.9).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Jumlah epoch pelatihan (default: 120).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate awal untuk AdamW (default: 3e-4).",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=64,
        help="Panjang sekuens input (default: 64).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Ukuran batch pelatihan (default: 16).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=250,
        help="Jumlah karakter yang dibangkitkan (default: 250).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        "--grad_accum_steps",
        dest="grad_accum_steps",
        type=int,
        default=1,
        help="Jumlah langkah akumulasi gradien sebelum update bobot (default: 1).",
    )
    parser.add_argument(
        "--warmup-epochs",
        "--warmup_epochs",
        dest="warmup_epochs",
        type=int,
        default=5,
        help="Jumlah epoch untuk linear learning rate warmup (default: 5).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gunakan Automatic Mixed Precision (torch.amp) jika didukung (default: True).",
    )
    parser.add_argument(
        "--data",
        "--data-path",
        dest="data_path",
        type=str,
        default=None,
        help="Path ke file dataset teks eksternal (default: gunakan teks sampel bawaan).",
    )
    parser.add_argument(
        "--save-vocab",
        "--save_vocab",
        dest="save_vocab",
        type=str,
        default=None,
        help="Path file JSON untuk menyimpan kosakata tokenizer.",
    )
    parser.add_argument(
        "--load-vocab",
        "--load_vocab",
        dest="load_vocab",
        type=str,
        default=None,
        help="Path file JSON untuk memuat kosakata tokenizer yang telah disimpan.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Jalankan mode interaktif (REPL) untuk generasi teks berulang tanpa reload model.",
    )
    return parser


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_generate(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: Optional[CharTokenizer] = None,
) -> None:
    """Memuat checkpoint dan membangkitkan teks."""
    optimizer = optim.AdamW(model.parameters())
    loaded_epoch = load_checkpoint(model, optimizer, path=CHECKPOINT_PATH, device=device)

    if loaded_epoch == 0:
        print("[!] Tidak ada checkpoint valid. Bangkitkan teks dari model tanpa pelatihan.")

    tok = tokenizer or default_tokenizer

    print("\nHASIL GENERASI TEKS MODEL:")
    hasil = generate_text(
        model,
        start_text=args.prompt,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        tokenizer=tok,
    )
    print(f"Prompt      : '{args.prompt}'")
    print(f"Temperature : {args.temperature}")
    print(f"Top-k       : {args.top_k}")
    print(f"Top-p       : {args.top_p}")
    print("-" * 50)
    print(hasil)
    print("-" * 50)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = get_device()
    print(f"Perangkat komputasi: {device}")

    # 1. Penanganan Dataset & Tokenizer
    dataset_text = load_dataset(args.data_path)

    if args.load_vocab:
        tokenizer = CharTokenizer.load(args.load_vocab)
    elif args.data_path is not None:
        tokenizer = CharTokenizer()
        tokenizer.train_from_text(dataset_text)
    else:
        tokenizer = default_tokenizer

    if args.save_vocab:
        tokenizer.save(args.save_vocab)

    print_vocab_info(tokenizer=tokenizer, text_len=len(dataset_text))

    # Inisialisasi model (target GPU, CPU sebagai fallback)
    model = LLMLite(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        max_seq_len=512,
        dropout=0.1,
    )

    print_model_summary(model)

    # --- Smoke Test ---
    if args.smoke_test:
        print("\n" + "=" * 60)
        print("  🔥 SMOKE TEST — Generate 1 sampel kalimat pendek")
        print("=" * 60)
        optimizer = optim.AdamW(model.parameters())
        load_checkpoint(model, optimizer, path=CHECKPOINT_PATH, device=device)
        hasil = generate_text(
            model,
            start_text=args.prompt,
            length=100,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            tokenizer=tokenizer,
        )
        print(f"Prompt : '{args.prompt}'")
        print("-" * 50)
        print(hasil)
        print("-" * 50)
        print("  [✓] Smoke test selesai.\n")
        return

    # --- Mode Interaktif (REPL) ---
    if args.interactive:
        run_interactive(model, args, device, tokenizer=tokenizer)
        return

    # Encode data teks untuk pelatihan/evaluasi
    dataset_tokens, _ = tokenizer.encode(dataset_text, report_oov=False)

    # --- Training ---
    if args.train:
        train_batches, val_batches = split_train_val(
            dataset_tokens,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            val_ratio=0.2,
        )
        print(f"Train batches: {len(train_batches)} | Val batches: {len(val_batches)}")

        train_model(
            model,
            train_batches,
            val_batches,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            val_interval=10,
            grad_accum_steps=args.grad_accum_steps,
            warmup_epochs=args.warmup_epochs,
            use_amp=args.amp,
        )

    # --- Generate ---
    if args.generate:
        run_generate(model, args, device, tokenizer=tokenizer)

    # --- Mode default: latih lalu generate ---
    if not args.train and not args.generate:
        train_batches, val_batches = split_train_val(
            dataset_tokens,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            val_ratio=0.2,
        )
        print(f"Train batches: {len(train_batches)} | Val batches: {len(val_batches)}")

        train_model(
            model,
            train_batches,
            val_batches,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            val_interval=10,
            grad_accum_steps=args.grad_accum_steps,
            warmup_epochs=args.warmup_epochs,
            use_amp=args.amp,
        )

        run_generate(model, args, device, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
