"""
Knowledge Agent - Modul Terisolasi Pengumpul Pengetahuan AI
Mengambil wawasan dari Model Guru (API eksternal) dan otomatis mengumpulkannya
menjadi dataset pelatihan untuk model lokal (llm_lite).

File ini terpisah dan independen dari llm_lite.py untuk menjaga keamanan API key
dan memastikan kode model utama tetap bersih saat dipublikasikan.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_OUTPUT_FILE = "knowledge_dataset.txt"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_api_key(args_key: Optional[str] = None) -> str:
    """
    Mengambil API key dengan urutan prioritas aman:
    1. Argumen CLI (--api-key)
    2. Environment variable (LLM_API_KEY, OPENAI_API_KEY, GROQ_API_KEY)
    3. Input interaktif pengguna saat runtime (tanpa tersimpan ke disk/git)
    """
    if args_key:
        return args_key.strip()

    for env_var in ("LLM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    print("\n[!] API Key belum terdeteksi dari argumen CLI atau environment variable.")
    print("    Silakan masukkan API Key Anda (key ini hanya disimpan di memori sesi saat ini):")
    try:
        user_key = input("API Key > ").strip()
        if not user_key:
            print("[✗] API Key kosong. Program dihentikan.")
            sys.exit(1)
        return user_key
    except (KeyboardInterrupt, EOFError):
        print("\nOperasi dibatalkan.")
        sys.exit(1)


def query_teacher_model(
    prompt: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 30,
) -> str:
    """
    Mengirimkan permintaan ke Teacher LLM menggunakan protokol OpenAI-compatible
    menggunakan library standar Python (urllib) tanpa dependensi eksternal.
    """
    # Bersihkan URL endpoint
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    if system_prompt is None:
        system_prompt = (
            "Anda adalah asisten AI pendidik yang memberikan pengetahuan faktual, "
            "jelas, informatif, dan terstruktur dalam Bahasa Indonesia yang baik dan baku. "
            "Jawaban Anda akan dijadikan teks bacaan pembelajaran untuk melatih model bahasa kecil."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "LLM-Lite-KnowledgeAgent/1.0",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8")
            result = json.loads(resp_body)
            content = result["choices"][0]["message"]["content"]
            return content.strip()
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP Error {e.code}: {error_msg}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gagal terhubung ke {url}: {e.reason}")


def append_knowledge_to_dataset(
    topic: str,
    knowledge: str,
    output_path: str = DEFAULT_OUTPUT_FILE,
) -> int:
    """
    Menyimpan data pengetahuan baru ke dalam file dataset teks.
    Format dirancang agar bersih dan langsung siap dilatihkan ke llm_lite.py.
    """
    separator = "\n" + "=" * 50 + "\n"
    entry = (
        f"{separator}"
        f"Topik: {topic.strip()}\n\n"
        f"{knowledge.strip()}\n"
    )

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(entry)

    # Kembalikan total ukuran karakter file setelah penambahan
    return os.path.getsize(output_path)


def get_dataset_stats(output_path: str = DEFAULT_OUTPUT_FILE) -> Dict[str, Any]:
    """Menghitung statistik pengetahuan yang terkumpul di file dataset."""
    if not os.path.exists(output_path):
        return {"exists": False, "bytes": 0, "chars": 0, "entries": 0}

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = content.count("Topik:")
    return {
        "exists": True,
        "bytes": len(content.encode("utf-8")),
        "chars": len(content),
        "entries": entries,
    }


def preview_last_entries(output_path: str = DEFAULT_OUTPUT_FILE, count: int = 2) -> None:
    """Menampilkan cuplikan entri pengetahuan terakhir."""
    if not os.path.exists(output_path):
        print("  [!] File dataset belum ada.")
        return

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("==================================================")
    non_empty = [p.strip() for p in parts if p.strip()]

    if not non_empty:
        print("  [!] Dataset masih kosong.")
        return

    print(f"\n--- Menampilkan {min(count, len(non_empty))} Entri Terakhir ---")
    for idx, p in enumerate(non_empty[-count:], 1):
        print(f"\n[Entri #{idx}]")
        print(p[:300] + ("..." if len(p) > 300 else ""))
    print("-" * 50 + "\n")


def run_interactive_agent(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL,
    output_path: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """
    Loop interaktif untuk mengajukan pertanyaan, menerima jawaban guru,
    dan otomatis menyimpannya ke bank data pengetahuan.
    """
    print("\n" + "=" * 65)
    print("  🧠 KNOWLEDGE AGENT — Pengumpul Pengetahuan Mandiri")
    print("=" * 65)
    print(f"  Target Endpoint : {base_url}")
    print(f"  Teacher Model   : {model_name}")
    print(f"  File Dataset    : {output_path}")
    print("  Perintah Khusus :")
    print("    :stats        -> Tampilkan statistik dataset yang terkumpul")
    print("    :preview      -> Tampilkan cuplikan pengetahuan terakhir")
    print("    :model <nama> -> Ganti model guru saat runtime")
    print("    exit / quit   -> Keluar dari program")
    print("=" * 65 + "\n")

    current_model = model_name

    while True:
        try:
            prompt = input("knowledge_agent> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  [✓] Selesai. Pengetahuan aman tersimpan.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", ":q"):
            print("  [✓] Selesai. Sampai jumpa!")
            break

        if prompt == ":stats":
            stats = get_dataset_stats(output_path)
            if stats["exists"]:
                print(f"  - File        : {output_path}")
                print(f"  - Total Topik : {stats['entries']:,} entri")
                print(f"  - Total Huruf : {stats['chars']:,} karakter ({stats['bytes'] / 1024:.1f} KB)")
            else:
                print(f"  - File {output_path} belum dibuat.")
            continue

        if prompt == ":preview":
            preview_last_entries(output_path)
            continue

        if prompt.startswith(":model "):
            new_m = prompt.split(maxsplit=1)[1].strip()
            if new_m:
                current_model = new_m
                print(f"  [✓] Model guru diubah ke: {current_model}")
            continue

        print(f"\n  [⏳] Menanyakan ke {current_model}...")
        t0 = time.time()
        try:
            jawaban = query_teacher_model(
                prompt=prompt,
                api_key=api_key,
                base_url=base_url,
                model_name=current_model,
            )
            elapsed = time.time() - t0
            print(f"  [✓] Jawaban diterima dalam {elapsed:.2f} detik:\n")
            print("-" * 55)
            print(jawaban)
            print("-" * 55)

            # Simpan ke dataset
            file_size = append_knowledge_to_dataset(prompt, jawaban, output_path)
            print(f"  [💾] Otomatis tersimpan ke '{output_path}' ({file_size:,} byte total).\n")

        except Exception as err:
            print(f"  [✗] Gagal mengambil pengetahuan: {err}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knowledge Agent — Pengumpul Pengetahuan AI Terisolasi",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default=None,
        help="API Key untuk Teacher LLM (jika tidak diset, akan membaca env var atau input aman).",
    )
    parser.add_argument(
        "--url",
        "--base-url",
        dest="base_url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL endpoint API OpenAI-compatible (default: '{DEFAULT_BASE_URL}').",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Nama model guru yang digunakan (default: '{DEFAULT_MODEL}').",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path file teks tujuan untuk menyimpan dataset (default: '{DEFAULT_OUTPUT_FILE}').",
    )
    parser.add_argument(
        "--ask",
        dest="ask_prompt",
        type=str,
        default=None,
        help="Pertanyaan satu kali langsung via CLI (non-interaktif).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Tampilkan statistik dataset pengetahuan saat ini dan keluar.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Periksa statistik langsung
    if args.stats:
        stats = get_dataset_stats(args.output_file)
        print("\nSTATISTIK DATASET PENGETAHUAN:")
        print(f"File Path    : {args.output_file}")
        print(f"Status       : {'Ditemukan' if stats['exists'] else 'Belum Ada'}")
        print(f"Total Entri  : {stats['entries']} topik")
        print(f"Total Ukuran : {stats['chars']:,} karakter ({stats['bytes'] / 1024:.2f} KB)\n")
        return

    # Ambil API Key
    api_key = get_api_key(args.api_key)

    # Mode Single-shot CLI
    if args.ask_prompt:
        print(f"\n[⏳] Mengambil wawasan untuk: '{args.ask_prompt}'...")
        try:
            jawaban = query_teacher_model(
                prompt=args.ask_prompt,
                api_key=api_key,
                base_url=args.base_url,
                model_name=args.model_name,
            )
            print("\nHASIL PENGETAHUAN:")
            print("-" * 55)
            print(jawaban)
            print("-" * 55)
            append_knowledge_to_dataset(args.ask_prompt, jawaban, args.output_file)
            print(f"[✓] Berhasil disimpan ke '{args.output_file}'\n")
        except Exception as e:
            print(f"[✗] Error: {e}\n")
            sys.exit(1)
        return

    # Mode Interaktif REPL
    run_interactive_agent(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        output_path=args.output_file,
    )


if __name__ == "__main__":
    main()
