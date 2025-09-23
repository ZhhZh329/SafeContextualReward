import os
import json
import requests
from tqdm import tqdm

def download_from_txts(data_dir, target_dir, mode="debug", debug_lines=5, extract_text=True):
    os.makedirs(target_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    for txt_file in txt_files:
        txt_path = os.path.join(data_dir, txt_file)
        folder_name = os.path.splitext(txt_file)[0]
        save_dir = os.path.join(target_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        with open(txt_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        if mode == "debug":
            urls = urls[:debug_lines]

        print(f"[INFO] 从 {txt_file} 下载 {len(urls)} 个文件到 {save_dir}")

        for i, url in enumerate(urls, 1):
            try:
                resp = requests.get(url, stream=True, timeout=30)
                resp.raise_for_status()

                base_name = os.path.splitext(os.path.basename(url.split("?")[0]))[0] or f"file_{i}"
                jsonl_path = os.path.join(save_dir, f"{base_name}.jsonl")

                # ===== 原样下载保存 JSONL =====
                total_size = int(resp.headers.get("content-length", 0))
                block_size = 1024
                with open(jsonl_path, "wb") as f_out, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"下载 {base_name}.jsonl",
                    ascii=True,
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=block_size):
                        if chunk:
                            f_out.write(chunk)
                            pbar.update(len(chunk))


                print(f"  ✅ 已下载: {base_name}.jsonl（未解析 text）")

            except Exception as e:
                print(f"  ❌ 下载失败 {url}，错误: {e}")


if __name__ == "__main__":
    data_dir = "./src/data_construction/download/redpajama_urls"
    target_dir = "./data/redpajama"

    # Debug 模式：每个 txt 只下前 1 行；并且默认 extract_text=True，如果只想保存 jsonl，把 extract_text=False
    download_from_txts(data_dir, target_dir, mode="debug", debug_lines=1, extract_text=True)
    # 全量模式
    # download_from_txts(data_dir, target_dir, mode="full", extract_text=True)
