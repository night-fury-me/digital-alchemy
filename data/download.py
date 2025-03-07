import os
import urllib.request
import lzma
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "QM7X_Dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

dataset_files = [
    "1000.xz", "2000.xz", "3000.xz", "4000.xz",
    "5000.xz", "6000.xz", "7000.xz", "8000.xz"
]
base_url = "https://zenodo.org/records/4288677/files/"

def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded / total_size, 1.0) * 100
    bar_length = 40
    filled_length = int(bar_length * percent / 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\rDownloading: [{bar}] {percent:.2f}%")
    sys.stdout.flush()

for filename in dataset_files:
    file_path = os.path.join(DATASET_DIR, filename)
    url = base_url + filename

    if os.path.exists(file_path):
        print(f"File already exists: {filename}, skipping download.")
        continue

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, file_path, progress_bar)
        print(f"\nDownload complete: {filename}")
    except Exception as e:
        print(f"\nFailed to download {filename}: {e}")

for filename in dataset_files:
    compressed_path = os.path.join(DATASET_DIR, filename)
    extracted_path = os.path.join(DATASET_DIR, filename.replace(".xz", ".hdf5"))

    if os.path.exists(extracted_path):
        print(f"File already extracted: {extracted_path}, skipping.")
        continue

    print(f"Extracting {filename}...")
    try:
        with lzma.open(compressed_path, "rb") as f_in, open(extracted_path, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"Extraction complete: {extracted_path}")

        os.remove(compressed_path)
        print(f"Deleted compressed file: {compressed_path}")
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")

print("\nVerifying extracted files:")
for filename in dataset_files:
    extracted_path = os.path.join(DATASET_DIR, filename.replace(".xz", ".hdf5"))
    if os.path.exists(extracted_path):
        print(f"{extracted_path} exists and is ready for use.")
    else:
        print(f"{extracted_path} is missing. Check for errors.")

print("\nAll files are ready for use in SchNetPack!")
