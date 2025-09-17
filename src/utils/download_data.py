import os
import subprocess
import shutil

# CONFIGURATION
DATA_DIR = "data"
ASPERA_BIN = shutil.which("ascp")  # auto-detect ascp binary
ASPERA_SERVER = "faspex.cancerimagingarchive.net"
PACKAGE_ID = "985"  # BCBM RadioGenomics package ID

# Aspera authentication for public packages
USERNAME = "anonymous"
PASSWORD = "anonymous"

# Remote path: replace this if TCIA changes the package structure
REMOTE_PATH = f"{USERNAME}@{ASPERA_SERVER}:/aspera/faspex/packages/{PACKAGE_ID}/"

def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[INFO] Created directory: {DATA_DIR}")
    else:
        print(f"[INFO] Directory already exists: {DATA_DIR}")

def download_data():
    """Download dataset using Aspera CLI."""
    if not ASPERA_BIN:
        raise RuntimeError(
            "Aspera CLI (ascp) not found. Install IBM Aspera and ensure 'ascp' is in your PATH."
        )

    print("[INFO] Starting download with Aspera...")
    cmd = [
        ASPERA_BIN,
        "-P", "33001",                  # Aspera default port
        "-O", "33001",                  # Fallback port
        "-T",                            # Disable encryption overhead for faster transfers
        "-k", "1",                        # Resume partially downloaded files
        "-Q",                             # Quiet mode (less verbose)
        "--overwrite=never",              # Don't overwrite existing files
        "--policy=fair",
        REMOTE_PATH,
        DATA_DIR
    ]

    subprocess.run(cmd, check=True)
    print("[INFO] Download complete.")

def cleanup_ds_store():
    """Remove all .DS_Store files recursively in data folder."""
    print("[INFO] Cleaning up .DS_Store files...")
    count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                count += 1
    print(f"[INFO] Removed {count} .DS_Store files.")

if __name__ == "__main__":
    try:
        ensure_data_dir()
        download_data()
        cleanup_ds_store()
        print("[INFO] All done! Files are ready in the 'data/' folder.")
    except Exception as e:
        print(f"[ERROR] {e}")
