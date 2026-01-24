"""
Dataset Zip Utility
===================

This script creates a compressed zip file from the generated Cut-Paste dataset.
Useful for transferring the augmented dataset between machines or cloud services.

Features:
    - Recursive compression of all files in dataset folder
    - Progress bar with file count
    - Final zip size report in GB
    - ZIP_DEFLATED compression for smaller file size

Usage:
    python zip_dataset.py --folder /path/to/cityscapes_cutpaste

    # Custom output path:
    python zip_dataset.py --folder /path/to/dataset --output /path/to/dataset.zip
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm


def zip_folder(folder_path: str, output_path: str = None):
    """
    Create a zip archive from a folder.

    Recursively compresses all files in the folder while preserving
    the directory structure inside the zip file.

    Args:
        folder_path: Path to the folder to compress.
        output_path: Path for output zip file (default: <folder_path>.zip).

    Raises:
        FileNotFoundError: If the input folder does not exist.
    """
    folder = Path(folder_path)

    # Validate input folder exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Default output path: same name as folder with .zip extension
    if output_path is None:
        output_path = f"{folder_path}.zip"

    # Count total files for progress bar (exclude directories)
    all_files = list(folder.rglob("*"))
    files_to_zip = [f for f in all_files if f.is_file()]

    print(f"Compressing {len(files_to_zip)} files from {folder_path}")
    print(f"Output: {output_path}")

    # Create zip file with DEFLATE compression (good balance of speed/size)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(files_to_zip, desc="Zipping"):
            # Store relative path inside zip to preserve folder structure
            arcname = file_path.relative_to(folder)
            zipf.write(file_path, arcname)

    # Report final zip size in GB
    zip_size = os.path.getsize(output_path) / (1024**3)  # Convert bytes to GB
    print(f"\nZip created: {output_path}")
    print(f"Size: {zip_size:.2f} GB")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create zip archive of generated Cut-Paste dataset"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/teamspace/studios/this_studio/cityscapes_cutpaste",
        help="Folder to compress (path to generated dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zip file path (default: <folder>.zip)",
    )

    args = parser.parse_args()

    # Run compression
    zip_folder(args.folder, args.output)
