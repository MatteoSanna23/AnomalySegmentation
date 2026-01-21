"""
Script per creare un file zip del dataset cut-paste generato.
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm


def zip_folder(folder_path: str, output_path: str = None):
    """
    Crea un file zip di una cartella.

    Args:
        folder_path: Path della cartella da comprimere
        output_path: Path del file zip di output (opzionale)
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder_path}")

    if output_path is None:
        output_path = f"{folder_path}.zip"

    # Conta file totali per progress bar
    all_files = list(folder.rglob("*"))
    files_to_zip = [f for f in all_files if f.is_file()]

    print(f"Comprimendo {len(files_to_zip)} file da {folder_path}")
    print(f"Output: {output_path}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(files_to_zip, desc="Zipping"):
            # Percorso relativo dentro lo zip
            arcname = file_path.relative_to(folder)
            zipf.write(file_path, arcname)

    # Dimensione finale
    zip_size = os.path.getsize(output_path) / (1024**3)  # GB
    print(f"\nZip creato: {output_path}")
    print(f"Dimensione: {zip_size:.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crea zip del dataset")
    parser.add_argument(
        "--folder",
        type=str,
        default="/teamspace/studios/this_studio/cityscapes_cutpaste",
        help="Cartella da comprimere",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path file zip di output (default: <folder>.zip)",
    )

    args = parser.parse_args()
    zip_folder(args.folder, args.output)
