# data/download_and_extract.py

import os
import gdown
import zipfile

def download_and_extract_polyvore():
    # Ruta al directorio raíz del proyecto (uno por encima del script)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "datasets")
    os.makedirs(data_dir, exist_ok=True)

    # Ruta del zip y carpeta de extracción
    zip_name = "polyvore.zip"
    zip_path = os.path.join(data_dir, zip_name)
    extract_path = os.path.join(data_dir, "polyvore")

    # Enlace de descarga
    url = "https://drive.google.com/uc?id=1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu"

    # Descargar si no existe
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name}...")
        gdown.download(url, zip_path, quiet=False)
    else:
        print(f"{zip_name} already exists. Skipping download.")

    # Extraer si no existe
    if not os.path.exists(extract_path):
        print(f"Extracting to {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print(f"{extract_path} already exists. Skipping extraction.")

    # Eliminar el zip
    os.remove(zip_path)
    print(f"Deleted zip file: {zip_path}")

if __name__ == "__main__":
    download_and_extract_polyvore()
