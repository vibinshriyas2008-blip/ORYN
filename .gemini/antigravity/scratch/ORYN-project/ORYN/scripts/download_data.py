import os
import requests
import zipfile
import io

def download_and_extract():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    print(f"Downloading data from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    print("Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(data_dir)
        
    extracted_dir = os.path.join(data_dir, 'ml-latest-small')
    if os.path.exists(extracted_dir):
        files = os.listdir(extracted_dir)
        print(f"Success! Files downloaded and extracted to {extracted_dir}")
        print(f"Files: {files}")
    else:
        print("Extraction failed or directory not found.")

if __name__ == "__main__":
    download_and_extract()
