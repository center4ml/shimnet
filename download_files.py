import urllib.request
from pathlib import Path
import argparse

ALL_FILES_TO_DOWNLOAD = {
    "weights": [{
        "url": "https://drive.google.com/uc?export=download&id=17fTNWl7YW6mPbbZWga0EfdoF_6S8fCke",
        "destination": "weights/shimnet_700MHz.pt"
    },
    {
        "url": "https://drive.google.com/uc?export=download&id=1_VxOpFGJcFsOa5DHOW2GJbP8RvHCmC1N",
        "destination": "weights/shimnet_600MHz.pt"
    }],
    "SCRF": [{
        "url": "https://drive.google.com/uc?export=download&id=113al7A__yYALx_2hkESuzFIDU3feVtNY",
        "destination": "data/scrf_61_700MHz.pt"
    }],
    "mupltiplets": [{
        "url": "https://drive.google.com/uc?export=download&id=1QGvV-Au50ZxaP1vFsmR_auI299Dw-Wrt",
        "destination": "data/multiplets_10000_parsed.txt"
    }],
    "development": []
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Download files: weighs (default), SCRF (optional), multiplet data (optional)',
    )
    parser.add_argument(
        '--weights',
        action='store_true',
        default=True,
        help='Download weights file (default behavior). Use --no-weights to opt out.',
    )
    parser.add_argument(
        '--no-weights',
        action='store_false',
        dest='weights',
        help='Do not download weights file.',
    )
    parser.add_argument('--SCRF', action='store_true', help='Download SCRF file') 
    parser.add_argument('--multiplets', action='store_true', help='Download multiplets data file')
    parser.add_argument('--development', action='store_true', help='Download development weights file')

    parser.add_argument('--all', action='store_true', help='Download all available files')

    args = parser.parse_args()
    # Set all individual flags if --all is specified
    if args.all:
        args.weights = True
        args.SCRF = True 
        args.multiplets = True
        args.development = True
        
    return args

def download_file(url, target):
    target = Path(target)
    if target.exists():
        response = input(f"File {target} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print(f"Download of {target} cancelled")
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {target}")
    except Exception as e:
        print(f"Failed to download file from {url}:\n {e}")


if __name__ == "__main__":
    args = parse_args()

    main_dir = Path(__file__).parent
    if args.weights:
        for file_data in ALL_FILES_TO_DOWNLOAD["weights"]:
            download_file(file_data["url"], main_dir / file_data["destination"])
    
    if args.SCRF:
        for file_data in ALL_FILES_TO_DOWNLOAD["SCRF"]:
            download_file(file_data["url"], main_dir / file_data["destination"])
    
    if args.multiplets:
        for file_data in ALL_FILES_TO_DOWNLOAD["mupltiplets"]:
            download_file(file_data["url"], main_dir / file_data["destination"])
    
    if args.development:
        for file_data in ALL_FILES_TO_DOWNLOAD["development"]:
            download_file(file_data["url"], main_dir / file_data["destination"])
