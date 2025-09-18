import urllib.request
import zipfile
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
    },
    {
        "url": "https://drive.google.com/uc?export=download&id=1J-DsPtaITXU3TFrbxaZPH800U1uIiwje",
        "destination": "data/scrf_81_600MHz.pt"
    }],
    "mupltiplets": [{
        "url": "https://drive.google.com/uc?export=download&id=1QGvV-Au50ZxaP1vFsmR_auI299Dw-Wrt",
        "destination": "data/multiplets_10000_parsed.txt"
    }],
    "workshops": [{
        "url": "https://drive.google.com/uc?export=download&id=1gj60yazOVF2P81Vtupju_Elt9TKY6hm0",
        "destination": "SCRF_extraction",
        "unzip": True
    }],
    "development": []
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Download files: weighs (default), SCRF (optional), multiplet data (optional)',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
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
    parser.add_argument('--SCRF', action='store_true', help='Download SCRF files - Shim Coil Response Functions') 
    parser.add_argument('--multiplets', action='store_true', help='Download multiplets data file')
    parser.add_argument('--workshops', action='store_true', dest='workshops', help='DDownload workshops data files')
    parser.add_argument('--development', action='store_true', help='Download development weights file')

    parser.add_argument('--all', action='store_true', help='Download all available files')

    args = parser.parse_args()
    # Set all individual flags if --all is specified
    if args.all:
        args.weights = True
        args.SCRF = True 
        args.multiplets = True
        args.development = True
        args.workshops = True
    return args

def download_file(url, target, overwrite=False, unzip=False):
    target = Path(target)
    try:
        if unzip: # download archive and unzip to target directory
            if target.exists() and not overwrite:
                response = input(f"Directory {target} already exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print(f"Download of {target} cancelled")
                    return
            target.mkdir(parents=True, exist_ok=True)
            tmp_file, _ = urllib.request.urlretrieve(url)
            with zipfile.ZipFile(tmp_file, 'r') as z:
                target.mkdir(parents=True, exist_ok=True)
                z.extractall(target)
            print(f"Extracted archive to {target}")
        else: # download single file and store at target location
            if target.exists() and not overwrite:
                response = input(f"File {target} already exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print(f"Download of {target} cancelled")
                    return
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, target)
            print(f"Downloaded {target}")
    except Exception as e:
        print(f"Failed to download file from {url}:\n {e}")


if __name__ == "__main__":
    args = parse_args()

    main_dir = Path(__file__).parent

    for files_group_name, files_data in ALL_FILES_TO_DOWNLOAD.items(): # iterate over "weights", "SCRF", "multiplets", "development"
        if getattr(args, files_group_name, False):
            for file_data in files_data:
                download_file(file_data["url"], main_dir / file_data["destination"], overwrite=args.overwrite, unzip=file_data.get("unzip", False))
