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
    },
    { # https://drive.google.com/file/d/1643Il3qgCupY0n8Mar6WBc2WVuoQRzie/view?usp=sharing
        "url": "https://drive.google.com/uc?export=download&id=1643Il3qgCupY0n8Mar6WBc2WVuoQRzie",
        "destination": "weights/shimnetV2_600MHz.pt"
    },
    { # https://drive.google.com/file/d/1VTvNXp-kpCF78G2ET_FldgV3s22jdTbz/view?usp=sharing
        "url": "https://drive.google.com/uc?export=download&id=1VTvNXp-kpCF78G2ET_FldgV3s22jdTbz",
        "destination": "weights/sshimnetV2RM_mono-click_finetune.pt"
    },
    { # https://drive.google.com/file/d/16Ut1Zt2LuIr6GyzPq4DUwV4IeMrcJt6f/view?usp=sharing
        "url": "https://drive.google.com/uc?export=download&id=16Ut1Zt2LuIr6GyzPq4DUwV4IeMrcJt6f",
        "destination": "weights/shimnetV2RM_bis-click_finetune.pt"
    }],
    "SCRF": [{
        "url": "https://drive.google.com/uc?export=download&id=113al7A__yYALx_2hkESuzFIDU3feVtNY",
        "destination": "data/scrf_61_700MHz.pt"
    },
    {
        "url": "https://drive.google.com/uc?export=download&id=1J-DsPtaITXU3TFrbxaZPH800U1uIiwje",
        "destination": "data/scrf_81_600MHz.pt"
    }],
    "multiplets": [{
        "url": "https://drive.google.com/uc?export=download&id=1QGvV-Au50ZxaP1vFsmR_auI299Dw-Wrt",
        "destination": "data/multiplets_10000_parsed.txt"
    }],
    "development": []
    #     { #https://drive.google.com/file/d/1_KzJ1fj69engsbh1PZJMdvVn-Nn14PgI/view?usp=drive_link
    #         "url": "https://drive.google.com/uc?export=download&id=1_KzJ1fj69engsbh1PZJMdvVn-Nn14PgI",
    #         "destination": "weights/23-0_2501.pt"
    #     },
    #     { # https://drive.google.com/file/d/1R8Okf82FU5IrBZUyw9ZFjz8TOwjajqjy/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1R8Okf82FU5IrBZUyw9ZFjz8TOwjajqjy",
    #         "destination": "weights/23-1_3907.pt"
    #     },
    #     { # https://drive.google.com/file/d/1PpAJggtG5xtssqop-s7Igbtr_StjH4i4/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1PpAJggtG5xtssqop-s7Igbtr_StjH4i4",
    #         "destination": "weights/23-end.pt"
    #     },
    #     { # https://drive.google.com/file/d/1tRzUatYyZNmkWFfgN7M_bZLBeowoh3Aq/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1tRzUatYyZNmkWFfgN7M_bZLBeowoh3Aq",
    #         "destination": "weights/24-end.pt"
    #     },
    #     { #https://drive.google.com/file/d/1nsep0IGd5OUG_EMBwL55s-1TxQxyABEL/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1nsep0IGd5OUG_EMBwL55s-1TxQxyABEL",
    #         "destination": "weights/25-end.pt"
    #     },
    #     { #https://drive.google.com/file/d/1WBVeZHuvC-FOTSJWla_gDNQjDmXn2h_n/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1WBVeZHuvC-FOTSJWla_gDNQjDmXn2h_n",
    #         "destination": "weights/26-end.pt"
    #     },
    #     { # 27-0_2501 https://drive.google.com/file/d/1B9XLQ0C0tSncQRysEqKIKpbi3AR6coXw/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1B9XLQ0C0tSncQRysEqKIKpbi3AR6coXw",
    #         "destination": "weights/27-0_2501.pt"
    #     },
    #     { # 27-1_3907 https://drive.google.com/file/d/1qjHdIp76T08Bz782mq1G3SD_Ii0MwqRr/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1qjHdIp76T08Bz782mq1G3SD_Ii0MwqRr",
    #         "destination": "weights/27-1_3907.pt"},
    #     { # 27-end https://drive.google.com/file/d/1_tTKZ9NmCl7K6lJKeMx03nBgs2ovN1pu/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1_tTKZ9NmCl7K6lJKeMx03nBgs2ovN1pu",
    #         "destination": "weights/27-end.pt"
    #     },
    #     { #28-end https://drive.google.com/file/d/1dc9aDSEAeiLCQfCmdI1X_6uIMPefEsop/view?usp=sharing
    #         "url": "https://drive.google.com/uc?export=download&id=1dc9aDSEAeiLCQfCmdI1X_6uIMPefEsop",
    #         "destination": "weights/28-end.pt"
    #     }
    # ]
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

def download_file(url, target, overwrite=False):
    target = Path(target)
    if target.exists() and not overwrite:
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
            download_file(file_data["url"], main_dir / file_data["destination"], args.overwrite)
    
    if args.SCRF:
        for file_data in ALL_FILES_TO_DOWNLOAD["SCRF"]:
            download_file(file_data["url"], main_dir / file_data["destination"], args.overwrite)
    
    if args.multiplets:
        for file_data in ALL_FILES_TO_DOWNLOAD["multiplets"]:
            download_file(file_data["url"], main_dir / file_data["destination"], args.overwrite)
    
    if args.development:
        for file_data in ALL_FILES_TO_DOWNLOAD["development"]:
            download_file(file_data["url"], main_dir / file_data["destination"], args.overwrite)
