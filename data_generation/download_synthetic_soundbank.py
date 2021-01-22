import argparse
import time

from desed import download_soundbank
from desed.utils import create_folder

if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default="../dataset",
                        help="The base folder in which to download the synthetic soundbank")
    args = parser.parse_args()

    create_folder(args.basedir)
    download_soundbank(args.basedir, sins_bg=True, tut_bg=True, split_train_valid=True)
    print("Synthetic soundbank downloaded")