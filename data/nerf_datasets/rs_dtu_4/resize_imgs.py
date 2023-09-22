import os
import argparse
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="DTU",
    help="Data directory",
)
parser.add_argument(
    "--imsize",
    type=int,
    default=128,
    help="Output image size",
)
parser.add_argument(
    "--num_workers", "-j",
    type=int,
    default=8,
    help="Num processes",
)
args = parser.parse_args()


def process_image(im_path):
    im = cv2.imread(im_path)
    if im is None:
        print('FAIL', im_path)
        return
    H, W, C = im.shape
    if H <= 600:
        return
    im = cv2.pyrDown(im)
    im = cv2.pyrDown(im)
    cv2.imwrite(im_path, im)


futures = []
dir_paths = [os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir)]
im_paths = [os.path.join(dirpath, 'image', x) for dirpath in dir_paths if os.path.isdir(dirpath) for x in os.listdir(os.path.join(dirpath, 'image'))]
#  mask_paths = [os.path.join(dirpath, 'mask', x) for dirpath in dir_paths if os.path.isdir(dirpath) for x in os.listdir(os.path.join(dirpath, 'mask'))]
#  im_paths.extend(mask_paths)
progress = tqdm(total=len(im_paths))
with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    for im_path in im_paths:
        futures.append(
            executor.submit(
                process_image,
                im_path,
            )
        )
    for future in futures:
        _ = future.result()
        progress.update(1)
