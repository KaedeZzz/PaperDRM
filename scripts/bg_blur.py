from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from paperdrm.stage0_loader import DataPaths


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def clear_image_folder(path: Path) -> None:
    """
    Remove existing image files so background regeneration does not mix old data.
    """
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            item.unlink(missing_ok=True)


if __name__ == "__main__":
    paths = DataPaths.from_root("data")
    images = []
    img_paths = sorted(paths.processed.glob('*.*'))
    debug_first = False
    for idx, img_path in enumerate(tqdm(img_paths, desc="blurring backgrounds")):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read {img_path}")
            img = cv2.GaussianBlur(img, (0, 0), 5)
            small = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(small, (0, 0), 20)
            lowpass = cv2.resize(blurred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            images.append(lowpass)

            if debug_first and idx == 0:
                img_arr = img.astype(np.float32)
                low_arr = lowpass.astype(np.float32)
                diff = img_arr - low_arr
                plt.hist(diff.ravel(), bins=100)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].set_title('Original Image')
                ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
                ax[1].set_title('Blurred Background')
                ax[1].imshow(lowpass, cmap='gray', vmin=0, vmax=255)
                plt.show()
                plt.imshow(diff, cmap='gray')
                plt.title('Difference Image')
                plt.colorbar()
                plt.show()
        except IOError:
            print(f"Could not open image at path: {img_path}")

    background_dir = paths.root / 'background'
    background_dir.mkdir(parents=True, exist_ok=True)
    clear_image_folder(background_dir)
    for i in tqdm(range(len(images)), desc='saving blurred images'):
        cv2.imwrite(str(background_dir / img_paths[i].name), images[i])
