from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from paperdrm.stage0_loader import DataPaths


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def clear_image_folder(path: Path) -> None:
    """
    Remove existing image files so a new run starts from a clean slate.
    """
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            item.unlink(missing_ok=True)


def rgb2grey(img, mode='luminosity'):
    """Convert RGB image to greyscale using luminosity method."""
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        if mode == 'average':
            grey_img = np.mean(img, axis=2)
        elif mode == 'luminosity':
            grey_img = 0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]
            grey_img = np.reshape(grey_img, (img.shape[0], img.shape[1]))
        else:
            raise ValueError("Mode not recognised. Use 'average' or 'luminosity'.")
        return grey_img.astype(np.uint8)
    else:
        raise ValueError("Input image must be either a 2D greyscale or a 3D RGB image.")
    

if __name__ == "__main__":
    paths = DataPaths.from_root("data")

    # Search for files with corresponding affix
    roi = None # region of interest of the images to be processed
    read_path = paths.raw # image path to read
    write_path = paths.processed # image path to write
    print(f"Reading images from folder: {read_path}")
    print(f"Saving processed images to folder: {write_path}")
    img_format = 'jpg'

    images = []
    sample_paths = sorted(read_path.glob('*.' + img_format))
    for image_path in tqdm(sample_paths, desc='loading images'):
        try:
            image = cv2.imread(str(image_path))
            images.append(image)
        except IOError:
            print(f"Could not open image at path: {image_path}")
    num_images = len(images)
    
    # Convert all images into greyscale
    for i in tqdm(range(num_images), desc='converting images into greyscale'):
        images[i] = rgb2grey(images[i], mode='luminosity')
        if roi is not None:
            # ROI will be implemented into CLI in the future
            imin, jmin, imax, jmax = roi # type: ignore
            images[i] = images[i][jmin:jmax, imin:imax]

    write_path.mkdir(parents=True, exist_ok=True)
    clear_image_folder(write_path)
    for i in tqdm(range(num_images), desc='saving greyscale images'):
        cv2.imwrite(str(write_path / sample_paths[i].name), images[i])
