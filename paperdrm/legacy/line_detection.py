import cv2
import numpy as np
from tqdm import tqdm


def hough_transform(edge_image, rho_res=1, theta_res=1):
    """
    Compute the Hough Transform accumulator for lines using vectorised binning.
    
    Parameters:
        edge_image : 2D numpy array (binary or grayscale edge map)
        rho_res    : resolution of rho in pixels
        theta_res  : resolution of theta in degrees
        
    Returns:
        accumulator : 2D array (rho x theta)
        rhos        : array of rho values
        thetas      : array of theta values (radians)
    """
    h, w = edge_image.shape
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    # Include the endpoint to avoid off-by-one at the far edge
    diag_len = int(np.ceil(np.hypot(h, w)))  # Diagonal length of the image
    rhos = np.arange(-diag_len, diag_len + rho_res, rho_res)
    num_rhos, num_thetas = len(rhos), len(thetas)

    ys, xs = np.nonzero(edge_image)
    if ys.size == 0:
        return np.zeros((num_rhos, num_thetas)), rhos, thetas

    edge_vals = edge_image[ys, xs].astype(np.float64)

    # Compute rho for all points and thetas at once: [N, T]
    rho_vals = xs[:, None] * cos_t[None, :] + ys[:, None] * sin_t[None, :]  # Shape: [N, T]
    rho_idx = np.round((rho_vals + diag_len) / rho_res).astype(np.int64)

    # Mask out-of-range bins (can happen due to rounding)
    valid = (rho_idx >= 0) & (rho_idx < num_rhos)
    if not valid.any():
        return np.zeros((num_rhos, num_thetas)), rhos, thetas

    theta_idx = np.broadcast_to(np.arange(num_thetas, dtype=np.int64), rho_idx.shape) # Shape: [N, T]
    flat_idx = rho_idx[valid] * num_thetas + theta_idx[valid]  # Shape: [M] where M is number of valid votes
    weights = np.broadcast_to(edge_vals[:, None], rho_vals.shape)[valid]  # Shape: [M]

    # Bin counts into flat accumulator then reshape
    acc_flat = np.bincount(flat_idx, weights=weights, minlength=num_rhos * num_thetas)
    cnt_flat = np.bincount(flat_idx, minlength=num_rhos * num_thetas)

    with np.errstate(divide="ignore", invalid="ignore"):
        acc_mean = np.divide(
            acc_flat, 
            cnt_flat, 
            out=np.zeros_like(acc_flat, dtype=np.float32), 
            where=cnt_flat > 0
        )

    accumulator = acc_mean.reshape(num_rhos, num_thetas)
    return accumulator, rhos, thetas


def find_hough_peaks(accumulator, num_peaks=5, threshold=1000):
    peaks = []
    acc = accumulator.copy()

    for _ in range(num_peaks):
        idx = np.argmax(acc)
        rho_idx, theta_idx = np.unravel_index(idx, acc.shape)
        
        if acc[rho_idx, theta_idx] < threshold:
            break

        peaks.append((rho_idx, theta_idx, acc[rho_idx, theta_idx]))
        
        # Zero out region around the peak to prevent finding multiple nearby peaks
        acc[max(0, rho_idx-10):rho_idx+10, max(0, theta_idx-10):theta_idx+10] = 0

    return peaks


def dominant_orientation_from_accumulator(accumulator: np.ndarray, thetas: np.ndarray, top_k: int = 1):
    """
    Return the strongest theta angles from a Hough accumulator.
    """
    if accumulator.size == 0:
        return []
    theta_scores = accumulator.sum(axis=0)
    idx = np.argsort(theta_scores)[::-1][:top_k]
    return [(float(thetas[i]), float(theta_scores[i])) for i in idx]


def rotate_image_to_orientation(image: np.ndarray, theta_rad: float, target_angle_deg: float = 0.0) -> np.ndarray:
    """
    Rotate image so the dominant Hough line at theta_rad aligns to target_angle_deg.
    """
    angle_deg = target_angle_deg - np.degrees(theta_rad)
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)


def overlay_hough_lines(
    image: np.ndarray,
    peaks: list[tuple[int, int, float]],
    rhos: np.ndarray,
    thetas: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw the highest-voted lines identified in the Hough accumulator.
    """
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = image.shape[:2]
    diag = int(np.hypot(h, w))

    for rho_idx, theta_idx, _ in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + diag * (-b)), int(y0 + diag * a))
        pt2 = (int(x0 - diag * (-b)), int(y0 - diag * a))
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    return overlay
