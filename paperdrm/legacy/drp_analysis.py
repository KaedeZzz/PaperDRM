from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.signal import wiener
from matplotlib import pyplot as plt

def bg_subtraction(samples: list[Image.Image], backgrounds: list[Image.Image], coeff: float = 1.0) -> list[Image.Image]:
    """
    Apply background subtraction to images.
    :param samples: list of all images.
    :param backgrounds: list of all backgrounds.
    :param coeff: coefficient of subtraction.
    :return: a list of normalized images after background subtraction.
    """
    num_samples = len(samples)
    if num_samples != len(backgrounds):
        raise ValueError('The number of samples does not match number of backgrounds')
    if samples[0].size != backgrounds[0].size:
        raise ValueError('The size of samples does not match size of backgrounds')

    # Apply Wiener filter to the backgrounds, the window size parameter is from original MATLAB code
    for i in tqdm(range(num_samples), desc='applying wiener filter'):
        img = backgrounds[i]
        img_array = np.array(img)
        filtered_array = wiener(img_array, mysize=(7, 7))
        filtered_bg = Image.fromarray(np.uint8(filtered_array))
        backgrounds[i] = filtered_bg

    # Normalize images by division by backgrounds
    norm_images = []
    for i in tqdm(range(num_samples), desc='normalizing images'):
        sample = np.array(samples[i]).astype(np.float32)
        back = np.array(backgrounds[i]).astype(np.float32)
        back[back == 0] = 1 # Prevent divide-by-zero
        norm = (sample / back) / coeff * 255
        norm_images.append(Image.fromarray(np.uint8(norm)))

    return norm_images


# def display_drp(drp_array: np.ndarray, param: ImageParam, cmap='jet', project: str = 'stereo', ax = None, scalebar: bool = True):
#     """
#     Returns a matplotlib axis of a DRP in polar coordinates.
#     :param drp_array: 2D DRP dataset in dimensions [th_num, ph_num].
#     :param param: an ImageParam instance describing the angle profile of DRP dataset.
#     :param cmap: matplotlib colormap name.
#     :param project: projection method used, either 'stereo' or 'direct'.
#     :param ax: matplotlib axis to draw on.
#     :param scalebar: boolean, whether to display a scalebar on the plot.
#     :return: a matplotlib axis of the DRP in polar coordinates.
#     """
#     # Normalize pixel value into int if they were float in [0,1]
#     if np.issubdtype(drp_array.dtype, np.floating) and drp_array.max() <= 1.0:
#         drp_array = (drp_array * 255).astype(np.uint8)

#     # Meshgrid of phi and theta
#     phi, theta = np.meshgrid(np.linspace(0, 360 + param.ph_step, param.ph_num + 1),
#                              np.linspace(param.th_min, param.th_max + param.th_step, param.th_num + 1))

#     # Projection mapping, from angles to x-y plane
#     if project == "stereo":
#         xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
#         yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
#     elif project == "direct":
#         xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
#         yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
#     else:
#         raise ValueError("Unknown project type. Use 'stereo' or 'direct'.")

#     # create axis if none provided
#     if ax is None:
#         fig, ax = plt.subplots()
#     # plot the color mesh
#     h = ax.pcolormesh(xx, yy, drp_array.T, cmap=cmap, shading='auto')
#     ax.set_aspect('equal') # This ensures projected mesh is circular, not elliptical
#     ax.axis('off')
#     if scalebar:
#         plt.colorbar(h, ax=ax)
#     return ax

# def drp_measure(image_pack: Images) -> list:
#     """
#     Let the user select points, then calculate and display the DRP for each point.
#     :param image_pack: ImagePack instance of input images and DRP parameters.
#     :return: DRP measurement for each point. The DRP matrix is in dimensions [th_num x ph_num].
#     """
#     fig, ax = plt.subplots()
#     images, params = image_pack
#     img_sample = images[0]
#     ax.imshow(img_sample, cmap="gray") # show the sample image first
#     ax.set_title("Click on image for DRP points, press ENTER to stop")
#     points = plt.ginput(n=-1, timeout=30)  # unlimited clicks, press Enter to stop, 30 seconds timeout
#     plt.close(fig)

#     if len(points) == 0:
#         print("No points selected.")
#         return []

#     num_points = len(points)
#     x = [int(p[0]) for p in points]
#     y = [int(p[1]) for p in points]

#     fig, axs = plt.subplots(num_points + 1)
#     # First subplot shows original image with markers
#     axs[0].imshow(img_sample, cmap="gray")
#     axs[0].scatter(x, y, c='r', marker='x', s=100)
#     for i in range(num_points):
#         axs[0].text(x[i] + 5, y[i] + 5, str(i + 1), fontsize=12, color='yellow')
#     axs[0].set_title("Selected Points")

#     drp_measurement = []
#     # calculate DRPs
#     for i in tqdm(range(num_points), desc='calculating DRP measurements'):
#         col, row = x[i], y[i]
#         drp_list = [images[k].getpixel((col, row)) for k in range(len(images))]
#         drp = np.reshape(drp_list, (params.ph_num, params.th_num))
#         drp = drp.T # in consistency with custom display function
#         drp_measurement.append(drp)

#         # Show DRP using custom display function
#         axs[i + 1] = display_drp(drp, ax=axs[i + 1])
#         axs[i + 1].set_title(f"DRP of point {i + 1}")

#     plt.tight_layout()
#     plt.show()
#     return drp_measurement


# def area_mean_drp(image_pack: Images, display: bool = False) -> np.ndarray:
#     """
#     Calculate and display DRP over an area on the image.
#     :param image_pack: ImagePack instance of input images and DRP parameters.
#     :param display: whether to display the DRP plot.
#     :return: a 2D Numpy array representing the DRP data. [th_num x ph_num]
#     """

#     images, params = image_pack
#     ph_num, th_num = params.ph_num, params.th_num
#     w, h = images[0].size[0], images[0].size[1]
#     drp_array = np.zeros((ph_num, th_num))
#     for i in tqdm(range(w * h), desc='Calculating area-mean DRP'):
#         col = i // h
#         row = i % h
#         drp_array += (drp(image_pack, loc=(col, row)) / (w * h))
#     return drp_array

