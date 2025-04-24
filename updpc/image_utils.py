import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tifffile import imread
import pandas as pd
from collections import defaultdict


def font(size):
    """
    Set the font size for matplotlib plots.

    Parameters
    ----------
    size : int
        The font size to set for the plots.
    """
    rcParams["font.size"] = size
    rcParams["axes.labelsize"] = size
    rcParams["axes.titlesize"] = size
    rcParams["xtick.labelsize"] = size
    rcParams["ytick.labelsize"] = size
    rcParams["legend.fontsize"] = size


def figure(figsize=(10, 10)):
    """
    Create a new figure with the specified size.

    Parameters
    ----------
    figsize : tuple, optional
        The size of the figure to create. Default is (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object created.
    """
    return plt.figure(figsize=figsize)


def imwrite(
    path, img, axes=None, pixel_size_um=None, imagej=True, metadata={}, **kwargs
):
    """
    Write an image to a TIFF file for ImageJ compatibility.
    By default, the axes metadata is set based on the number of dimensions of the input image: "YX" for 2D, "TYX" for 3D, and "TZYX" for 4D.

    Parameters
    ----------
    path : str
        The path to save the TIFF file.
    img : numpy.ndarray
        The image data to save.
    axes : str, optional
        The axes metadata to set for the image.
        If None, the axes metadata is set based on the number of dimensions of the input image.
        Default is None.
    imagej : bool, optional
        Whether to save the image with ImageJ metadata. Default is True.
    metadata : dict, optional
        Additional metadata to save with the image. Default is an empty dictionary.
    """
    if axes is None:
        dim = len(img.shape)
        if dim == 2:
            metadata["axes"] = "YX"
        elif dim == 3:
            metadata["axes"] = "TYX"
        elif dim == 4:
            metadata["axes"] = "TZYX"
    else:
        metadata["axes"] = axes

    if pixel_size_um is not None:
        metadata["unit"] = "um"
        resolution = 1 / pixel_size_um
        tifffile.imwrite(
            path,
            img,
            imagej=imagej,
            metadata=metadata,
            resolution=(resolution, resolution),
            **kwargs,
        )
    else:
        tifffile.imwrite(path, img, imagej=imagej, metadata=metadata, **kwargs)


def imwrite_f32(path, img, **kwargs):
    """
    Write an image to a TIFF file as a 32-bit floating-point numpy array.

    Parameters
    ----------
    path : str
        The path to save the TIFF file.
    img : numpy.ndarray
        The image data to save.
    kwargs : dict
        Additional keyword arguments to pass to tifffile.imwrite.
    """
    imwrite(path, np.array(img, dtype=np.float32), imagej=True, **kwargs)


def imwrite_uint8(path, img, **kwargs):
    """
    Write an image to a TIFF file as a 8-bit unsigned integer numpy array.

    Parameters
    ----------
    path : str
        The path to save the TIFF file.
    img : numpy.ndarray
        The image data to save.
    kwargs : dict
        Additional keyword arguments to pass to tifffile.imwrite.
    """
    imwrite(path, np.array(img, dtype=np.uint8), imagej=True, **kwargs)


def imwrite_int16(path, img, **kwargs):
    """
    Write an image to a TIFF file as a 16-bit integer numpy array.

    Parameters
    ----------
    path : str
        The path to save the TIFF file.
    img : numpy.ndarray
        The image data to save.
    kwargs : dict
        Additional keyword arguments to pass to tifffile.imwrite.
    """
    imwrite(path, np.array(img, dtype=np.int16), imagej=True, **kwargs)


def cb(var=""):
    """
    Add a colorbar to the current plot.

    Parameters
    ----------
    var : str, optional
        The variable name to display on the colorbar. Default is an empty string.
    """
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(var)


def axcb(ax, mappable, clabel=""):
    """
    Add a colorbar to the specified axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the colorbar to.
    mappable : matplotlib.cm.ScalarMappable
        The mappable object to use for the colorbar.
    clabel : str, optional
        The label to display on the colorbar. Default is an empty string.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(clabel)


def ax_scatter(ax, x, y, c, clabel="", cmap="hsv", **kwargs):
    """
    Create a scatter plot on the specified axes with a clear colorbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to create the scatter plot on.
    x : numpy.ndarray
        The x-coordinates of the points to plot.
    y : numpy.ndarray
        The y-coordinates of the points to plot.
    c : numpy.ndarray
        The color values of the points to plot.
    clabel : str, optional
        The label to display on the colorbar. Default is an empty string.
    cmap : str, optional
        The colormap to use for the scatter plot. Default is "hsv".
    """
    plot = ax.scatter(x, y, c=c, cmap=cmap, **kwargs)
    axcb(ax, plot, clabel)


def ax_imshow(ax, img, clabel="", cmap="gray", cbar=True, title=None, **kwargs):
    """
    Display an image on the specified axes with clear colorbar and title.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to display the image on.
    img : numpy.ndarray
        The image data to display.
    clabel : str, optional
        The label to display on the colorbar. Default is an empty string.
    cmap : str, optional
        The colormap to use for displaying the image. Default is "gray".
    cbar : bool, optional
        Whether to display a colorbar. Default is True.
    title : str, optional
        The title to display above the image. Default is None.
    """
    print("ax_imshow is deprecated. Use imshow(ax=ax) instead.")
    plot = ax.imshow(img, cmap=cmap, **kwargs)
    if cbar:
        axcb(ax, plot, clabel)
    if title is not None:
        ax.set_title(title)


def imshow(
    img,
    figsize=(10, 10),
    title="",
    clabel="",
    cmap="gray",
    cbar=True,
    ax=None,
    **kwargs,
):
    """
    Display an image using matplotlib with clear colorbar and title.

    Parameters
    ----------
    img : numpy.ndarray
        The image data to display.
    figsize : tuple, optional
        The size of the figure to display the image on. Default is (10, 10).
    title : str, optional
        The title to display above the image. Default is an empty string.
    clabel : str, optional
        The label to display on the colorbar. Default is an empty string.
    cmap : str, optional
        The colormap to use for displaying the image. Default is "gray".
    cbar : bool, optional
        Whether to display a colorbar. Default is True.
    ax : matplotlib.axes.Axes, optional
        The axes to display the image on. Default is None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object used to display the image.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    plot = ax.imshow(img, cmap=cmap, **kwargs)
    if cbar:
        axcb(ax, plot, clabel)
    if title is not None:
        ax.set_title(title)
    return ax


def imimshow(img, figsize=(15, 4), cmap="gray", title="", **kwargs):
    """
    Display an image with real, imaginary, and absolute parts side by side.

    Parameters
    ----------
    img : numpy.ndarray
        The image data to display.
    figsize : tuple, optional
        The size of the figure to display the image on. Default is (20, 10).
    cmap : str, optional
        The colormap to use for displaying the image. Default is "gray".
    title : str, optional
        The title to display above the image. Default is an empty string.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax = axes[0]
    imshow(img.real, cmap=cmap, ax=ax)
    ax.set_title("real")
    ax.set_aspect(1)
    ax = axes[1]
    imshow(img.imag, cmap=cmap, ax=ax)
    ax.set_title("imag")
    ax.set_aspect(1)
    ax = axes[2]
    imshow(np.abs(img), cmap=cmap, ax=ax)
    ax.set_title("abs")
    ax.set_aspect(1)
    plt.tight_layout()
    fig.suptitle(title)


def hexbin(x, y, gridsize=50, cmap="Reds", ax=None, **kwargs):
    """
    Create a hexbin plot of (x, y) with a colorbar.

    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates of the points to plot.
    y : numpy.ndarray
        The y-coordinates of the points to plot.
    cmap: str, optional
        The colormap to use. Default is "Blues".
    ax : matplotlib.axes.Axes, optional
        The axes to create the hexbin plot on. Default is None.
    kwargs : dict
        Additional keyword arguments to pass to matplotlib.pyplot.hexbin.
    """
    if ax is None:
        _, ax = plt.subplots()
    plot = ax.hexbin(x, y, mincnt=1, vmin=0, gridsize=gridsize, cmap=cmap, **kwargs)
    axcb(ax, plot)


def compress(img, compress_num, interpolation=cv2.INTER_AREA, **kwargs):
    """
    Compress an image by a specified factor. Default interpolation is cv2.INTER_AREA.

    Parameters
    ----------
    img : numpy.ndarray
        The image data to compress.
    compress_num : int
        The factor by which to compress the image.
    interpolation : int, optional
        The interpolation method to use. Default is cv2.INTER_AREA.

    Returns
    -------
    numpy.ndarray
        The compressed image data.
    """
    fxy = 1 / compress_num
    return cv2.resize(
        img,
        dsize=None,
        fx=fxy,
        fy=fxy,
        interpolation=interpolation,
        **kwargs,
    )


def contrast(img, Imin, Imax, return_int=False):
    """
    Adjust the contrast of an image by rescaling the intensity values to the range [0, 255].
    By default, the output image is returned as a floating-point numpy array.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to adjust the contrast of.
    Imin : float
        The minimum intensity value to map to 0.
    Imax : float
        The maximum intensity value to map to 255.
    return_int : bool, optional
        Whether to return the output image as a floating-point numpy array or as a uint8 numpy array. Default is False.

    Returns
    -------
    numpy.ndarray
        The adjusted image with contrast rescaled to the range [0, 255].
    """
    ret = img.astype(float) - Imin
    ret = ret * 255 / (Imax - Imin)
    ret = np.clip(ret, 0, 255)
    return ret.astype(np.uint8) if return_int else ret


def contrast_array(imgs, Imin, Imax, return_int=False):
    """
    Adjust the contrast of a list of images by rescaling the intensity values to the range [0, 255].
    By default, the output images are returned as floating-point numpy arrays.

    Parameters
    ----------
    imgs : list of numpy.ndarray
        The input images to adjust the contrast of.
    Imin : float
        The minimum intensity value to map to 0.
    Imax : float
        The maximum intensity value to map to 255.
    return_int : bool, optional
        Whether to return the output images as floating-point numpy arrays or as uint8 numpy arrays. Default is False.

    Returns
    -------
    numpy.ndarray
        The adjusted images with contrast rescaled to the range [0, 255].
    """
    ret = [contrast(img, Imin, Imax, return_int=True) for img in imgs]
    return np.array(ret, dtype=np.uint8) if return_int else np.array(ret)


def auto_contrast(img, ignore_percent=2, return_int=False):
    """
    Adjust the contrast of an image automatically.
    The contrast is adjusted by rescaling the intensity values to the range [0, 255].
    The minimum and maximum intensity values are computed by ignoring `ignore_percent` of the pixel in the bottom and top of the intensity distribution.
    By default, the output image is returned as a floating-point numpy array.

    Parameters
    ----------
    img: numpy.ndarray
        The input image to adjust the contrast of.
    ignore_percent: int, optional
        The percentage of pixels to ignore in the bottom and top of the intensity distribution.
        Default is 2.
    return_int: bool, optional
        Whether to return the output image as a floating-point numpy array or as a uint8 numpy array.
        Default is False.

    Returns
    -------
    numpy.ndarray
        The adjusted image with contrast rescaled to the range [0, 255].
        If `return_int` is True, the output image is returned as a uint8 numpy array.
        Otherwise, the output image is returned as a floating-point numpy array.
    """
    ret = img - np.percentile(img, ignore_percent)
    ret = ret * 255 / np.percentile(ret, 100 - ignore_percent)
    ret = np.clip(ret, 0, 255)
    return ret.astype(np.uint8) if return_int else ret


def auto_contrast_nsigma(img, nsigma=2):
    """
    Adjust the contrast of an image automatically.
    The minimum and maximum intensity values are computed by ignoring the values that are more than `nsigma` standard deviations away from the mean.

    Parameters
    ----------
    img: numpy.ndarray
        The input image to adjust the contrast of.
    nsigma: float, optional
        The number of standard deviations to consider as outliers.
        Default is 2.

    Returns
    -------
    numpy.ndarray
        The adjusted image clipped to the range [mean - nsigma * std, mean + nsigma * std].
    """
    mean = img.mean()
    std = img.std()
    return np.clip(img, mean - nsigma * std, mean + nsigma * std)


def auto_contrast_by_values(img, base, ratio, return_int=False):
    """
    The contrast is adjusted by mapping the intensity values to the range [0, 255] using the formula `(img - base) * 255 / ratio`.
    In other words, the minimum intensity value `base` is mapped to 0 and the maximum intensity value `base + ratio` is mapped to 255.
    By default, the output image is returned as a floating-point numpy array.

    Parameters
    ----------
    img: numpy.ndarray
        The input image to adjust the contrast of.
    base: float
        The minimum intensity value to map to 0.
    ratio: float
        The maximum intensity value to map to 255.
    return_int: bool, optional
        Whether to return the output image as a floating-point numpy array or as a uint8 numpy array.
        Default is False.

    Returns
    -------
    numpy.ndarray
        The adjusted image with contrast rescaled to the range [0, 255].
        If `return_int` is True, the output image is returned as a uint8 numpy array.
        Otherwise, the output image is returned as a floating-point numpy array.
    """

    ret = img - base
    ret = ret * 255 / ratio
    ret = np.clip(ret, 0, 255)
    return ret.astype(np.uint8) if return_int else ret


def make_base_ratio(imgs, ignore_percent):
    """
    Calculate the base and ratio values for the contrast adjustment of a list of images.
    The base value is the minimum intensity value to map to 0.
    The ratio value is the maximum intensity value minus the base value to map to 255.
    The base and ratio values are calculated by ignoring `ignore_percent` of the pixels in the bottom and top of the intensity distribution.
    Namely, `base = np.percentile(imgs, ignore_percent)` and `ratio = np.percentile(imgs, 100 - ignore_percent) - base`.

    Parameters
    ----------
    imgs: list of numpy.ndarray
        The input images to calculate the base and ratio values for.
    ignore_percent: int
        The percentage of pixels to ignore in the bottom and top of the intensity distribution.

    Returns
    -------
    base: float
        The minimum intensity value to map to 0.
    ratio: float
        The maximum intensity value to map to 255.
    """

    base = np.percentile(imgs, ignore_percent)
    ratio = np.percentile(imgs, 100 - ignore_percent) - base
    return base, ratio


def auto_contrast_array(imgs, ignore_percent=2, return_int=False):
    """
    Adjust the contrast of a list of images automatically.

    Parameters
    ----------
    imgs: list of numpy.ndarray
        The input images to adjust the contrast of.
    ignore_percent: int, optional
        The percentage of pixels to ignore in the bottom and top of the intensity distribution.
        Default is 2.
    return_int: bool, optional
        Whether to return the output images as floating-point numpy arrays or as uint8 numpy arrays.
        Default is False.

    Returns
    -------
    numpy.ndarray
        The adjusted images with contrast rescaled to the range [0, 255].
        If `return_int` is True, the output images are returned as uint8 numpy arrays.
        Otherwise, the output images are returned as floating-point numpy arrays.
    """
    return (
        np.array(
            [auto_contrast(img, ignore_percent, return_int=True) for img in imgs],
            dtype=np.uint8,
        )
        if return_int
        else np.array([auto_contrast(img, ignore_percent) for img in imgs])
    )


def get_tiff_shape(file_path):
    """
    Get the shape of the first image in a TIFF file.

    Parameters
    ---------
    file_path: str
        The path to the TIFF file.

    Returns
    -------
    tuple:
        The shape of the first image in the TIFF file.
    """
    with tifffile.TiffFile(file_path) as tif:
        shape = tif.pages[0].shape
    return shape


def intensity_on_circle(
    img, cy, cx, rad, ax=None, angle_point_num=None, show_image=True
):
    """
    Calculate the intensity values along a circle of a given radius centered at a specified point.

    Parameters
    ----------
    img : numpy.ndarray
        The 2D image to analyze.
    cy : int
        The y-coordinate of the center of the circle.
    cx : int
        The x-coordinate of the center of the circle.
    rad : float
        The radius of the circle.
    ax : matplotlib.axes.Axes, optional
        The axes to display the image with the circle. Default is None. If None, a new figure is created.
    angle_point_num : int, optional
        The number of points to sample along the circle. Default is None.
        If None, the number of points is calculated based on the circumference.
    show_image : bool, optional
        Whether to display the image with the circle. Default is True.

    Returns
    -------
    theta : numpy.ndarray
        The angles of the points on the circle. The angles are in the range [0, 2*pi] radians.
    intensity : list
        The intensity values along the circle.
    """

    if angle_point_num is None:
        angle_point_num = int(2 * np.pi * rad) + 1

    # Define the coordinates of the circle
    theta = np.linspace(0, 2 * np.pi, angle_point_num)
    x = cx + rad * np.cos(theta)
    y = cy + rad * np.sin(theta)

    # Calculate the intensity values along the circle
    intensity = [img[int(round(yy)), int(round(xx))] for xx, yy in zip(x, y)]

    if show_image:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax_imshow(ax, img, cmap="gray")
        circle = plt.Circle((cx, cy), rad, color="r", fill=False)
        ax.add_artist(circle)

    return theta, intensity


# def mean_without_outliers(imgs, nsigma=3, each_pixel=True, zero_sigma=False):
#     """
#     Calculate the mean of a list of images without outliers.
#     Outliers are defined as the values that are more than `nsigma` standard deviations away from the mean.
#     By default, the mean is calculated for each pixel across the images.
#     If `each_pixel` is False, the outliers are detected based on the mean and standard deviation of the pixel values across the images.
#     If `zero_sigma` is True, the standard deviation of the pixel values is set to 1e-6 if it is zero.

#     Parameters
#     ----------
#     imgs : list of numpy.ndarray
#         The input images to calculate the mean without outliers.
#     nsigma : float, optional
#         The number of standard deviations to consider as outliers. Default is 3.
#     each_pixel : bool, optional
#         Whether to calculate the mean for each pixel across the images. Default is True.
#     zero_sigma : bool, optional
#         Whether to set the standard deviation of the pixel values to 1e-6 if it is zero. Default is False.

#     Returns
#     -------
#     numpy.ndarray
#         The mean of the images without outliers.
#     """
#     mean_img = np.mean(imgs, axis=0)
#     std_img = np.std(imgs, axis=0)
#     if np.any(std_img == 0):
#         if zero_sigma:
#             std_img[std_img == 0] = 1e-6
#             print("Zero standard deviation is replaced with 1e-6.")
#         else:
#             print("Warning: Zero standard deviation is detected.")
#     in_nsigma = np.abs(imgs - mean_img) < nsigma * std_img
#     print("Frame length: ", len(imgs))
#     print("out of nsigma number: ", in_nsigma.size - in_nsigma.sum())
#     print("out of nsigma rate: ", 1 - in_nsigma.sum() / in_nsigma.size)
#     if each_pixel:
#         in_nsigma = in_nsigma.astype(int)
#         return (imgs * in_nsigma).sum(axis=0) / in_nsigma.sum(axis=0)
#     else:
#         print(
#             "Outliers are detected based on the mean and standard deviation of the pixel values across the images."
#         )
#         inlier_indices = np.all(in_nsigma, axis=(1, 2))
#         print("outlier number:", inlier_indices.size - inlier_indices.sum())
#         print("outlier rate:", 1 - inlier_indices.sum() / inlier_indices.size)
#         return imgs[inlier_indices].mean(axis=0)


def mean_without_outliers(arrays, nsigma=3, each_element=True, zero_sigma=False):
    """
    Calculate the mean of a list of arrays without outliers.
    Outliers are defined as the values that are more than `nsigma` standard deviations away from the mean.
    By default, the mean is calculated for each element across the arrays.
    If `each_element` is False, the outliers are detected based on the mean and standard deviation of the entire arrays.
    If `zero_sigma` is True, the standard deviation is set to 1e-6 where it is zero to avoid division by zero.

    Parameters
    ----------
    arrays : list of numpy.ndarray or numpy.ndarray
        The input arrays to calculate the mean without outliers.
    nsigma : float, optional
        The number of standard deviations to consider as outliers. Default is 3.
    each_element : bool, optional
        Whether to calculate the mean for each element across the arrays. Default is True.
    zero_sigma : bool, optional
        Whether to set the standard deviation to 1e-6 where it is zero. Default is False.

    Returns
    -------
    numpy.ndarray
        The mean of the arrays without outliers.
    """
    arrays = np.array(arrays)
    mean_array = np.mean(arrays, axis=0)
    std_array = np.std(arrays, axis=0)
    if np.any(std_array == 0):
        if zero_sigma:
            std_array[std_array == 0] = 1e-6
            print("Zero standard deviation is replaced with 1e-6.")
        else:
            print("Warning: Zero standard deviation is detected.")
    in_nsigma = np.abs(arrays - mean_array) < nsigma * std_array
    print("Total number of elements:", in_nsigma.size)
    print("Number of outliers:", in_nsigma.size - in_nsigma.sum())
    print("Outlier rate:", (1 - in_nsigma.sum() / in_nsigma.size) * 100, "%")
    if each_element:
        in_nsigma = in_nsigma.astype(int)
        return (arrays * in_nsigma).sum(axis=0) / in_nsigma.sum(axis=0)
    else:
        print("Outliers are detected based on the entire arrays.")
        # Compute over all dimensions except the first (which indexes the arrays)
        axes = tuple(range(1, arrays.ndim))
        inlier_indices = np.all(in_nsigma, axis=axes)
        print("Number of outlier arrays:", inlier_indices.size - inlier_indices.sum())
        print(
            "Outlier array rate:",
            (1 - inlier_indices.sum() / inlier_indices.size) * 100,
            "%",
        )
        return arrays[inlier_indices].mean(axis=0)


# ANCHOR - CV2


def shift_image(img, tx, ty):
    """
    Shift an image by a specified translation vector.
    Positive values shift the image to the right and down.
    If the translation is integer, the image is shifted by the corresponding number of pixels.

    Parameters
    ----------
    img : numpy.ndarray
        The image data to shift.
    tx : float
        The translation in the x-direction.
    ty : float
        The translation in the y-direction.

    Returns
    -------
    numpy.ndarray
        The shifted image data.
    """
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=float)
    return cv2.warpAffine(img.astype(float), translation_matrix, dsize=img.shape[::-1])


def blur(img, sigma, ksize=None):
    """
    Apply a Gaussian blur to an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image data to blur.
    sigma : float
        The standard deviation of the Gaussian kernel.
    ksize : int, optional
        The kernel size for the Gaussian blur. Default is None.
        If None, the kernel size is calculated as int(2 * round(3 * sigma) + 1).

    Returns
    -------
    numpy.ndarray
        The blurred image data.
    """
    if ksize is None:
        ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def findContours(img):
    """
    Find contours in a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        The binary image to find contours in.

    Returns
    -------
    tuple
        A tuple of contours and hierarchy.
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def center_contour(c):
    """
    Calculate the center of a contour.

    Parameters
    ----------
    c : numpy.ndarray
        The contour to calculate the center of.

    Returns
    -------
    tuple
        The x and y coordinates of the center of the contour.
    """
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def dilate(img, ksize=15):
    """
    Dilate a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        The binary image to dilate.
    ksize : int, optional
        The kernel size for the dilation. Default is 15.

    Returns
    -------
    numpy.ndarray
        The dilated binary image.
    """
    return cv2.dilate(
        img.astype(np.uint8),
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)),
    )


def erode(img, ksize=15):
    """
    Erode a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        The binary image to erode.
    ksize : int, optional
        The kernel size for the erosion. Default is 15.

    Returns
    -------
    numpy.ndarray
        The eroded binary image.
    """
    return cv2.erode(
        img.astype(np.uint8),
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)),
    )


def fill(img, ksize=15):
    """
    Fill holes in a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        The binary image to fill holes in.
    ksize : int, optional
        The kernel size for the dilation and erosion. Default is 15.

    Returns
    -------
    numpy.ndarray
        The binary image with holes filled.
    """
    return erode(dilate(img, ksize), ksize)


def read_all_metadata(path, verbose=True):
    """
    Read the path to TIF, and return all the metadata.

    Returns
    ------
    metadata: pd.DataFrame of metadata common to every pages.
    df: pd.DataFrame of page-dependent metadata.
    """

    tmp = defaultdict(list)

    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            for tag in page.tags.values():
                name, value = tag.name, tag.value
                tmp[name].append(value)

    tmp_single = defaultdict(list)
    cols_single = []
    for col in tmp.keys():
        if isinstance(tmp[col][0], dict) or len(set(tmp[col][1:])) <= 1:
            tmp_single[col] = [tmp[col][0]]
            cols_single.append(col)

    for col in cols_single:
        tmp.pop(col)

    metadata = pd.DataFrame(tmp_single)
    df = pd.DataFrame(tmp)
    if verbose:
        for col in tmp.keys():
            print_info(len(tmp[col]))
        display(metadata.T)
        display(df)

    return metadata, df


# SECTION - Metadata
# ANCHOR - Thorcam Metadata
tsi_tag_names = {
    "32768": "SIGNIFICANT_BITS_INT",
    "32770": "BIN_X_INT",
    "32771": "BIN_Y_INT",
    "32772": "ROI_ORIGIN_X_INT",
    "32773": "ROI_ORIGIN_Y_INT",
    "32774": "ROI_PIXELS_X_INT",
    "32775": "ROI_PIXELS_Y_INT",
    "32776": "EXPOSURE_TIME_US",
    "32777": "PIXEL_CLOCK_HZ_INT",
    "32778": "NUM_TAPS_INT",
    "32779": "FRAME_NUMBER_INT",
    "32780": "POLARIZATION_IMAGE_TYPE",
    "32781": "TIME_STAMP_RELATIVE_NS_HIGH_BITS",
    "32782": "TIME_STAMP_RELATIVE_NS_LOW_BITS",
}


def analyze_frametime(df, stamp_rate=5.3):
    """
    Analyze the frame time of a ThorCam TSI image.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the metadata of the TSI image.
    stamp_rate : float, optional
        The rate of the timestamp in MHz. Default is 5.3.

    Returns
    -------
    None
    """
    frame = df["FRAME_NUMBER_INT"] - df["FRAME_NUMBER_INT"].min()
    stamp = (
        (
            df["TIME_STAMP_RELATIVE_NS_LOW_BITS"]
            - df["TIME_STAMP_RELATIVE_NS_LOW_BITS"].min()
        )
        / 10**6
        * stamp_rate
    )

    fig, axes = plt.subplots(1, 5, figsize=(4 * 5, 3))
    dstamp = stamp.diff().dropna()
    dframe = frame.diff().dropna()
    dstamp[dstamp < 0] += 2**32 / 10**6 * stamp_rate

    ax = axes[0]
    ax.plot(frame, marker=".")
    ax.set_xlabel("Frame number")
    ax.set_title("Frame index")

    ax = axes[1]
    ax.plot(dframe, marker=".")
    ax.set_xlabel("Frame number")
    ax.set_title("Frame index diff.")
    plt.tight_layout()

    ax = axes[2]
    ax.hist(dstamp, bins="sturges")
    ax.set_xlabel("Frame time")
    ax.set_title(
        f"All, Mean: {round(dstamp.mean(), 1)} ms ({round(1000/dstamp.mean())} FPS)"
    )

    ax = axes[3]
    dstamp_2sigma = dstamp[np.abs(dstamp - dstamp.mean()) < 2 * dstamp.std()]
    ax.hist(dstamp_2sigma, bins="sturges")
    ax.set_xlabel("Frame time")
    ax.set_title(
        f"2 StdDev around Mean, Mean: {round(dstamp_2sigma.mean(), 1)} ms ({round(1000/dstamp_2sigma.mean())} FPS)"
    )

    ax = axes[4]
    time1frame = dstamp / dframe
    ax.hist(time1frame, bins="sturges")
    ax.set_xlabel("\nFrame time for 1 frame index [ms]")
    ax.set_title(
        f"1 frame, Mean: {round(time1frame.mean(), 1)} ms ({round(1000/time1frame.mean())} FPS)"
    )


def read_tsi_metadata(path, frame_col="", ns_col="", verbose=True):
    """
    Read the path to TIF taken by ThorCam, and return the metadata.

    Parameters
    ----------
    path : str
        The path to the TIF file.
    frame_col : str, optional
        The name of the column for the frame number. Default is "".
    ns_col : str, optional
        The name of the column for the timestamp. Default is "".
    verbose : bool, optional
        Whether to display the metadata. Default is True.

    Returns
    ------
    metadata : pd.DataFrame
        The metadata common to every pages.
    df: pd.DataFrame of FRAME_NUMBER_INT and TIME_STAMP_RELATIVE_NS_LOW_BITS
        The metadata of the TSI image.
    """
    multi_tag_name = ["32779", "32782"]

    with tifffile.TiffFile(path) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = [value]

        multi_tags = {}
        for name in multi_tag_name:
            multi_tags[name] = tif_tags[name].copy()

        for page in tif.pages[1:]:
            for name in multi_tag_name:
                multi_tags[name].append(page.tags[name].value)

    metadata = pd.DataFrame(tif_tags)
    metadata = metadata.rename(columns=tsi_tag_names)

    df = pd.DataFrame(multi_tags)
    if frame_col == "":
        frame_col = "FRAME_NUMBER_INT"
    if ns_col == "":
        ns_col = "TIME_STAMP_RELATIVE_NS_LOW_BITS"
    df = df.rename(columns={"32779": frame_col, "32782": ns_col})

    if verbose:
        analyze_frametime(df)
        display(metadata.T)

    return metadata, df


def timestamp_tsi(path):
    """
    Read the path to TIF taken by ThorCam, and return the timestamp.

    Parameters
    ----------
    path : str
        The path to the TIF file.

    Returns
    -------
    numpy.ndarray
        The timestamp of the TSI image.
    """
    stamp_key = str(TAG_FRAME)
    stamp = []

    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            stamp.append(page.tags[stamp_key].value)

    return np.array(stamp)


# ANCHOR - Pycromanager-based handmade code Metadata
TAG_BITDEPTH = 32768
TAG_EXPOSURE = 32776
TAG_FRAME = 32779
TAG_TIME = 32782
NUMBER_OF_IMAGES = 40


pytsi_tag_names = {
    str(TAG_BITDEPTH): "SIGNIFICANT_BITS_INT",
    str(TAG_EXPOSURE): "EXPOSURE_TIME_US",
    str(TAG_FRAME): "FRAME_NUMBER_INT",
    str(TAG_TIME): "TIME_STAMP_RELATIVE_NS",
}


def analyze_timestamp(stamp):
    """
    Analyze the timestamp of a PyTSI image.

    Parameters
    ----------
    stamp : numpy.ndarray
        The timestamp of the PyTSI image.

    Returns
    -------
    None
    """
    _, axes = plt.subplots(1, 5, figsize=(4 * 5, 3))

    ax = axes[0]
    ax.plot(stamp, marker=".")
    ax.set_xlabel("Frame number")
    ax.set_title("Time stamp [ms]")

    ax = axes[1]
    dstamp = stamp.diff().dropna()
    ax.hist(dstamp, bins="sturges")
    ax.set_xlabel("Frame time [ms]")
    if dstamp.mean() > 0:
        ax.set_title(
            f"Mean: {round(dstamp.mean())} ms ({round(1000/dstamp.mean())} FPS)"
        )

    ax = axes[2]
    dstamp_2sigma = dstamp[np.abs(dstamp - np.mean(dstamp)) < 2 * dstamp.std()]
    if len(dstamp_2sigma) == 0:
        dstamp_2sigma = np.array([np.mean(dstamp)])
    if dstamp_2sigma.mean() > 0:
        ax.hist(dstamp_2sigma, bins="sturges")
        ax.set_title(
            f"$2\\sigma$ from mean, Mean: {round(dstamp_2sigma.mean())} ms ({round(1000/dstamp_2sigma.mean())} FPS)"
        )
    ax.set_xlabel("Frame time [ms]")

    ax = axes[3]
    dstamp_min = dstamp[np.abs(dstamp) < np.abs(dstamp).min() * 1.5]
    if len(dstamp_min) > 0:
        ax.hist(dstamp_min, bins="sturges")
        if dstamp_min.mean() > 0:
            ax.set_title(
                f"< 1.5 * Min, Mean: {round(dstamp_min.mean())} ms ({round(1000/dstamp_min.mean())} FPS)"
            )
        ax.set_xlabel("Frame time [ms]")

    ax = axes[4]
    dstamp_int = (dstamp // dstamp.min()).astype(int)
    ax.plot(dstamp_int, marker=".")
    ax.set_xlabel("Frame number")
    ax.set_title("Frame time // Min(Frame time)")
    plt.tight_layout()


def read_pytsi_metadata(path, frame_col="", ns_col="", verbose=True):
    """
    Read the path to TIF taken by ThorCam, and return the metadata.

    Parameters
    ----------
    path : str
        The path to the TIF file.
    frame_col : str, optional
        The name of the column for the frame number. Default is "".
    ns_col : str, optional
        The name of the column for the timestamp. Default is "".
    verbose : bool, optional
        Whether to display the metadata. Default is True.

    Returns
    ------
    metadata : pd.DataFrame
        The metadata common to every pages.
    """
    multi_tag_name = [str(TAG_FRAME), str(TAG_TIME)]

    with tifffile.TiffFile(path) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = [value]

        multi_tags = {}
        for name in multi_tag_name:
            multi_tags[name] = tif_tags[name].copy()

        for page in tif.pages[1:]:
            for name in multi_tag_name:
                multi_tags[name].append(page.tags[name].value)

    metadata = pd.DataFrame(tif_tags)
    metadata = metadata.rename(columns=pytsi_tag_names)

    df = pd.DataFrame(multi_tags)
    if frame_col == "":
        frame_col = "FRAME_NUMBER_INT"
    if ns_col == "":
        ns_col = "TIME_STAMP_RELATIVE_NS"
    df = df.rename(columns={str(TAG_FRAME): frame_col, str(TAG_TIME): ns_col})

    if verbose:
        stamp = (
            df["TIME_STAMP_RELATIVE_NS"] - df["TIME_STAMP_RELATIVE_NS"].min()
        ) / 10**6
        analyze_timestamp(stamp)
        display(metadata.T)

    return metadata, df


# ANCHOR - Metamorph Metadata
def description_to_series(string):
    """
    Convert a string of metadata to a pandas Series.

    Parameters
    ----------
    string : str
        The string of metadata.

    Returns
    -------
    pd.Series
        The metadata as a pandas Series.
    """
    metadata = {}
    pattern = r'<prop id="(.+?)" type="(.+?)" value="(.+?)"\/>'
    matches = re.findall(pattern, string)
    for match in matches:
        metadata[match[0]] = match[2]
    return pd.Series(metadata)


def read_metamorph_metadata(path, verbose=True):
    """
    Read the path to TIF taken by MetaMorph, and return the metadata.

    Returns
    ------
    metadata: pd.DataFrame
        The metadata common to every pages.
    df: pd.DataFrame
        The metadata of the MetaMorph image.
    """

    tmp = []

    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            description = page.tags["ImageDescription"].value
            # print_info(description)
            tmp.append(description_to_series(description))
    tmp = pd.concat(tmp, axis=1).T

    tmp_single = defaultdict(list)
    cols_single = []
    for col in tmp.keys():
        if len(set(tmp[col][1:])) == 1:
            tmp_single[col] = [tmp[col][0]]
            cols_single.append(col)

    for col in cols_single:
        tmp.pop(col)

    metadata = pd.DataFrame(tmp_single)
    # metadata = metadata.rename(columns=tsi_tag_names)

    df = pd.DataFrame(tmp)

    if verbose:
        display(metadata.T)
        display(df)

    return metadata, df


# ANCHOR - Micromanager Metadata


def read_micromanager_metadata(path, verbose=True):
    """
    Read the path to TIF taken by Micronmanager, and return the metadata.

    Parameters
    ----------
    path : str
        The path to the TIF file.

    Returns
    ------
    metadata: pd.DataFrame
        The metadata common to every pages.
    """

    with tifffile.TiffFile(path) as tif:
        metadatas = []
        try:  # OME TIFF
            for page in tif.pages:
                metadata = page.tags["MicroManagerMetadata"].value
                display(
                    pd.Series(metadata)[
                        ["Exposure-ms", "ElapsedTime-ms", "ReceivedTime"]
                    ]
                )
                metadatas.append(metadata)
        except:  # Separated TIFF
            for page in tif.pages:
                metadata = page.tags["IJMetadata"].value
                metadatas.append(metadata)
    if verbose:
        print()
        for metadata in metadatas:
            display(metadata)
    return metadatas


def read_micromanager_time(path):
    """
    Read the path to TIF taken by Micronmanager, and return the timestamp.

    Parameters
    ----------
    path : str
        The path to the TIF file.

    Returns
    -------
    list
        The timestamps of the Micromanager image.
    """
    with tifffile.TiffFile(path) as tif:
        try:  # OME TIFF
            return [
                datetime.strptime(
                    page.tags["MicroManagerMetadata"].value["ReceivedTime"],
                    "%Y-%m-%d %H:%M:%S.%f +0900",
                )
                for page in tif.pages
            ]
        except:  # Separated TIFF
            return [
                datetime.strptime(
                    json.loads(page.tags["IJMetadata"].value["Info"])["ReceivedTime"],
                    "%Y-%m-%d %H:%M:%S.%f +0900",
                )
                for page in tif.pages
            ]
