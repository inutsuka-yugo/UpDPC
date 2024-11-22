import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import binned_statistic
import cv2


# ANCHOR - Analytical least squares fitting
def least_dy2_line(x, y, sigma=None, returnS=False):
    """
    Calculate the best line `y = a * x + b` satisfying least $(\\Delta y)^2$ (vertical offset).

    Parameters
    ----------
    x : array_like
        x values.
    y : array_like
        y values.
    sigma : array_like, optional
        Standard deviation of y values.
    returnS : bool, optional
        If True, return covariance matrix elements.

    Returns
    -------
    a : float
        Slope.
    b : float
        Intercept.
    error : float
        Minimum $(\\Delta y)^2$.
    Sxx : float, optional
        Covariance matrix element (x).
    Syy : float, optional
        Covariance matrix element (y).
    Sxy : float, optional
        Covariance matrix element (x, y).
    """
    # Compute weights
    if sigma is None:
        weights = None
    else:
        weights = np.power(sigma, -2)

    # Define average function with correct weights
    average = lambda a: np.average(a, axis=0, weights=weights)

    # Compute means
    meanx = average(x)
    meany = average(y)

    # Compute deltas
    dx = x - meanx
    dy = y - meany

    # Compute covariance terms
    Sxx = average(dx**2)
    Sxy = average(dx * dy)
    Syy = average(dy**2)

    # Compute slope and intercept
    a = Sxy / Sxx
    b = meany - a * meanx

    # Compute error
    error = Syy - Sxy**2 / Sxx

    if returnS:
        return a, b, error, Sxx, Syy, Sxy
    else:
        return a, b, error


def least_dy2_line_vectorized(x, y, sigma=None, returnS=False):
    """
    Vectorized version of the least_dy2_line function to handle multiple (Y, X) pairs in parallel.

    Parameters
    ----------
    x : array_like
        1D or 2D array of x values. If 2D, each row corresponds to a different (Y, X) pair.
    y : array_like
        2D array of y values, where each row corresponds to a different (Y, X) pair.
    sigma : array_like, optional
        2D array of standard deviations of y values, same shape as y.
    returnS : bool, optional
        If True, return covariance matrix elements.

    Returns
    -------
    a : array_like
        1D array of slopes for each (Y, X) pair.
    b : array_like
        1D array of intercepts for each (Y, X) pair.
    error : array_like
        1D array of minimum $(\\Delta y)^2$ for each (Y, X) pair.
    Sxx : array_like, optional
        1D array of covariance matrix elements (x) for each (Y, X) pair.
    Syy : array_like, optional
        1D array of covariance matrix elements (y) for each (Y, X) pair.
    Sxy : array_like, optional
        1D array of covariance matrix elements (x, y) for each (Y, X) pair.
    """
    # Compute weights
    if sigma is None:
        weights = None
    else:
        weights = np.power(sigma, -2)

    # Get the number of datasets (rows in y)
    m = y.shape[0]

    # Ensure x is 2D and matches the shape of y
    if x.ndim == 1:
        x = np.tile(x, (m, 1))  # Repeat x for each dataset

    # Ensure weights have the correct shape
    if weights is not None:
        if weights.ndim == 1:
            weights = np.tile(weights, (m, 1))
        elif weights.shape != y.shape:
            raise ValueError("Weights and y must have the same shape")

    # Define average function with correct weights
    average = lambda a: np.average(a, axis=1, weights=weights)

    # Compute means
    meanx = average(x)
    meany = average(y)

    # Compute deltas
    dx = x - meanx[:, np.newaxis]
    dy = y - meany[:, np.newaxis]

    # Compute covariance terms
    Sxx = average(dx**2)
    Sxy = average(dx * dy)
    Syy = average(dy**2)

    # Compute slope and intercept
    a = Sxy / Sxx
    b = meany - a * meanx

    # Compute error
    error = Syy - (Sxy**2) / Sxx

    if returnS:
        return a, b, error, Sxx, Syy, Sxy
    else:
        return a, b, error


def least_dist2_line(x, y, sigma=None, returnS=False):
    """
    Calculate the best line `y = a * x + b` satisfying least distance squares (perpendicular offset).

    Parameters
    ----------
    x : array_like
        x values.
    y : array_like
        y values.
    sigma : array_like, optional
        Standard deviation of y values.
    returnS : bool, optional
        If True, return covariance matrix elements.

    Returns
    -------
    a : float
        Slope.
    b : float
        Intercept.
    error : float
        Minimum distance square.
    Sxx : float, optional
        Covariance matrix element (x).
    Syy : float, optional
        Covariance matrix element (y).
    Sxy : float, optional
        Covariance matrix element (x, y).
    """
    # Compute weights
    if sigma is None:
        weights = None
    else:
        weights = np.power(sigma, -2)

    # Define average function with correct weights
    average = lambda a: np.average(a, axis=0, weights=weights)

    # Compute means
    meanx = average(x)
    meany = average(y)

    # Compute deltas
    dx = x - meanx
    dy = y - meany

    # Compute covariance terms
    Sxx = average(dx**2)
    Sxy = average(dx * dy)
    Syy = average(dy**2)

    # Compute slope and intercept
    dS = Syy - Sxx
    B = dS / Sxy / 2

    a = B + np.sqrt(B**2 + 1)
    b = meany - a * meanx

    # Compute error
    error = Sxx + (dS - 2 * Sxy * a) / (a**2 + 1)
    # error = Sxx - Sxy/a  ## big calc error

    if returnS:
        return a, b, error, Sxx, Syy, Sxy
    else:
        return a, b, error


def least_dist2_line_vectorized(x, y, sigma=None, returnS=False):
    """
    Vectorized version of the least_dist2_line function to handle multiple (Y, X) pairs in parallel.

    Parameters
    ----------
    x : array_like
        1D or 2D array of x values. If 2D, each row corresponds to a different (Y, X) pair.
    y : array_like
        2D array of y values, where each row corresponds to a different (Y, X) pair.
    sigma : array_like, optional
        2D array of standard deviations of y values, same shape as y.
    returnS : bool, optional
        If True, return covariance matrix elements.

    Returns
    -------
    a : array_like
        1D array of slopes for each (Y, X) pair.
    b : array_like
        1D array of intercepts for each (Y, X) pair.
    error : array_like
        1D array of minimum distance squares for each (Y, X) pair.
    Sxx : array_like, optional
        1D array of covariance matrix elements (x) for each (Y, X) pair.
    Syy : array_like, optional
        1D array of covariance matrix elements (y) for each (Y, X) pair.
    Sxy : array_like, optional
        1D array of covariance matrix elements (x, y) for each (Y, X) pair.
    """
    # Compute weights
    if sigma is None:
        weights = None
    else:
        weights = np.power(sigma, -2)

    # Get the number of datasets (rows in y)
    m = y.shape[0]

    # Ensure x is 2D and matches the shape of y
    if x.ndim == 1:
        x = np.tile(x, (m, 1))  # Repeat x for each dataset

    # Ensure weights have the correct shape
    if weights is not None:
        if weights.ndim == 1:
            weights = np.tile(weights, (m, 1))
        elif weights.shape != y.shape:
            raise ValueError("Weights and y must have the same shape")

    # Define average function with correct weights
    average = lambda a: np.average(a, axis=1, weights=weights)

    # Compute means
    meanx = average(x)
    meany = average(y)

    # Compute deltas
    dx = x - meanx[:, np.newaxis]
    dy = y - meany[:, np.newaxis]

    # Compute covariance terms
    Sxx = average(dx**2)
    Sxy = average(dx * dy)
    Syy = average(dy**2)

    # Compute slope and intercept
    dS = Syy - Sxx
    B = dS / Sxy / 2
    a = B + np.sqrt(B**2 + 1)
    b = meany - a * meanx

    # Compute error
    error = Sxx + (dS - 2 * Sxy * a) / (a**2 + 1)
    # error = Sxx - Sxy/a  ## big calc error

    if returnS:
        return a, b, error, Sxx, Syy, Sxy
    else:
        return a, b, error


# ANCHOR - Intensity profile analysis
def radial_intensity(img, center, r_max=None, bin_width=10, statistic="mean"):
    """
    Calculate the radial intensity distribution of an image centered at a specified point.

    Parameters
    ----------
    img : numpy.ndarray
        The 2D image to analyze.
    center : tuple of float
        The (cy, cx) coordinates of the center of the image.
    r_max : float, optional
        The maximum radius to consider. Default is None, which considers the entire image.
    bin_width : float, optional
        The width of the radial bins in pixels. Default is 10.
    statistic : str, optional
        The statistic to compute in each bin. Default is "mean".

    Returns
    -------
    numpy.ndarray
        The radial bins.
    numpy.ndarray
        The mean intensity in each radial bin.
    """
    cy, cx = center

    # Calculate the distance from each pixel to the center point
    x, y = np.meshgrid(np.arange(img.shape[1]) - cx, np.arange(img.shape[0]) - cy)
    r = np.sqrt(x**2 + y**2)

    # Create radial bins of 10 pixels width
    if r_max is None:
        r_max = r.max()

    num_bins = int(np.ceil(r_max / bin_width))
    bins = np.linspace(0, num_bins * bin_width, num_bins + 1)

    # Compute the mean intensity in each bin

    radial_means, _, _ = binned_statistic(
        r[r < r_max].flatten(),
        img[r < r_max].flatten(),
        bins=bins,
        statistic=statistic,
    )
    return bins[:-1], radial_means


def value_on_line(img, theta, cx, cy):
    """
    Computes the value of the given image on the line y = tan(theta)*(x-cx) + cy,
    along with the distance from (cx, cy) to each point on the line.

    This function only considers lines with 0 <= theta <= pi/2.

    Parameters
    ----------
    img : numpy.ndarray
        The 2-dimensional image.
    theta : float
        The angle of the line in radians. This value must be in the range [0, pi/2].
    cx : float
        The x-coordinate of the center point.
    cy : float
        The y-coordinate of the center point.

    Returns
    -------
    dists : list of float
        A list of the distances from (cx, cy) to each point on the line.
    values : list of float
        A list of the values of the image on the line.
    """

    assert theta >= 0
    assert theta <= np.pi / 2

    # Get the shape of the image
    rows, cols = img.shape

    # Create an empty list to store the values on the line
    values = []
    dists = []

    if theta == np.pi / 2:
        dists = np.arange(rows) - cy
        values = img[:, cx]
    elif theta < np.pi / 2:
        # Iterate over the columns of the image
        for x in range(cols):
            # Compute the corresponding y value on the line
            y = int(np.tan(theta) * (x - cx) + cy)

            # Check if the y value is within the bounds of the image
            if 0 <= y < rows:
                values.append(img[y, x])
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                dist *= (x > cx) * 2 - 1
                dists.append(dist)
    else:  # theta > np.pi / 2
        # Iterate over the columns of the image
        for y in range(rows):
            # Compute the corresponding y value on the line
            x = int((y - cy) / np.tan(theta) + cx)

            # Check if the y value is within the bounds of the image
            if 0 <= x < cols:
                values.append(img[y, x])
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                dist *= (y > cy) * 2 - 1
                dists.append(dist)
    return dists, values


def intensity_on_circle(
    img, cy, cx, rad, ax=None, angle_point_num=None, show_image=True, **kwargs
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
    **kwargs : dict
        Additional keyword arguments to pass to the plot function.

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
    y = cy - rad * np.sin(theta)

    # Calculate the intensity values along the circle
    intensity = [img[int(round(yy)), int(round(xx))] for xx, yy in zip(x, y)]

    if show_image:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray", **kwargs)
        circle = plt.Circle((cx, cy), rad, color="r", fill=False)
        ax.arrow(cx + rad, cy, 0, -1, head_width=10, head_length=10, fc="r", ec="r")
        ax.add_artist(circle)

    return theta, intensity


def plot_intensity_on_circle(
    img,
    cy,
    cx,
    rad,
    pixel_size,
    ylabel="Intensity",
    ax_img=None,
    ax=None,
    label=None,
    angle_point_num=None,
    **kwargs,
):
    """
    Plot the intensity values on a circle of a given radius centered at a specified point.

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
    pixel_size : float
        The pixel size in micrometers.
    ylabel : str, optional
        The label for the y-axis. Default is "Intensity".
    ax_img : matplotlib.axes.Axes, optional
        The axes to display the image. Default is None. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the intensity values. Default is None. If None, a new figure is created.
    label : str, optional
        The label for the plot. Default is None. If provided, a legend is displayed.
    angle_point_num : int, optional
        The number of points to sample along the circle. Default is None.
        If None, the number of points is calculated based on the circumference.
    **kwargs : dict
        Additional keyword arguments to pass to the plot function.
    """
    theta, intensity = intensity_on_circle(img, cy, cx, rad, ax_img, angle_point_num)

    # Plot the intensity values on the circle
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2))
    ax.plot(theta / np.pi * 4, intensity, label=label)
    # ax.set_xticks(range(9), [str(i * 45) + "\N{DEGREE SIGN}" for i in range(9)])
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{i * 45}\N{DEGREE SIGN}" for i in range(9)])
    ax.grid(True)
    ax.set_xlabel("Angle")
    ax.set_ylabel(ylabel)
    ax.set_title(
        "Center ({}, {}), Radius {} pix = {} um\n".format(
            cx, cy, rad, round(rad * pixel_size, 3)
        )
    )
    if label is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Calculate the circumference of the circle
    circumference = rad * theta * pixel_size

    ax = ax.twiny()
    ax.plot(circumference, intensity, **kwargs)
    ax.set_xlabel("Arc length [um]")


# ANCHOR - Image alignment
def align_points(points1, points2):
    """
    Align two sets of 3 points using rotation, translation, and scaling.
    The points2 are transformed to match the points1.

    Parameters
    ----------
    points1 : list of tuple
        List of 3 (x, y) tuples for the first set of points.
    points2 : list of tuple
        List of 3 (x, y) tuples for the second set of points.

    Returns
    -------
    tuple
        The optimal scale factor, rotation angle (in radians), and translation vector.
    """

    def objective_function(params):
        scale, angle, tx, ty = params
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        transformed_points1 = [scale * np.dot(R, p) + [tx, ty] for p in points1]

        return np.array(transformed_points1).flatten() - np.array(points2).flatten()

    # Initial guess: unit scale, no rotation, no translation
    initial_guess = [1, 0, 0, 0]
    # Perform optimization
    result = least_squares(objective_function, initial_guess)
    # Extract optimal parameters
    scale, angle, tx, ty = result.x
    return scale, angle, tx, ty


def transform_image(image, scale, angle, tx, ty, template_shape):
    """
    Apply scaling, rotation, translation, and slicing to an image.

    Parameters
    ----------
    image : np.array
        The input image to be transformed
    scale : float
        The scaling factor (magnification)
    angle : float
        The rotation angle in radians
    tx : float
        The x-coordinate of the top-left corner of the template in the image
    ty : float
        The y-coordinate of the top-left corner of the template in the image
    template_shape : tuple
        The shape of the template to crop

    Returns
    -------
    transformed_image : np.array
        The transformed image
    """
    # Get the image dimensions
    h, w = image.shape[:2]
    # Create the scaling matrix
    scale_matrix = np.array([[scale, 0], [0, scale]])
    # Create the rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    # Combine the scaling and rotation matrices
    transform_matrix = np.dot(scale_matrix, rotation_matrix)
    # Add the translation vector
    transform_matrix = np.hstack([transform_matrix, [[tx], [ty]]])
    # Apply affine transformation
    transformed_image = cv2.warpAffine(image, transform_matrix, (w, h))[
        : template_shape[0], : template_shape[1]
    ]
    return transformed_image


# ANCHOR - Image processing
def below_nsigma(img, nsigma):
    """
    Returns a 2D boolean mask indicating whether each pixel in the 2D image is below the specified number of standard deviations from the mean intensity.

    Parameters
    ----------
    img : numpy.ndarray
        The 2D image to analyze.
    nsigma : float
        The number of standard deviations below the mean intensity to consider.

    Returns
    -------
    numpy.ndarray
        A boolean mask indicating whether each pixel in the image is below the specified number of standard deviations from the mean intensity
    """
    return img < np.mean(img) - nsigma * np.std(img)


def embed_image_in_constant(
    img, target_height, target_width, return_indices=False, constant_value=0
):
    """
    Embeds an input image into a larger image filled with a constant value.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to embed.
    target_height : int
        The height of the target image.
    target_width : int
        The width of the target image.
    return_indices : bool, optional
        If True, return the indices of the embedded image in the target image. Default is False.
    constant_value : int, optional
        The constant value to fill the target image with. Default is 0.

    Returns
    -------
    Union[numpy.ndarray, Tuple[numpy.ndarray, int, int]]
        A numpy array representing the larger image with the input image embedded.
        If return_indices is True, a tuple containing the embedded image and the indices of the embedded image
        in the target image is returned.
    """
    original_height, original_width = img.shape[:2]

    if target_height < original_height or target_width < original_width:
        raise ValueError(
            "Target dimensions must be larger than the original image dimensions."
        )

    embedded_image = np.full(
        (target_height, target_width), constant_value, dtype=img.dtype
    )
    start_y = (target_height - img.shape[0]) // 2
    start_x = (target_width - img.shape[1]) // 2
    embedded_image[
        start_y : start_y + img.shape[0], start_x : start_x + img.shape[1]
    ] = img
    if return_indices:
        return embedded_image, start_y, start_x
    else:
        return embedded_image


def embed_image_in_mean(img, target_height, target_width, return_indices=False):
    """
    Embeds an input image into a larger image filled with the mean intensity value.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to embed.
    target_height : int
        The height of the target image.
    target_width : int
        The width of the target image.
    return_indices : bool, optional
        If True, return the indices of the embedded image in the target image. Default is False.

    Returns
    -------
    Union[numpy.ndarray, Tuple[numpy.ndarray, int, int]]
        A numpy array representing the larger image with the input image embedded.
        If return_indices is True, a tuple containing the embedded image and the indices of the embedded image
        in the target image is returned.
    """
    return embed_image_in_constant(
        img, target_height, target_width, return_indices, np.mean(img)
    )


def embed_image_in_zeros(img, target_height, target_width, return_indices=False):
    """
    Embeds an input image into a larger image filled with zeros.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to embed.
    target_height : int
        The height of the target image.
    target_width : int
        The width of the target image.
    return_indices : bool, optional
        If True, return the indices of the embedded image in the target image. Default is False.

    Returns
    -------
    Union[numpy.ndarray, Tuple[numpy.ndarray, int, int]]
        A numpy array representing the larger image with the input image embedded.
        If return_indices is True, a tuple containing the embedded image and the indices of the embedded image
        in the target image is returned.
    """
    return embed_image_in_constant(img, target_height, target_width, return_indices, 0)


# ANCHOR - Histogram analysis
def gaussian(x, scaling, mean, std_dev):
    """
    Gaussian function.

    Parameters
    ----------
    x : array_like
        The independent variable.
    scaling : float
        The scaling factor of the Gaussian.
    mean : float
        The mean of the Gaussian.
    std_dev : float
        The standard deviation of the Gaussian.

    Returns
    -------
    array_like
        The value of the Gaussian function at the given points `x`.
    """
    return (
        scaling
        / (std_dev * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    )


def multi_gaussian_list(x, *params):
    """
    List of Gaussian functions with a variable number of parameters.

    Parameters
    ----------
    x : array_like
        The independent variable.
    *params : list
        A list of parameters for each Gaussian component in the form [scaling1, mean1, standard deviation1, scaling2, mean2, standard deviation2, ...].

    Returns
    -------
    array_like
        The value of the multi-Gaussian function at the given points `x`.
    """
    return [gaussian(x, *params[i : i + 3]) for i in range(0, len(params), 3)]


def multi_gaussian(x, *params):
    """
    Multi-Gaussian function with a variable number of components.

    Parameters
    ----------
    x : array_like
        The independent variable.
    *params : list
        A list of parameters for each Gaussian component in the form [scaling1, mean1, standard deviation1, scaling2, mean2, standard deviation2, ...].

    Returns
    -------
    array_like
        The value of the multi-Gaussian function at the given points `x`.
    """
    return np.sum(multi_gaussian_list(x, *params), axis=0)


def fit_multi_gaussian(x, y, num_gaussians=1, initial_guess=None):
    """
    Fit a multi-Gaussian function to the given data using curve_fit.

    Parameters
    ----------
    x : array_like
        The independent variable.
    y : array_like
        The dependent variable.
    num_gaussians : int
        The number of Gaussian components to fit. Default is 1.
    initial_guess : list, optional
        An initial guess for the Gaussian parameters [scaling1, mean1, standard deviation1, scaling2, mean2, standard deviation2, ...]. Default is None.

    Returns
    -------
    list
        The optimal parameters for the multi-Gaussian fit [scaling1, mean1, standard deviation1, scaling2, mean2, standard deviation2, ...].
    list
        The covariance matrix for the optimal parameters.
    """
    if initial_guess is None:
        mean_guess = np.average(x, weights=y)
        std_dev_guess = np.sqrt(np.average((x - mean_guess) ** 2, weights=y))
        scaling_guess = y.max() * np.sqrt(2 * np.pi) * std_dev_guess / num_gaussians
        initial_guess = [scaling_guess, mean_guess, std_dev_guess] * num_gaussians
    params, cov = curve_fit(multi_gaussian, x, y, p0=initial_guess)
    return params, cov


def plot_histogram_with_multi_gaussian_fit(
    data,
    num_gaussians=1,
    initial_guess=None,
    ax=None,
    bins="auto",
    label=None,
    hist_color="gray",
    plot_color="C0",
    **kwargs,
):
    """
    Plot a histogram with a multi-Gaussian fit.

    Parameters
    ----------
    data : array_like
        The data to plot.
    num_gaussians : int
        The number of Gaussian components to fit. Default is 1.
    initial_guess : list, optional
        An initial guess for the Gaussian parameters [scaling1, mean1, standard deviation1, scaling2, mean2, standard deviation2, ...]. Default is None.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the histogram and multi-Gaussian fit. Default is None. If None, a new figure is created.
    bins : int or str, optional
        The number of bins to use for the histogram. Default is "auto".
    label : str, optional
        The label for the plot. Default is None. If provided, a legend is displayed.
    **kwargs : dict
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    list
        The optimal parameters for the multi-Gaussian fit [scaling1, mean1, standard deviation1, scaling2, mean2, standard deviation2, ...].
    """
    if ax is None:
        _, ax = plt.subplots()
    hist_data, bin_edges, _ = ax.hist(
        data, bins=bins, density=True, alpha=0.75, color=hist_color, label=label
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        params, _ = fit_multi_gaussian(
            bin_centers, hist_data, num_gaussians, initial_guess
        )
    except RuntimeError:
        if num_gaussians > 1:
            print(
                "Error: Unable to fit multi-Gaussian function. Instead, fitting a single Gaussian function."
            )
            if initial_guess is not None:
                initial_guess = initial_guess[:3]
            params, _ = fit_multi_gaussian(
                bin_centers, hist_data, initial_guess, num_gaussians=1
            )
        else:
            raise RuntimeError("Unable to fit Gaussian function.")
    fine_bins = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    ax.plot(
        fine_bins,
        multi_gaussian(fine_bins, *params),
        label="Multi-Gaussian fit",
        color=plot_color,
        **kwargs,
    )
    baseline = np.zeros_like(fine_bins)
    for i, gaussian_values in enumerate(multi_gaussian_list(fine_bins, *params)):
        ax.fill_between(
            fine_bins,
            baseline,
            gaussian_values,
            alpha=1 / num_gaussians,
            color=f"C{i+1}",
        )
    if label is not None:
        ax.legend()
    return params
