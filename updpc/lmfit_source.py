import lmfit
import numpy as np
import matplotlib.pyplot as plt
from updpc.image_utils import ax_imshow, imwrite_f32, blur


def fit_source(
    img_out,
    func,
    initial_params,
    method="leastsq",
    fix_params=[],
    title=None,
    save_path=None,
    verbose=True,
    seed=0,
    **kwargs,
):
    """
    Fit a 2D image with a given function using lmfit.
    Parameters are initialized with the given initial_params and are bounded to be within 50% of the initial value.
    Residuals and the best fit are plotted.

    Parameters
    ----------
    img_out : 2D array
        The image to fit.
    func : callable
        A function that takes X, Y, and the parameters as arguments.
    initial_params : list of floats
        Initial values of the parameters.
    method : str
        The fitting method to use.
    fix_params : list of str
        Parameters to fix.
    title : str
        Title of the plot.
    save_path : str
        Path to save the plot.
    verbose : bool
        Whether to print the fitting report and show the plot.
    seed : int
        Seed for the random number generator.
    kwargs : dict
        Additional arguments to pass to the fitting method.

    Returns
    -------
    result : lmfit.model.ModelResult
        The result of the fitting.
    """
    np.random.seed(seed)

    ysize, xsize = img_out.shape
    X, Y = np.arange(xsize), np.arange(ysize)

    gmodel = lmfit.Model(func, independent_vars=["X", "Y"])
    params = gmodel.make_params()
    for initial_param, param_name in zip(initial_params, params):
        gmodel.set_param_hint(
            param_name,
            value=initial_param,
            min=initial_param * 2 / 3,
            max=initial_param * 3 / 2,
        )
    if params["B"].value != 0:
        gmodel.set_param_hint("B", min=0)
    else:
        gmodel.set_param_hint("B", max=0)
        print("B is fixed to 0.")
    for param_name in fix_params:
        gmodel.set_param_hint(param_name, vary=False)
    result = gmodel.fit(img_out, X=X, Y=Y, method=method, **kwargs)
    img_fit = result.best_fit
    if save_path is not None:
        imwrite_f32(save_path, [img_out, img_fit, img_out - img_fit], axes="CYX")

    if verbose:
        print(result.fit_report())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(title)
        ax = axes[0]
        ax_imshow(ax, img_out)
        ax.set_title("Original")
        ax = axes[1]
        ax_imshow(ax, img_fit)
        ax.set_title(label="Best fit")
        ax = axes[2]
        ax_imshow(ax, img_out - img_fit)
        ax.set_title("Residual")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    return result


def cosine_source(X, Y, rad, rad_in, w_in, f_cos, cx, cy, A, B):
    """
    Create a 2D image of a circle with a cosine function inside and rectangles outside.
    Circle: center coordinates (cx, cy), radii (rad, rad_in), value A * cos(sqrt(x^2 + y^2) / f_cos) + B
    Inner circle: radius rad_in, center (cx, cy), value 0
    Rectangles: width w_in * 2, center (cx, cy), value B

    Parameters
    ----------
    X : np.ndarray
        1-dimensional array of x-coordinates
    Y : np.ndarray
        1-dimensional array of y-coordinates
    rad : float
        Outer radius of the circle
    rad_in : float
        Inner radius of the circle
    w_in : float
        Half-width of the rectangle
    f_cos : float
        Frequency of the cosine function
    cx : float
        x-coordinate of the center of the circle
    cy : float
        y-coordinate of the center of the circle
    A : float
        Value of the maximum of the cosine function
    B : float
        Value of the background
    """
    x_from_cx = X - cx
    y_from_cy = Y - cy
    r_from_cxy = np.sqrt(x_from_cx[np.newaxis, :] ** 2 + y_from_cy[:, np.newaxis] ** 2)

    source = A * np.cos(r_from_cxy / f_cos)
    source[r_from_cxy > rad] = 0

    source[r_from_cxy < rad_in] = 0
    source[:, np.abs(x_from_cx) < w_in] = 0
    source[np.abs(y_from_cy) < w_in, :] = 0

    return source + B


def cosine_source_2center(X, Y, rad, rad_in, w_in, f_cos, cx, cy, A, B, bx, by):
    """
    Create a 2D image of a circle with a cosine function inside and rectangles outside.
    Circle: center coordinates (cx, cy), radii (rad, rad_in), value A * cos(sqrt(x^2 + y^2) / f_cos) + B
    Inner circle: radius rad_in, center (bx, by), value 0
    Rectangles: width w_in * 2, center (bx, by), value B

    Parameters
    ----------
    X : np.ndarray
        1-dimensional array of x-coordinates
    Y : np.ndarray
        1-dimensional array of y-coordinates
    rad : float
        Outer radius of the circle
    rad_in : float
        Inner radius of the circle
    w_in : float
        Half-width of the rectangle
    f_cos : float
        Frequency of the cosine function
    cx : float
        x-coordinate of the center of the circle
    cy : float
        y-coordinate of the center of the circle
    A : float
        Value of the maximum of the cosine function
    B : float
        Value of the background
    bx : float
        x-coordinate of the center of the rectangle
    by : float
        y-coordinate of the center of the rectangle
    """
    x_from_cx = X - cx
    y_from_cy = Y - cy
    r_from_cxy = np.sqrt(x_from_cx[np.newaxis, :] ** 2 + y_from_cy[:, np.newaxis] ** 2)

    source = A * np.cos(r_from_cxy / f_cos)
    source[r_from_cxy > rad] = 0

    x_from_bx = X - bx
    y_from_by = Y - by
    r_from_bxy = np.sqrt(x_from_bx[np.newaxis, :] ** 2 + y_from_by[:, np.newaxis] ** 2)

    source[r_from_bxy < rad_in] = 0
    source[:, np.abs(x_from_bx) < w_in] = 0
    source[np.abs(y_from_by) < w_in, :] = 0

    return source + B


def cosine_source_2center_blur(
    X, Y, rad, rad_in, w_in, f_cos, cx, cy, A, B, bx, by, sigma
):
    """
    Create a 2D image of a circle with a cosine function inside and rectangles outside.
    Circle: center coordinates (cx, cy), radii (rad, rad_in), value A * cos(sqrt(x^2 + y^2) / f_cos) + B
    Inner circle: radius rad_in, center (bx, by), value 0
    Rectangles: width w_in * 2, center (bx, by), value B
    Blur the image with a Gaussian filter with sigma.

    Parameters
    ----------
    X : np.ndarray
        1-dimensional array of x-coordinates
    Y : np.ndarray
        1-dimensional array of y-coordinates
    rad : float
        Outer radius of the circle
    rad_in : float
        Inner radius of the circle
    w_in : float
        Half-width of the rectangle
    f_cos : float
        Frequency of the cosine function
    cx : float
        x-coordinate of the center of the circle
    cy : float
        y-coordinate of the center of the circle
    A : float
        Value of the maximum of the cosine function
    B : float
        Value of the background
    bx : float
        x-coordinate of the center of the rectangle
    by : float
        y-coordinate of the center of the rectangle
    sigma : float
        Standard deviation of the Gaussian filter
    """
    x_from_cx = X - cx
    y_from_cy = Y - cy
    r_from_cxy = np.sqrt(x_from_cx[np.newaxis, :] ** 2 + y_from_cy[:, np.newaxis] ** 2)

    source = A * np.cos(r_from_cxy / f_cos)
    source[r_from_cxy > rad] = 0

    x_from_bx = X - bx
    y_from_by = Y - by
    r_from_bxy = np.sqrt(x_from_bx[np.newaxis, :] ** 2 + y_from_by[:, np.newaxis] ** 2)

    source[r_from_bxy < rad_in] = 0
    source[:, np.abs(x_from_bx) < w_in] = 0
    source[np.abs(y_from_by) < w_in, :] = 0

    return blur(source + B, ksize=15, sigma=sigma)
