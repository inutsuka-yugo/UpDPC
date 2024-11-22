import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import fft, fft2, fftfreq, fftshift, ifftshift, ifft, ifft2

from .image_utils import *
from .polarization_image_converter import *

# Based on https://github.com/Waller-Lab/DPC_withAberrationCorrection

pi = np.pi
naxis = np.newaxis
F = fft2
IF = ifft2
F1 = fft
IF1 = ifft


def pupilGen(fxlin, fylin, wavelength, na, na_in=0.0):
    """
    pupilGen creates a circular pupil function in Fourier space.
    Inputs:
            fxlin     : 1D spatial frequency coordinate in horizontal direction
            fylin     : 1D spatial frequency coordinate in vertical direction
            wavelength: wavelength of incident light
            na        : numerical aperture of the imaging system
            na_in     : put a non-zero number smaller than na to generate an annular function
    Output:
            pupil     : pupil function
    """
    pupil = np.array(
        fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 <= (na / wavelength) ** 2
    )
    if na_in != 0.0:
        pupil[
            fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 < (na_in / wavelength) ** 2
        ] = 0.0
    return pupil


def genGrid(size, dx):
    """
    genGrid creates a 1D coordinate vector.
    Inputs:
            size : length of the coordinate vector
            dx   : step size of the 1D coordinate
    Output:
            grid : 1D coordinate vector
    """
    xlin = np.arange(size, dtype="complex64")
    grid = (xlin - size / 2) * dx  # edited
    # grid = (xlin - size // 2) * dx
    return grid


def cosSourceGen(fxlin, fylin, wavelength, na_ill, na_cos=np.inf):
    """
    cosSourceGen creates a source pattern with cosine function in Fourier space.
    Inputs:
        fxlin     : 1D spatial frequency coordinate in horizontal direction
        fylin     : 1D spatial frequency coordinate in vertical direction
        wavelength: wavelength of incident light
        na_ill    : numerical aperture of the illumination system
        na_cos    : numerical aperture of the cosine function length scale
    Output:
        source     : source function
    """
    source = np.array(
        fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 <= (na_ill / wavelength) ** 2
    )
    r2 = np.abs(fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2)
    rad = na_ill / wavelength
    source = np.cos(np.sqrt(r2) / (na_cos / wavelength))
    source[r2 > rad**2] = 0
    if source.min() < 0:
        raise ValueError(
            "Negative value in source pattern. Please check na_ill and na_cos values."
        )
    return source


def polMaskGen(fxlin, fylin, wavelength, na_in, w_in, source_pols):
    """
    polMaskGen creates a polarization angle pattern for a polarization mask,
    consisting of 2x2 polarizers connected by the coordinate axes of the Fourier plane.
    It also creates a mask blocking the boundaries of the polarizers.
    Inputs:
        fxlin     : 1D spatial frequency coordinate in horizontal direction
        fylin     : 1D spatial frequency coordinate in vertical direction
        wavelength: wavelength of incident light
        na_in     : numerical aperture of the center mask blocking the direct beam
        w_in      : half width of rectangular mask blocking the beam between the adjacent polarizers in NA unit
        source_pols: 2x2 int list of polarizer angles in the same order as the mask in degrees
    Output:
        mask_pol   : mask with 4 polarizers connected with a half width rectangular mask
        mask_block : mask blocking the direct beam and the beam between the adjacent polarizers
    """
    mask_pol = np.zeros((len(fylin), len(fxlin)), dtype="int")

    # Slow 42.3 ms ± 2.89 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # mask_pol += (fylin[:, naxis] < 0) * (fxlin[naxis, :] < 0) * source_pols[0][0]
    # mask_pol += (fylin[:, naxis] < 0) * (fxlin[naxis, :] > 0) * source_pols[0][1]
    # mask_pol += (fylin[:, naxis] > 0) * (fxlin[naxis, :] < 0) * source_pols[1][0]
    # mask_pol += (fylin[:, naxis] > 0) * (fxlin[naxis, :] > 0) * source_pols[1][1]

    # Fast 23.9 ms ± 1.49 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    mask_pol[(fylin[:, naxis] < 0) & (fxlin[naxis, :] < 0)] = source_pols[0][0]
    mask_pol[(fylin[:, naxis] < 0) & (fxlin[naxis, :] > 0)] = source_pols[0][1]
    mask_pol[(fylin[:, naxis] > 0) & (fxlin[naxis, :] < 0)] = source_pols[1][0]
    mask_pol[(fylin[:, naxis] > 0) & (fxlin[naxis, :] > 0)] = source_pols[1][1]

    mask_block = np.ones((len(fylin), len(fxlin)), dtype=np.uint8)
    mask_block[:, np.abs(fxlin) < w_in / wavelength] = 0
    mask_block[np.abs(fylin) < w_in / wavelength, :] = 0
    mask_block[
        fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 <= (na_in / wavelength) ** 2
    ] = 0
    return mask_pol, mask_block


class UpDPCSolver:
    """
    UpDPCSolver class provides methods to preprocess or simulate 2D UpDPC measurements
    and solves phase retrieval problems with Tikhonov regularziation.

    Attributes
    ----------
    dpc_imgs : np.ndarray
        The corrected UpDPC images normalized by dividing the background intensity.
    wavelength : float
        The wavelength of the light.
    na : float
        The numerical aperture of the imaging system.
    na_in : float
        The numerical aperture of the center mask blocking the direct beam.
    pixel_size : float
        The pixel size of the UpDPC images.
    source_pols : list of list of int
        The polarizer angles for the source patterns in degrees.
    na_ill : float
        The numerical aperture of the illumination system.
    na_cos : float
        The numerical aperture of the cosine function length scale.
    w_in : float
        The half width of the rectangular mask blocking the beam between the adjacent polarizers in NA unit.
    a0 : float
        The average transmission of the UpDPC images.
    fxlin : np.ndarray
        The 1D spatial frequency coordinate in the horizontal direction.
    fylin : np.ndarray
        The 1D spatial frequency coordinate in the vertical direction.
    dpc_num : int
        The number of UpDPC images in each frame.
    reg_u : float
        The regularization parameter for absorption.
    reg_p : float
        The regularization parameter for phase.
    pupil : np.ndarray
        The pupil function.
    source : np.ndarray
        The source patterns.
    Hu : np.ndarray
        The transfer functions for absorption.
    Hp : np.ndarray
        The transfer functions for phase.

    Methods
    -------
    setRegularizationParameters(reg_u=1e-2, reg_p=1e-2)
        Set regularization parameters.
    normalization()
        Normalize the raw UpDPC measurements by dividing and subtracting out the mean intensity.
    sourceGen()
        Generate UpDPC source patterns.
    WOTFGen()
        Generate transfer functions for each UpDPC source pattern.
    deconvTikhonov(AHA, determinant, fIntensity)
        Solve the UpDPC absorption and phase deconvolution with Tikhonov regularization.
    solve()
        Compute auxiliary functions and output multi-frame absortion and phase results.
    plot_images(figsize=(20, 8), **kwargs)
        Plot the UpDPC images.
    prepare_plot()
        Prepare the plot for the source patterns and the pupil function.
    plot_source(figsize=(20, 8), **kwargs)
        Plot the UpDPC source patterns.
    plot_pupil(figsize=(5, 8), **kwargs)
        Plot the pupil function.
    plot_wotf(fontsize=10, figsize=(20, 8))
        Plot the WOTF functions.
    IfromPhaseAbs(phase, absorption)
        Compute the UpDPC intensity from the phase and absorption.
    """

    def __init__(
        self,
        dpc_imgs,
        wavelength,
        na,
        na_in,
        pixel_size,
        source_pols,
        na_ill=None,
        na_cos=np.inf,
        w_in=0,
        a0=1,
    ):
        """
        Initialize system parameters and functions for UpDPC phase microscopy.


        Parameters
        ----------
        dpc_imgs : np.ndarray
            The UpDPC images. The first dimension is the number of images, and the last two dimensions are the image size.
        wavelength : float
            The wavelength of the light.
        na : float
            The numerical aperture of the imaging system.
        na_in : float, optional
            The numerical aperture of the center mask blocking the direct beam. The default is None.
        pixel_size : float
            The pixel size of the UpDPC images.
        source_pols : list of list of int
            The polarizer angles for the source patterns in degrees.
        na_ill : float, optional
            The numerical aperture of the illumination system. The default is None. If None, it is set to na.
        na_cos : float, optional
            The numerical aperture of the cosine function length scale. The default is np.inf.
        w_in : float, optional
            The half width of the rectangular mask blocking the beam between the adjacent polarizers in NA unit. The default is 0.
        a0 : float, optional
            The average transmission of the UpDPC images. The default is 1.

        Raises
        ------
        ValueError
            If the source pattern has negative values.
        """
        self.dpc_imgs = dpc_imgs.astype("float32")
        self.wavelength = wavelength
        self.na = na
        self.na_in = na_in
        self.pixel_size = pixel_size
        self.source_pols = source_pols
        if na_ill is None:
            self.na_ill = na
        else:
            self.na_ill = na_ill
        self.na_cos = na_cos
        self.w_in = w_in
        self.a0 = a0
        self.fxlin = ifftshift(
            genGrid(dpc_imgs.shape[-1], 1.0 / dpc_imgs.shape[-1] / self.pixel_size)
        )
        self.fylin = ifftshift(
            genGrid(dpc_imgs.shape[-2], 1.0 / dpc_imgs.shape[-2] / self.pixel_size)
        )
        self.dpc_num = len(dpc_imgs)
        self.normalization()
        self.pupil = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.sourceGen()
        self.WOTFGen()

    def setRegularizationParameters(self, reg_u=1e-2, reg_p=1e-2):
        """
        Set regularization parameters.

        Parameters
        ----------
        reg_u : float, optional
            The regularization parameter for absorption. The default is 1e-2.
        reg_p : float, optional
            The regularization parameter for phase. The default is 1e-2.
        """
        # Tikhonov regularization parameters
        self.reg_u = reg_u
        self.reg_p = reg_p

    def normalization(self):
        """
        Normalize the raw UpDPC measurements by dividing and subtracting out the mean intensity.
        """
        for img in self.dpc_imgs:
            # img /= uniform_filter(img, size=img.shape[0] // 2)
            # meanIntensity = img.mean()
            # img /= meanIntensity  # normalize intensity with DC term
            img /= img.mean()  # normalize intensity with DC term
            img -= 1.0  # subtract the DC term
            img *= self.a0

    def sourceGen(self):
        """
        Generate UpDPC source patterns.
        """
        self.source = []
        cos_source = cosSourceGen(
            self.fxlin, self.fylin, self.wavelength, self.na_ill, self.na_cos
        )
        for camera_pol in QUADLIST_POL_DEGREES:
            mask_pol, mask_block = polMaskGen(
                self.fxlin,
                self.fylin,
                self.wavelength,
                self.na_in,
                self.w_in,
                self.source_pols,
            )
            self.source.append(
                cos_source
                * mask_block
                * np.cos((mask_pol - camera_pol) * pi / 180) ** 2
            )
        self.source = np.asarray(self.source)

    def WOTFGen(self):
        """
        Generate transfer functions for each UpDPC source pattern.
        """
        self.Hu = []
        self.Hp = []
        for source_i in self.source:
            FSP_cFP = F(source_i * self.pupil) * F(self.pupil).conj()
            I0 = (source_i * self.pupil * self.pupil.conj()).sum()
            self.Hu.append(2.0 * IF(FSP_cFP.real) / I0)
            self.Hp.append(-2.0 * IF(FSP_cFP.imag) / I0)
            # self.Hp.append(2.0j * IF(1j * FSP_cFP.imag) / I0)
        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def deconvTikhonov(self, AHA, determinant, fIntensity):
        """
        Solve the UpDPC absorption and phase deconvolution with Tikhonov regularization.
        Inputs:
                AHA, determinant: auxiliary functions
                fIntensity      : Fourier spectra of D
                PC intensities
        Output:
                The optimal absorption and phase given the input UpDPC intensities and regularization parameters
        """
        AHy = np.asarray(
            [
                (self.Hu.conj() * fIntensity).sum(axis=0),
                (self.Hp.conj() * fIntensity).sum(axis=0),
            ]
        )
        absorption = IF((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant).real
        phase = IF((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / determinant).real

        return absorption + 1.0j * phase

    def solve(self, return_residual=False, dpc_imgs=None):
        """
        Compute auxiliary functions and output multi-frame absortion and phase results.
        If return_residual is True, return the residual of the UpDPC images.

        Parameters
        ----------
        return_residual : bool, optional
            Return the residual of the UpDPC images. The default is False.
        dpc_imgs : np.ndarray, optional
            If not None, set the UpDPC images. The default is None.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            The multi-frame absorption and phase results. If return_residual is True, return the residual of the UpDPC images.
        """
        if dpc_imgs is not None:
            self.dpc_imgs = dpc_imgs.astype("float64")
            self.normalization()
        dpc_result = []
        AHA = [
            (self.Hu.conj() * self.Hu).sum(axis=0) + self.reg_u,
            (self.Hu.conj() * self.Hp).sum(axis=0),
            (self.Hp.conj() * self.Hu).sum(axis=0),
            (self.Hp.conj() * self.Hp).sum(axis=0) + self.reg_p,
        ]
        determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
        if return_residual:
            residual = 0
            for frame_index in range(self.dpc_imgs.shape[0] // self.dpc_num):
                fIntensity = np.asarray(
                    [
                        F(self.dpc_imgs[frame_index * self.dpc_num + image_index])
                        * np.exp(
                            -2.0j
                            * pi
                            * (
                                self.fxlin[naxis, :]
                                * QUADLIST_DISPLACEMENT_XY[image_index][0]
                                + self.fylin[:, naxis]
                                * QUADLIST_DISPLACEMENT_XY[image_index][1]
                            )
                            * self.pixel_size
                        )
                        for image_index in range(self.dpc_num)
                    ]
                )
                dpc_result.append(self.deconvTikhonov(AHA, determinant, fIntensity))
                residual += np.sum(
                    np.abs(
                        fIntensity
                        - self.Hu * F(dpc_result[-1].real)
                        - self.Hp * F(dpc_result[-1].imag)
                    )
                    ** 2
                )
            return np.asarray(dpc_result), np.sqrt(residual)
        for frame_index in range(self.dpc_imgs.shape[0] // self.dpc_num):
            fIntensity = np.asarray(
                [
                    F(self.dpc_imgs[frame_index * self.dpc_num + image_index])
                    * np.exp(
                        -2.0j
                        * pi
                        * (
                            self.fxlin[naxis, :]
                            * QUADLIST_DISPLACEMENT_XY[image_index][0]
                            + self.fylin[:, naxis]
                            * QUADLIST_DISPLACEMENT_XY[image_index][1]
                        )
                        * self.pixel_size
                    )
                    for image_index in range(self.dpc_num)
                ]
            )
            dpc_result.append(self.deconvTikhonov(AHA, determinant, fIntensity))
        return np.asarray(dpc_result)

    def plot_images(self, figsize=(20, 8), **kwargs):
        """
        Plot the UpDPC images.

        Parameters
        ----------
        figsize : tuple of int, optional
            The size of the figure. The default is (20, 8).
        kwargs : dict
            Additional arguments for ax_imshow.
        """
        _, axes = plt.subplots(
            1, self.dpc_num, sharex=True, sharey=True, figsize=figsize
        )
        if self.dpc_num == 1:
            axes = [axes]
        for ax, image in zip(axes, self.dpc_imgs):
            ax_imshow(ax, image, **kwargs)
        plt.show()

    def prepare_plot(self):
        """
        Prepare the plot for the source patterns and the pupil function.
        """
        self.pixel_y, self.pixel_x = self.dpc_imgs.shape[-2:]
        fx_pix = 1 / self.pixel_x / self.pixel_size
        fy_pix = 1 / self.pixel_y / self.pixel_size
        self.na_lambda_pix_x = int(self.na / self.wavelength / fx_pix) + 1
        self.na_lambda_pix_y = int(self.na / self.wavelength / fy_pix) + 1

    def plot_source(self, figsize=(20, 8), **kwargs):
        """
        Plot the UpDPC source patterns.

        Parameters
        ----------
        figsize : tuple of int, optional
            The size of the figure. The default is (20, 8).
        kwargs : dict
            Additional arguments for ax_imshow.
        """
        self.prepare_plot()
        _, axes = plt.subplots(
            1, self.dpc_num, sharex=True, sharey=True, figsize=figsize
        )
        if self.dpc_num == 1:
            axes = [axes]
        for ax, source in zip(axes, self.source):
            ax_imshow(
                ax,
                fftshift(source)[
                    self.pixel_y // 2
                    - self.na_lambda_pix_y : self.pixel_y // 2
                    + self.na_lambda_pix_y,
                    self.pixel_x // 2
                    - self.na_lambda_pix_x : self.pixel_x // 2
                    + self.na_lambda_pix_x,
                ],
                **kwargs,
            )
            ax.axis("off")
            ax.set_aspect(self.na_lambda_pix_x / self.na_lambda_pix_y)
        plt.show()

    def plot_pupil(self, figsize=(5, 8), **kwargs):
        """
        Plot the pupil function.

        Parameters
        ----------
        figsize : tuple of int, optional
            The size of the figure. The default is (5, 8).
        kwargs : dict
            Additional arguments for ax_imshow.
        """
        self.prepare_plot()
        if np.iscomplexobj(self.pupil):
            _, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
            # Real
            ax = axes[0]
            ax_imshow(
                ax,
                fftshift(self.pupil.real)[
                    self.pixel_y // 2
                    - self.na_lambda_pix_y : self.pixel_y // 2
                    + self.na_lambda_pix_y,
                    self.pixel_x // 2
                    - self.na_lambda_pix_x : self.pixel_x // 2
                    + self.na_lambda_pix_x,
                ],
                **kwargs,
            )
            ax.axis("off")
            ax.set_aspect(self.na_lambda_pix_x / self.na_lambda_pix_y)
            ax.set_title("Real")
            # Imaginary
            ax = axes[1]
            ax_imshow(
                ax,
                fftshift(self.pupil.imag)[
                    self.pixel_y // 2
                    - self.na_lambda_pix_y : self.pixel_y // 2
                    + self.na_lambda_pix_y,
                    self.pixel_x // 2
                    - self.na_lambda_pix_x : self.pixel_x // 2
                    + self.na_lambda_pix_x,
                ],
                **kwargs,
            )
            ax.axis("off")
            ax.set_aspect(self.na_lambda_pix_x / self.na_lambda_pix_y)
            ax.set_title("Imaginary")
            plt.show()
        else:
            _, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
            ax_imshow(
                ax,
                fftshift(self.pupil)[
                    self.pixel_y // 2
                    - self.na_lambda_pix_y : self.pixel_y // 2
                    + self.na_lambda_pix_y,
                    self.pixel_x // 2
                    - self.na_lambda_pix_x : self.pixel_x // 2
                    + self.na_lambda_pix_x,
                ],
                **kwargs,
            )
            ax.axis("off")
            ax.set_aspect(self.na_lambda_pix_x / self.na_lambda_pix_y)
            plt.show()

    def plot_wotf(self, fontsize=10, figsize=(20, 8)):
        """
        Plot the WOTF functions.

        Parameters
        ----------
        fontsize : int, optional
            The font size of the title. The default is 10.
        figsize : tuple of int, optional
            The size of the figure. The default is (20, 8).
        """
        self.prepare_plot()
        self.na_lambda_pix_x *= 2
        self.na_lambda_pix_y *= 2
        font(fontsize)

        max_na_x = max(self.fxlin.real * self.wavelength / self.na)
        min_na_x = min(self.fxlin.real * self.wavelength / self.na)
        max_na_y = max(self.fylin.real * self.wavelength / self.na)
        min_na_y = min(self.fylin.real * self.wavelength / self.na)

        # plot the transfer functions
        _, ax = plt.subplots(2, self.dpc_num, sharex=True, sharey=True, figsize=figsize)
        if self.dpc_num == 1:
            ax = ax[:, naxis]
        for plot_index in range(ax.size):
            plot_row = plot_index // self.dpc_num
            plot_col = np.mod(plot_index, self.dpc_num)
            divider = make_axes_locatable(ax[plot_row, plot_col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if plot_row == 0:
                plot = ax[plot_row, plot_col].imshow(
                    fftshift(self.Hu[plot_col].real)[
                        self.pixel_y // 2
                        - self.na_lambda_pix_y : self.pixel_y // 2
                        + self.na_lambda_pix_y,
                        self.pixel_x // 2
                        - self.na_lambda_pix_x : self.pixel_x // 2
                        + self.na_lambda_pix_x,
                    ],
                    cmap="jet",
                    extent=[min_na_x, max_na_x, min_na_y, max_na_y],
                    #  clim=[-2., 2.]
                )
                ax[plot_row, plot_col].set_title(
                    r"Re[$H_{\mathrm{A}}$]" + f" for Source {plot_col+1}"
                )
                plt.colorbar(
                    plot,
                    cax=cax,
                    #  ticks=[-2., 0, 2.]
                )
            else:
                plot = ax[plot_row, plot_col].imshow(
                    fftshift(self.Hp[plot_col].imag)[
                        self.pixel_y // 2
                        - self.na_lambda_pix_y : self.pixel_y // 2
                        + self.na_lambda_pix_y,
                        self.pixel_x // 2
                        - self.na_lambda_pix_x : self.pixel_x // 2
                        + self.na_lambda_pix_x,
                    ],
                    cmap="jet",
                    extent=[min_na_x, max_na_x, min_na_y, max_na_y],
                    #  clim=[-.8, .8]
                )
                ax[plot_row, plot_col].set_title(
                    r"Im[$H_{\mathrm{P}}$]" + f" for Source {plot_col+1}"
                )
                plt.colorbar(
                    plot,
                    cax=cax,
                    #  ticks=[-.8, 0, .8]
                )
            # ax[plot_row, plot_col].set_xlim(-2.2, 2.2)
            # ax[plot_row, plot_col].set_ylim(-2.2, 2.2)
            ax[plot_row, plot_col].axis("off")
            ax[plot_row, plot_col].set_aspect(1)
        plt.show()

    def I_from_phase_abs_wogrid(self, phase, absorption):
        """
        Compute the UpDPC image intensity without 2x2 grid effect from the phase and absorption.

        Parameters
        ----------
        phase : np.ndarray
            The phase.
        absorption : np.ndarray
            The absorption.

        Returns
        -------
        np.ndarray
            The UpDPC image intensity.
        """
        Is = (
            self.a0**2
            * np.array(
                [
                    Hu_i[0, 0] / 2 + (IF(Hu_i * F(absorption) + Hp_i * F(phase)))
                    for Hu_i, Hp_i in zip(self.Hu, self.Hp)
                ]
            ).real
        )
        return Is

    def I_from_phase_abs(self, phase, absorption):
        """
        Compute the UpDPC image intensity from the phase and absorption.

        Parameters
        ----------
        phase : np.ndarray
            The phase.
        absorption : np.ndarray
            The absorption.

        Returns
        -------
        np.ndarray
            The UpDPC image intensity.
        """
        Is = self.I_from_phase_abs_wogrid(phase, absorption)
        return np.array(
            [
                Is[i, QUADLIST_INDEX_Y[i] :: 2, QUADLIST_INDEX_X[i] :: 2]
                for i in range(4)
            ]
        )


class UpDPCSolver_ipd(UpDPCSolver):
    def solve(self):
        """
        Compute auxiliary functions and output multi-frame absortion and phase results.
        """
        dpc_result = []
        AHA = [
            (self.Hu.conj() * self.Hu).sum(axis=0) + self.reg_u,
            (self.Hu.conj() * self.Hp).sum(axis=0),
            (self.Hp.conj() * self.Hu).sum(axis=0),
            (self.Hp.conj() * self.Hp).sum(axis=0) + self.reg_p,
        ]
        determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
        for frame_index in range(self.dpc_imgs.shape[0] // self.dpc_num):
            fIntensity = np.asarray(
                [
                    F(self.dpc_imgs[frame_index * self.dpc_num + image_index])
                    # * np.exp(
                    #     -2.0j
                    #     * pi
                    #     * (
                    #         self.fxlin[naxis, :]
                    #         * QUADLIST_DISPLACEMENT_XY[image_index][0]
                    #         + self.fylin[:, naxis]
                    #         * QUADLIST_DISPLACEMENT_XY[image_index][1]
                    #     )
                    #     * self.pixel_size
                    # )
                    for image_index in range(self.dpc_num)
                ]
            )
            dpc_result.append(self.deconvTikhonov(AHA, determinant, fIntensity))
        return np.asarray(dpc_result)
