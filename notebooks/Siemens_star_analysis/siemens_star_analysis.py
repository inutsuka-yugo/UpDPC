import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from updpc import *
from tqdm import tqdm

na_in = 0.330
w_in = 0.073
na_cos = 1.435
na_ill = 1.33

mag, na = 1000 / 3, 1.4
pixel_size_cam = 3.45 * 2  # pixel size of camera in micron
pixel_size = pixel_size_cam / mag  # in micron

yellow = (1, 192 / 255, 0)
rad_out = 10  # um


def main(
    img_dir,
    nsigma_BG=2,
    skip_if_exist=True,
    nsigma_mean_images=2,
    type="UpDPC",
    wavelength=None,
    na_ill=None,
    not_overwrite_phase=True,
    reg_p=1e-1,
    not_overwrite_centered=True,
    RI_medium=1.33,
    RI_glass=1.45,
    carve_height=0.05,
    not_overwrite_MTF=True,
    postprocessing=True,
    bigrange=False,
    star_name_list=["_x", "_y", "_z"],
    source_pols=[[90, -45], [45, 0]],
    star_is_low_RI=True,
):

    tif_paths = filter_str_list(
        list_tifs(img_dir), include=star_name_list, include_logic="all"
    )
    print("tif_paths:", len(tif_paths), "images")

    ### ANCHOR Take background
    print("Taking background image")
    bg_paths = filter_str_list(tif_paths, include=["background"], exclude=["AVG"])
    bgdf = pd.DataFrame()
    bgdf["path"] = bg_paths

    # Thorlabs camera files have time in the filename as the second last element.
    # We can use this to distinguish between different fields of view.
    bgdf["time"] = [path.split("_")[-2] for path in bg_paths]

    bg_path = join(img_dir, "background.tif")

    def concatenate_images(imgs):
        """
        Concatenate 2D and 3D images to 3D images.
        2D images are converted to 3D images of a single time frame.
        """
        for i, img in enumerate(imgs):
            if img.ndim == 2:
                imgs[i] = img[np.newaxis]
        return np.concatenate(imgs, axis=0)

    if exists(bg_path) and skip_if_exist:
        print("Background image already exists at", bg_path)
    else:
        bg_paths = filter_str_list(
            list_tifs(img_dir), include=["background"], exclude=["AVG"]
        )
        bgs = []
        for pathtime, sdf in tqdm(bgdf.groupby("time")):
            avg_bg_path = join(
                img_dir, f"AVGwo{nsigma_BG}sigma_background_" + pathtime + ".tif"
            )
            if exists(avg_bg_path):
                bgs.append(imread(avg_bg_path))
            else:
                bgs.append(
                    mean_without_outliers(
                        concatenate_images(
                            [imread(path) for path in tqdm(sdf["path"])]
                        ),
                        nsigma=nsigma_BG,
                    )
                )
                imwrite_f32(avg_bg_path, bgs[-1])
        bg = np.median(bgs, axis=0)
        imwrite_f32(bg_path, bg)
        print("Saved background image to", bg_path)
    line_breaks(2)

    camera_dark_noise = imread(
        "../../data/calibration_result/intercept_intensity_exposure.tif"
    )
    I0 = quad_to_raw(imread(bg_path)) - camera_dark_noise

    assert np.isnan(camera_dark_noise).sum() == 0
    assert np.isnan(I0).sum() == 0

    ### ANCHOR Take mean
    print("Taking mean of images")
    if bigrange:
        pathdf = pd.DataFrame({"path": filter_str_list(tif_paths, include=["range"])})
        if len(pathdf) == 0:
            print("No bigrange images found.")
            return None
        mean_dir = join(img_dir, "Mean_bigrange")
        makedirs(mean_dir)
    else:
        mean_dir = join(img_dir, "Mean")
        makedirs(mean_dir)

        pathdf = pd.DataFrame(
            {
                "path": filter_str_list(
                    tif_paths,
                    exclude="range",
                )
            }
        )

    def param_from_path(path, param):
        return float(basename(path).split(f"_{param}")[-1].split("_")[0])

    for param in ["x", "y", "z"]:
        try:
            pathdf[param] = pathdf["path"].apply(
                lambda path: param_from_path(path, param)
            )
        except:
            # pathdf[param] = np.nan
            pathdf[param] = 0

    print("Number of images to take mean:", len(pathdf))

    for xyz, sdf in tqdm(pathdf.groupby(["x", "y", "z"])):
        paths = sdf["path"].values
        # print("XYZ:", xyz)
        # print("Number of images:", len(paths))
        prefix = basename(commonhead(paths[0], paths[1]))
        out_path = join(mean_dir, f"{prefix}mean{nsigma_mean_images}sigma.tif")
        if exists(out_path):
            continue
        img_mean = mean_without_outliers(
            [imread(path) for path in tqdm(paths)], nsigma_mean_images
        )
        imwrite_f32(out_path, img_mean)

    ## ANCHOR Phase Retrieval (UpDPC) or Intensity Calibration (DIC, ZPC)
    tif_paths = list_tifs(mean_dir)

    if type == "UpDPC":
        ### ANCHOR Save z-stack (t-stack)
        folder_base = f"PH_ill{na_ill}_in{na_in}_rect{w_in}_cos{na_cos}_reg{reg_p}"
        out_dir = join(mean_dir, folder_base)
        makedirs(out_dir, exist_ok=True)

        def save_phase(
            tif_path,
            reg_p=reg_p,
            reg_u=10,
            out_dir=out_dir,
            not_overwrite=True,
        ):
            file_base = basename_noext(tif_path)
            out_path = join(out_dir, file_base + f"_{reg_p}_phase.tif")
            if exists(out_path) and not_overwrite:
                print(out_path, "skip")
                return out_dir
            print(out_path, "Start")
            phases = []
            # absorptions = []
            dpc_raw_array = imread(tif_path)
            if dpc_raw_array.ndim == 2:
                dpc_raw_array = dpc_raw_array[np.newaxis]
            if dpc_raw_array.ndim != 3:
                print(tif_path, f"is {dpc_raw_array.ndim}-dim image.")
                return -1
            dpc_raw_array = (dpc_raw_array - camera_dark_noise) / I0
            dpc_images = np.vstack(
                [raw_to_quadarray(dpc_raw) for dpc_raw in dpc_raw_array]
            )
            solver = UpDPCSolver(
                dpc_images,
                wavelength,
                na,
                na_in,
                pixel_size,
                source_pols=source_pols,
                na_ill=na_ill,
                w_in=w_in,
                na_cos=na_cos,
            )

            solver.setRegularizationParameters(reg_u=reg_u, reg_p=reg_p)
            dpc_result = solver.solve()
            phases = dpc_result.imag
            imwrite(out_path, np.array(phases, dtype=np.float32), imagej=True)
            return out_dir

        for tif_path in tqdm(filter_str_list(tif_paths, exclude="background")[:]):
            try:
                out_dir = save_phase(
                    tif_path, reg_p=reg_p, not_overwrite=not_overwrite_phase
                )
            except Exception as e:
                print(tif_path, e)
                continue
    else:
        folder_base = f"divBG"
        out_dir = join(mean_dir, folder_base)
        makedirs(out_dir, exist_ok=True)

        def save_calib(
            tif_path,
            not_overwrite=True,
            out_dir=out_dir,
        ):
            file_base = basename_noext(tif_path)
            out_path = join(out_dir, file_base + f"_divBG.tif")
            if exists(out_path) and not_overwrite:
                # print(out_path, "skip")
                return out_dir
            print(out_path, "Start")
            dpc_raw_array = imread(tif_path)
            if dpc_raw_array.ndim == 2:
                dpc_raw_array = dpc_raw_array[np.newaxis]
            if dpc_raw_array.ndim != 3:
                print(tif_path, f"is {dpc_raw_array.ndim}-dim image.")
                return -1
            dpc_raw_array = (dpc_raw_array - camera_dark_noise) / I0
            dpc_images = np.vstack(
                [raw_to_intensity(dpc_raw) for dpc_raw in dpc_raw_array]
            )
            imwrite_f32(out_path, dpc_images)
            return out_dir

        for tif_path in tqdm(filter_str_list(tif_paths, exclude="background")[:]):
            out_dir = save_calib(tif_path, not_overwrite=not_overwrite_phase)

    paths_phase = filter_str_list(list_tifs(out_dir), exclude="background")
    line_breaks(2)

    # Siemens-star analysis
    print("Siemens-star analysis")
    ppdf = pd.DataFrame({"path": paths_phase})
    ppdf["key"] = [basename(path).split("_z")[0] for path in ppdf["path"]]

    for param in ["x", "y", "z"]:
        try:
            ppdf[param] = ppdf["path"].apply(
                lambda x: float(basename(x).split(f"_{param}")[-1].split("_")[0])
            )
        except:
            ppdf[param] = 0
    # ppdf.plot("x", "y", style=".--")

    ### ANCHOR Find the focus
    print("Finding the focus")
    centered_dir = join(out_dir, "centered")
    makedirs(centered_dir, exist_ok=True)
    paths_goodZ_path = join(centered_dir, "goodZ.csv")

    if exists(paths_goodZ_path) and not_overwrite_centered:
        paths_goodZ = pd.read_csv(paths_goodZ_path)["path"].values
    else:
        paths_goodZ = []

        for key, sdf in tqdm(ppdf.groupby("key")):
            try:
                phases = np.array([imread(path) for path in tqdm(sdf["path"])])
            except Exception as e:
                print(e)
                print([(imread(path).shape, path) for path in tqdm(sdf["path"])])
                # continue
                raise e
            mean_log_power_spectrum = [
                (
                    np.log(np.abs(F(phase))).mean()
                    if np.isnan(phase).sum() == 0
                    else -np.inf
                )
                for phase in phases
            ]
            if np.max(mean_log_power_spectrum) == -np.inf:
                _, axes = plt.subplots(1, len(phases), figsize=(3 * len(phases), 3))
                for ax, phase in zip(axes, phases):
                    ax.imshow(dilate(np.isnan(phase)), cmap="gray")
                    ax.set_title(np.isnan(phase).sum())
                plt.suptitle(key)
                plt.tight_layout()
                plt.show()
                print("No valid phase images in", key)
                # continue
                raise ValueError("No valid phase images in", key)
            goodZ = np.argmax(mean_log_power_spectrum)
            path_goodZ = sdf["path"].values[goodZ]
            paths_goodZ.append(path_goodZ)
        pd.DataFrame({"path": paths_goodZ}).to_csv(paths_goodZ_path, index=False)

    if not postprocessing:
        return None

    paths_goodZ = filter_str_list(paths_goodZ, include=star_name_list)

    ### ANCHOR Functions to find the center
    half_size = 500
    embed_size = 1250

    if star_is_low_RI:
        bw_correction = 1
    else:
        bw_correction = -1

    if type == "UpDPC":

        def find_center_siemens(
            img,
            sigma_blur=50,
            nsigma=1,
            ksize_dilate=50,
            plot=False,
        ):
            thresh, start_y, start_x = embed_image_in_zeros(
                below_nsigma(blur(bw_correction * img, sigma_blur), nsigma),
                embed_size,
                embed_size,
                return_indices=True,
            )
            cloud = dilate(
                thresh,
                ksize_dilate,
            )
            if plot:
                _, axes = plt.subplots(1, 3, figsize=(15, 5))
                ax_imshow(axes[0], img, cmap="gray", title="Original Image")
                ax_imshow(axes[1], thresh, cmap="gray", title="Thresholded Image")
                ax_imshow(axes[2], cloud, cmap="gray", title="Dilated Image")
                plt.tight_layout()
                plt.show()
            contours, _ = findContours(cloud)
            contourIndex = np.argmax([cv2.contourArea(c) for c in contours])
            cX, cY = center_contour(contours[contourIndex])
            if plot:
                ax = imshow(
                    cv2.drawContours(cloud, contours, contourIndex, 2, 10),
                    figsize=(3, 3),
                )
                ax.scatter(cX, cY, s=10, alpha=0.5)
                plt.show()
            return cX - start_x, cY - start_y

    elif type == "ZPC":

        def find_center_siemens(
            img,
            sigma_blur=25,
            nsigma=2.5,
            ksize_dilate=100,
            plot=False,
        ):
            thresh, start_y, start_x = embed_image_in_zeros(
                below_nsigma(blur(-bw_correction * img, sigma_blur), nsigma),
                embed_size,
                embed_size,
                return_indices=True,
            )
            cloud = dilate(
                thresh,
                ksize_dilate,
            )
            if plot:
                _, axes = plt.subplots(1, 3, figsize=(15, 5))
                ax_imshow(axes[0], img, cmap="gray", title="Original Image")
                ax_imshow(axes[1], thresh, cmap="gray", title="Thresholded Image")
                ax_imshow(axes[2], cloud, cmap="gray", title="Dilated Image")
                plt.tight_layout()
                plt.show()
            contours, _ = findContours(cloud)
            cX, cY = center_contour(contours[0])
            if plot:
                ax = imshow(
                    cv2.drawContours(cloud, contours, 0, 2, 10),
                    figsize=(3, 3),
                    title="Center",
                )
                ax.scatter(cX, cY, s=10, alpha=0.5)
                plt.show()
            return cX - start_x, cY - start_y

    elif type == "DIC":

        def calculate_kernel_size(thresh):
            # Label connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                thresh, connectivity=8
            )
            # Exclude background
            sizes = stats[1:, -1]  # Get sizes of all components except the background
            if len(sizes) == 0:
                return 3  # Default if no foreground components
            kernel_size = int(np.sqrt(max(1, np.median(sizes) - 2 * np.std(sizes))))
            return kernel_size

        def draw_line(img, theta, rho, alpha=0.5):
            h, w = img.shape[:2]
            if np.isclose(np.sin(theta), 0):
                x1, y1 = rho, 0
                x2, y2 = rho, h
            else:
                calc_y = lambda x: rho / np.sin(theta) - x * np.cos(theta) / np.sin(
                    theta
                )
                x1, y1 = 0, calc_y(0)
                x2, y2 = w, calc_y(w)

            # float -> int
            x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
            plt.plot([x1, x2], [y1, y2], "b-", alpha=alpha)
            plt.xlim(0, w)
            plt.ylim(0, h)

        def find_center_siemens(
            img,
            sigma_blur=35,
            nsigma=1.5,
            ksize_dilate=None,
            nsigma_intersection=2,
            plot=False,
        ):
            blurred = blur(bw_correction * img, sigma_blur)
            thresh = np.abs(blurred - np.median(blurred)) > nsigma * blurred.std()
            thresh = thresh.astype(np.uint8)
            if thresh.sum() == 0:
                print("Median:", np.median(blurred))
                print("std:", blurred.std())
                print("No thresholded pixels found. Decrease nsigma.")
                if plot:
                    _, axes = plt.subplots(1, 2, figsize=(10, 5))
                    ax_imshow(axes[0], img, cmap="gray", title="Original Image")
                    ax_imshow(axes[1], blurred, cmap="gray", title="Blurred Image")
                    plt.tight_layout()
                    plt.show()
                return None

            # Remove small artifacts using morphological opening
            kernel_size = calculate_kernel_size(thresh)
            print("kernel size", kernel_size)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Detect edges using the Canny detector
            edges = cv2.Canny(thresh_cleaned, 0.5, 0.5, apertureSize=3)

            if plot:
                _, axes = plt.subplots(1, 3, figsize=(15, 5))
                ax_imshow(axes[0], img, cmap="gray", title="Original Image")
                ax_imshow(
                    axes[1], thresh_cleaned, cmap="gray", title="Thresholded Image"
                )
                ax_imshow(axes[2], edges, cmap="gray", title="Edges")
                plt.tight_layout()
                plt.show()

            # Use the Hough Line Transform to find lines
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
            if lines is None:
                return None
            intersection_points = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    rho1, theta1 = lines[i][0]
                    rho2, theta2 = lines[j][0]
                    # Calculate the intersection point between each pair of lines
                    A = np.array(
                        [
                            [np.cos(theta1), np.sin(theta1)],
                            [np.cos(theta2), np.sin(theta2)],
                        ]
                    )
                    camera_dark_noise = np.array([rho1, rho2])
                    try:
                        intersection_point = np.linalg.solve(A, camera_dark_noise)
                        intersection_points.append(intersection_point)
                    except np.linalg.LinAlgError:
                        # If the lines are parallel, they won't intersect
                        pass

            # Calculate the average intersection point which is the center of the star
            center = mean_without_outliers(
                intersection_points, nsigma=nsigma_intersection
            )

            # Draw each line
            if plot:
                plt.imshow(
                    cv2.cvtColor(
                        auto_contrast(img, return_int=True) // 2, cv2.COLOR_GRAY2RGB
                    )
                )
                plt.title("Detected Lines and Intersections")
                good_alpha_for_line_num = 1 / np.sqrt(len(lines))
                for line in lines:
                    rho, theta = line[0]
                    draw_line(img, theta, rho, alpha=good_alpha_for_line_num)
                plt.plot(center[0], center[1], "yx", markersize=10, label="Center")
                plt.gca().add_patch(
                    plt.Circle(center, int(rad_out / pixel_size), color="y", fill=False)
                )
                plt.legend()
                plt.axis("off")
                plt.show()

            return (int(center[0]), int(center[1]))

    ### ANCHOR Find the center
    phases_centered = []
    paths_centered = []

    print("Number of images to center:", len(paths_goodZ))

    for path in tqdm(paths_goodZ):
        phase = embed_image_in_mean(imread(path), embed_size, embed_size)
        if np.isnan(phase).sum() > 0:
            print("NaNs found in the image.")
            imshow(np.isnan(phase), cmap="jet", title=path)
            plt.show()
            continue

        centering_path = join(img_dir, f"{basename(img_dir)}_centering_params.csv")
        try:
            if exists(centering_path) and not_overwrite_centered:
                sigma_blur, nsigma, ksize_dilate = pd.read_csv(centering_path).values[0]
                cX, cY = find_center_siemens(
                    phase,
                    sigma_blur=int(sigma_blur),
                    nsigma=float(nsigma),
                    ksize_dilate=int(ksize_dilate),
                    plot=False,
                )
            else:
                cX, cY = find_center_siemens(phase, plot=False)
                pd.DataFrame(
                    {
                        "sigma_blur": [sigma_blur],
                        "nsigma": [nsigma],
                        "ksize_dilate": [ksize_dilate],
                    }
                ).to_csv(centering_path, index=False)
            assert cX >= half_size and cY >= half_size
            assert cX <= embed_size - half_size and cY <= embed_size - half_size
        except:
            while True:
                plt.show()
                try:
                    sigma_blur = int(input("sigma_blur: "))
                    nsigma = float(input("nsigma: "))
                    ksize_dilate = int(input("ksize_dilate: "))
                except:
                    if input("Invalid input. Do you want to exit? (y/n)") == "y":
                        raise ValueError("User exit.")
                print(
                    "sigma_blur:",
                    sigma_blur,
                    "nsigma:",
                    nsigma,
                    "ksize_dilate:",
                    ksize_dilate,
                )
                try:
                    cX, cY = find_center_siemens(
                        phase,
                        plot=True,
                        sigma_blur=sigma_blur,
                        nsigma=nsigma,
                        ksize_dilate=ksize_dilate,
                    )
                    assert cX >= half_size and cY >= half_size
                    print("Is the centering correct? (y/n)")
                    if input() == "y":
                        pd.DataFrame(
                            {
                                "sigma_blur": [sigma_blur],
                                "nsigma": [nsigma],
                                "ksize_dilate": [ksize_dilate],
                            }
                        ).to_csv(centering_path, index=False)
                        break
                    else:
                        print("Retry centering.")
                except Exception as e:
                    print(e)
                    pass
        phases_centered.append(
            phase[cY - half_size : cY + half_size, cX - half_size : cX + half_size]
        )
        paths_centered.append(path)
    print("Number of images centered:", len(phases_centered))
    phases_centered = np.array(
        [
            phase
            for phase in phases_centered
            if phase.shape == (2 * half_size, 2 * half_size)
        ]
    )
    print("Number of valid images centered:", len(phases_centered))

    if type == "UpDPC":
        centered_path = join(centered_dir, f"phases_reg{reg_p}_size{half_size*2}.tif")
    else:
        centered_path = join(centered_dir, f"{type}s_size{half_size*2}.tif")

    imwrite_f32(
        centered_path,
        phases_centered,
        metadata={"labels": [basename(path) for path in paths_centered]},
    )
    print("Saved centered images to", centered_path)

    ### ANCHOR MTF analysis
    print("MTF analysis")
    if type == "UpDPC":
        phi0 = 2 * pi * (RI_glass - RI_medium) * carve_height / wavelength
    else:
        phi0 = 1

    def plot_MTF(
        img,
        cy,
        cx,
        pixel_size=pixel_size,
        ax=None,
        label=None,
        angle_point_num=None,
        rad_min=int(0.5 / pixel_size),
        rad_max=int(rad_out / pixel_size),
        **kwargs,
    ):
        rads = range(rad_min, rad_max)
        us = []
        mtfs = []
        for rad in tqdm(rads):
            u = 1 / rad * 40 / (2 * pi) / pixel_size
            us.append(u)
            theta, intensity = intensity_on_circle(
                img, cy, cx, rad, ax, angle_point_num, show_image=False
            )
            us_ = fftfreq(len(theta), (theta[1] - theta[0]) * rad * pixel_size)
            u_match = np.argmin(np.abs(us_ - u))
            mtfs.append(np.abs(F1(intensity)[u_match]) / len(intensity))

        mtfs = np.array(mtfs) / phi0 * pi
        return np.array(us), mtfs

    ### ANCHOR Single image
    center_pred = (half_size, half_size)
    MTF_dir = join(
        dirname(dirname(img_dir)),
        f"MTF_ill{na_ill}_in{na_in}_rect{w_in}_cos{na_cos}",
    )
    makedirs(MTF_dir, exist_ok=True)

    for clean_path, img in zip(paths_centered, phases_centered):
        print(clean_path)
        MTF_image_path = join(MTF_dir, basename(clean_path))
        if exists(MTF_image_path) and not_overwrite_MTF:
            print(MTF_image_path, "skip")
            continue
        imwrite_f32(MTF_image_path, img)
        cv2.imwrite(
            join(MTF_dir, basename(clean_path).replace(".tif", ".png")),
            auto_contrast(img, ignore_percent=0, return_int=True),
        )
        us, mtfs = plot_MTF(img, *center_pred)

        #### ANCHOR Theoritical limit
        xlim = (0.75, 6.25)
        ylim = (0, 1.1)
        fig, ax = plt.subplots(figsize=(10, 2))

        if type == "UpDPC":
            MTFlabel = r"MTF$_\phi$"
            phaselabel = "Phase [rad]"
        else:
            MTFlabel = f"pMTF for {type}"
            phaselabel = "Normalized intensity [a.u.]"
        ax.plot(us, mtfs, alpha=1)

        cutoff = (na_ill + na) / wavelength
        ax.vlines(
            cutoff,
            0,
            ax.get_ylim()[1],
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Theoretical limit",
        )

        ax.set_xlim(*xlim)
        # ax.set_ylim(*ylim)
        ax.set_ylim(ylim[0])
        ax.legend()

        ax.set_xlabel("Spatial frequency [line pairs / um]")
        ax.set_ylabel(MTFlabel)

        ax2 = ax.twiny()
        ax2.set_xlim(*xlim)
        # ax2.set_ylim(*ylim)
        ax2.set_ylim(ylim[0])
        current_ticks = plt.xticks()[0]
        current_ticks = current_ticks[
            (xlim[0] < current_ticks) & (current_ticks < xlim[1])
        ]
        tick_labels = [
            f"{round(500/tick)}" if tick != 0 else "0" for tick in current_ticks
        ]
        # ax2.set_xticks(current_ticks, tick_labels)
        ax2.set_xticks(current_ticks)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel("Corresponding gap distance [nm]")

        # ax.legend(loc="center left")
        try:
            svg_path = f"X{basename_noext(clean_path).split('_X')[1]}.svg"
        except:
            svg_path = f"z{basename_noext(clean_path).split('_z')[1]}.svg"
        plt.savefig(
            join(
                centered_dir,
                "MTF_" + svg_path,
            ),
            bbox_inches="tight",
        )

        pd.DataFrame({"us": us, "mtfs": mtfs}).to_csv(
            join(
                MTF_dir,
                f"{basename_noext(clean_path)}_MTF.csv",
            ),
            index=False,
        )

        for gap_um in np.append(np.arange(0.1, 0.26, 0.01), np.arange(0.3, 0.9, 0.1)):
            rad_pix = int(80 * gap_um / 2 / pi / pixel_size) + 1
            fig, ax = plt.subplots(figsize=(10, 2))
            try:
                plot_intensity_on_circle(
                    img,
                    *center_pred,
                    rad_pix,
                    pixel_size,
                    ylabel=phaselabel,
                    ax=ax,
                    color=yellow,
                )
                ax.set_title(
                    f"Gap distance: {2 * pi * rad_pix * pixel_size / 80 * 1000:.2f} nm"
                )
                fig.savefig(
                    join(
                        centered_dir,
                        f"{gap_um*1000:.0f}nm_{svg_path}",
                    ),
                    bbox_inches="tight",
                )
            except Exception as e:
                print(e)
                pass
            fig.clf()
            plt.close(fig)
        plt.close("all")
