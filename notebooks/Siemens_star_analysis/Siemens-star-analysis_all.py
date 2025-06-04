import sys
from os import chdir
from os.path import abspath, dirname

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import siemens_star_analysis

from updpc import *

chdir(join(dirname(dirname(dirname(abspath(__file__)))), "data", "siemens_star"))

# Parameter
root_dirs = ["../../data/siemens_star/"]

not_overwrite_phase = True
# not_overwrite_phase = False
reg_p = 0.1
# reg_p = 0.01
not_overwrite_centered = True
# not_overwrite_centered = False
not_overwrite_MTF = True
not_overwrite_MTF = False

line_breaks(1)

# Main
if __name__ == "__main__":
    for root_dir in root_dirs[::-1]:
        conditions = listdir(root_dir)
        print("conditions:", conditions)
        line_breaks(1)
        for condition in conditions[::1]:
            print("condition:", condition, "start.")
            line_breaks(1)
            try:
                carve_height = int(condition.split("nm")[0]) / 1000
            except:
                print("condition:", condition, "is not a valid directory.")
                line_breaks(5)
                continue
            for img_dir in list_folders(join(root_dir, condition))[::1]:
                try:
                    wavelength = (
                        float(basename(img_dir).split("nm")[0].split("_")[-1]) / 1000
                    )
                except:
                    print("img_dir:", img_dir, "is not a valid directory.")
                    line_breaks(5)
                    continue

                print("img_dir:", img_dir, "start.")
                line_breaks(1)
                if "ZPC" in img_dir:
                    type = "ZPC"
                    na_ill = 0.55
                elif "ODIC" in img_dir:
                    type = "DIC"
                    na_ill = 1.33
                elif "DIC" in img_dir:
                    type = "DIC"
                    na_ill = 0.9
                else:
                    type = "UpDPC"
                    na_ill = 1.33

                # if wavelength != 0.445:
                #     continue
                # if wavelength != 0.660:
                #     continue
                # if type != "UpDPC":
                #     continue
                # if type != "DIC":
                #     continue
                # if na_ill != 1.33:
                #     continue
                # if na_ill != 0.9:
                #     continue
                # if carve_height != 0.025:
                #     continue

                siemens_star_analysis.main(
                    img_dir,
                    nsigma_BG=2,
                    skip_if_exist=True,
                    nsigma_mean_images=2,
                    type=type,
                    wavelength=wavelength,
                    na_ill=na_ill,
                    not_overwrite_phase=not_overwrite_phase,
                    reg_p=reg_p,
                    not_overwrite_centered=not_overwrite_centered,
                    RI_medium=1.33,
                    carve_height=carve_height,
                    not_overwrite_MTF=not_overwrite_MTF,
                )
                print("img_dir:", img_dir, "done.")
                line_breaks(3)
            print("condition:", condition, "done.")
            line_breaks(3)
        print("All done.")
