# UpDPC

This repository contains Python code for simulating Upgraded polarization-resolved differential phase contrast (UpDPC) microscopy, microscope implementation, and subsequent analysis.

The phase retrieval codes are written to be compatible with [the Waller lab's DPC codes][1].

[1]: https://github.com/Waller-Lab/DPC_withAberrationCorrection/tree/master

## Table of Contents

- [UpDPC](#updpc)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
    - [Option 1: Using miniconda (Recommended)](#option-1-using-miniconda-recommended)
    - [Option 2: Using pip](#option-2-using-pip)
  - [Usage](#usage)
    - [Data Files](#data-files)
    - [Observation with ThorCam Setup](#observation-with-thorcam-setup)
    - [Example Workflow](#example-workflow)
    - [Analysis with Siemens star observation](#analysis-with-siemens-star-observation)
  - [Citation](#citation)
  - [License](#license)

## Repository Structure

The repository has the following structure:

```bash
│  .gitignore
│  LICENSE
│  README.md
│  requirements.txt
│
├─ data (downloaded separately)      # Sample data
│  ├─ calibration_result             # Calibration data of camera
│  ├─ raw                            # Raw images
│  ├─ siemens_star                   # Image of Siemens star pattern with UpDPC setup
│  ├─ source                         # Source images for fitting and analysis
│  └─ WLI                            # White light interferometer data to evaluate the carved glass
├─ notebooks                         # Jupyter notebooks for data analysis and processing
│  └─ Siemens_star_analysis          # Code computing MTF from simulated and experimental Siemens star images
│
└─ updpc                             # Main Python scripts for the UpDPC algorithm
```

## Installation

### Option 1: Using miniconda (Recommended)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/inutsuka-yugo/UpDPC.git
   cd UpDPC
   ```

2. **Create and activate the conda environment**

   This repository provides an `environment.yml` file that includes all required packages from conda-forge, a free and open-source repository:

   ```bash
   conda env create -f environment.yml
   conda activate updpc
   ```

### Option 2: Using pip

1. **Clone the repository:**

   ```bash
   git clone https://github.com/inutsuka-yugo/UpDPC.git
   cd UpDPC
   ```

2. **(Optional) Create a Virtual Environment**
   It is recommended to use a virtual environment to avoid conflicts with other Python packages. For example, you can create one using venv:

   ```bash
   python -m venv updpc_env
   source updpc_env/bin/activate      # On Linux/Mac
   updpc_env\Scripts\activate         # On Windows
   ```

3. **Install dependencies:**
   Ensure that you have Python 3.8 installed. You can install the required dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Files

The `data` folder is not included in this repository due to its large size. You can download it from Google Drive.

1. **Download the `UpDPC_data_yinutsuka.zip` from [Google Drive Link](https://drive.google.com/file/d/1UQF8Zqn5t33CkcqZ9V9yitfGQGt_uPXd/view?usp=sharing).**
2. **Unzip the data:** After downloading, unzip data.zip and place the data folder in the root directory of this repository.

### Observation with ThorCam Setup

As an example, the Python code to perform UpDPC with a Windows PC and a Thorlabs CS505MUP1 or CS505MUP1 polarization camera is included.
To run this, you must set up the SDK for Python as follows:

1. Download the ThorCam software from [ThorCam website](https://www.thorlabs.co.jp/software_pages/ViewSoftwarePage.cfm?Code=ThorCam).
2. Unzip `C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific_Camera_Interfaces.zip`.
3. Follow the `Python README.txt` in the unzipped folder to set up the software.
4. Copy and paste the `SDK\Python Toolkit` folder from the extracted folder into the `updpc` folder of the cloned repository.
5. Rename the pasted `Python Toolkit` folder to the `thorcam` folder.

<!-- 例として、Windows PC と Thorlabs 製の偏光カメラである CS505MUP1 や CS505MUP1 を使って UpDPC を使用する場合の Python コードも付属している。
実行する場合は、以下の手順で Python 用の SDK をセットアップする必要がある。
- ThorCam ソフトウェアを https://www.thorlabs.co.jp/software_pages/ViewSoftwarePage.cfm?Code=ThorCam からダウンロードする
- `C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific_Camera_Interfaces.zip` を解凍する
- 解凍したフォルダ内の `Python README.txt` に従ってセットアップする
- 解凍したフォルダ内の `SDK\Python Toolkit` フォルダをコピーし、この clone した repository の `updpc` フォルダ内にペーストする
- ペーストした `Python Toolkit` を `thorcam` フォルダに rename する -->

### Example Workflow

1. **Source fitting**: Observe the source intensity pattern with the Bertrand lens and analyze this pupil image using `notebooks/source_fitting_choose-lmfit-method.ipynb`.
2. **UpDPC observation**: Use `notebooks/real_time_phase_retrieval_with_acquisition.ipynb` to observe computational phase image in real time. Depending on the camera, `updpc/thorcam_controller.py` may need to be modified.
3. **Phase Retrieval**: Retrieve phase shift from raw images after the observation in `notebooks/phase_retrieval_after_acquisition.ipynb`.

### Analysis with Siemens star observation

In the original paper, we evaluated the spatial resolution and quantification by observing Siemens star patterns engraved on glass. Analysis codes for the minimum data are also included in this repository.

1. **Evaluation of the target depth**: Observe the pattern with white light interferometry and analyze the data with `notebooks/WLI_image_analysis.ipynb`.
2. **Evaluation of spatial resolution**: Observe Siemens star patterns and analyze these images with `notebooks/Siemens_star_analysis/Siemens-star-analysis_manual.ipynb` to calculate the pseudo modulation transfer function. After setting parameters, modify `notebooks/Siemens_star_analysis/siemens_star_analysis.py` and run `Siemens-star-analysis_all.py` to process images under various conditions.
3. **Compare with theory**: Discuss results with the simulation code in `notebooks/Siemens_star_analysis/`.

## Citation

If you use this repository in your research, please cite our paper:

> Yugo Inutsuka, Koki Yamamoto, Masafumi Kuroda, Yasushi Okada, "[Paper Title]," bioRxiv, 2024, DOI: [DOI link]

Alternatively, you can use the citation information provided in `CITATION.txt`.

<!--
For BibTeX users:

```bibtex
@article{
}
``` -->

## License

This project is licensed under the [MIT License](LICENSE).
