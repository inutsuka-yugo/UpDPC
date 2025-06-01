try:
    from .thorcam.examples.windows_setup import configure_path

    configure_path()

    from .thorcam.examples.thorlabs_tsi_sdk.tl_camera import TLCameraSDK
except ModuleNotFoundError as e:
    print(e)
    raise ModuleNotFoundError(
        "***Message from the author*** Please set up the ThorCam SDK for Python referring to the README.md file."
    )
except Exception as e:
    raise e

import datetime
import queue
import threading
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk

from .file_utils import *
from .image_utils import *

THORCAM_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S.%f"


def thorcam_now():
    return datetime.datetime.now().strftime(THORCAM_TIMESTAMP_FORMAT)


pytsi_tag_names = {
    "32768": "SIGNIFICANT_BITS_INT",
    "32769": "EXPOSURE_TIME_US",
    "32779": "FRAME_NUMBER_INT",
    "32782": "TIME_STAMP_RELATIVE_NS",
}


TAG_BITDEPTH = 32768
TAG_EXPOSURE = 32776
TAG_FRAME = 32779
TAG_TIME = 32782
NUMBER_OF_IMAGES = 40


def read_pytsi_metadata(path, frame_col="", ns_col=""):
    """
    Read the path to TIF taken by ThorCam, and return the metadata.

    Returns
    ------
    metadata: pd.DataFrame of metadata common to every pages.
    df: pd.DataFrame of FRAME_NUMBER_INT and TIME_STAMP_RELATIVE_NS_LOW_BITS.
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
    metadata = metadata.rename(columns=pytsi_tag_names)

    df = pd.DataFrame(multi_tags)
    if frame_col == "":
        frame_col = "FRAME_NUMBER_INT"
    if ns_col == "":
        ns_col = "TIME_STAMP_RELATIVE_NS"
    df = df.rename(columns={"32779": frame_col, "32782": ns_col})

    return metadata, df


class ImageAcquisitionThread(threading.Thread):
    def __init__(self, camera, out_dir, filename, **kwargs):
        """
        Parameters
        ----------
        camera : TLCamera
            Camera object.
        out_dir : str
            Output directory.
        filename : str
            Filename.
        """
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = (
            0  # Do not want to block for long periods of time
        )
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        """
        Returns
        -------
        queue.Queue
            Image queue.
        """
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        """
        Parameters
        ----------
        frame : Frame

        Returns
        -------
        Image
        """
        image = frame.image_buffer
        return image

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    image = self._get_image(frame)
                    self._image_queue.put_nowait(image)
            except queue.Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print(
                    "Encountered error: {error}, image acquisition will stop.".format(
                        error=error
                    )
                )
                break
        print("Image acquisition has stopped")


class ImageAcquisitionThread_Save(threading.Thread):
    def __init__(self, camera, out_dir, filename, num_img=np.inf, root=None):
        """
        Parameters
        ----------
        camera : TLCamera
            Camera object.
        out_dir : str
            Output directory.
        filename : str
            Filename.
        num_img : int, optional
            Number of images to take. The default is np.inf.
        root : tk.Tk, optional
            Tkinter root. The default is None.
        """
        super(ImageAcquisitionThread_Save, self).__init__()
        self._camera = camera
        self.out_path = join(out_dir, f"{filename}_{thorcam_now()}")
        self.frames_counted = 0

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = (
            0  # Do not want to block for long periods of time
        )
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self.num_img = num_img
        self.root = root

    def get_output_queue(self):
        """
        Returns
        -------
        queue.Queue
            Image queue.
        """
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        """
        Parameters
        ----------
        frame : Frame

        Returns
        -------
        Image
        """
        image = frame.image_buffer

        # Save images #####################################################################################
        with tifffile.TiffWriter(f"{self.out_path}_{self.frames_counted}.tif") as tiff:
            tiff.write(
                data=image,  # np.ushort image data array from the camera
                extratags=[
                    (
                        TAG_BITDEPTH,
                        "I",
                        1,
                        self._camera.bit_depth,
                        True,
                    ),  # custom TIFF tag for bit depth
                    (TAG_EXPOSURE, "I", 1, self._camera.exposure_time_us, True),
                    (TAG_FRAME, "I", 1, self.frames_counted, False),
                    (TAG_TIME, "Q", 1, frame.time_stamp_relative_ns_or_null, False),
                ],  # custom TIFF tag for exposure
            )

        self.frames_counted += 1
        ####################################################################################################

        return image

    def run(self):
        while (not self._stop_event.is_set()) and (self.frames_counted < self.num_img):
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    image = self._get_image(frame)
                    self._image_queue.put_nowait(image)
            except queue.Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print(
                    "Encountered error: {error}, image acquisition will stop.".format(
                        error=error
                    )
                )
                break
        try:
            self.root.quit()
        except Exception:
            print("Image acquisition has stopped")


def makeCanvas(func):
    """
    func(image): 'raw' image --> displayed image
    """

    class LiveViewCanvas(tk.Canvas):
        def __init__(self, parent, image_queue):
            """
            Parameters
            ----------
            parent : tk.Tk
                Parent.
            image_queue : queue.Queue
                Image queue.
            """
            self.image_queue = image_queue
            self._image_width = 0
            self._image_height = 0
            tk.Canvas.__init__(self, parent)
            self.label_info = tk.Label(parent, text="Brightness")
            self.label_info.pack()
            self.pack()
            self._get_image()

        def _get_image(self):
            try:
                image = self.image_queue.get_nowait()
                self._image = ImageTk.PhotoImage(
                    master=self, image=Image.fromarray(func(image))
                )
                height, width = self._image.height(), self._image.width()
                if (width != self._image_width) or (height != self._image_height):
                    # resize the canvas to match the new image size
                    self._image_width = width
                    self._image_height = height
                    self.config(width=width, height=height)
                self.create_image(0, 0, image=self._image, anchor="nw")
                self.label_info.config(
                    text=f"Mean: {np.mean(image):.2f}, "
                    f"Min: {np.min(image):.2f}, "
                    f"Max: {np.max(image):.2f}"
                )
                self.label_info.update()

            except queue.Empty:
                pass
            self.after(10, self._get_image)

    return LiveViewCanvas


def make_cameraFunc(Thread, Canvas):
    def func(exposure_time_ms=8, out_dir=".", filename="", force=False, num_img=np.inf):
        with TLCameraSDK() as sdk:
            camera_list = sdk.discover_available_cameras()
            with sdk.open_camera(camera_list[0]) as camera:
                camera.exposure_time_us = int(exposure_time_ms * 1000)
                if force:
                    pass
                else:
                    print(
                        "Exposure time is",
                        camera.exposure_time_us / 1000,
                        "ms. Push Enter",
                    )
                    if input() != "":
                        print("Exit")
                        return -1
                root = tk.Tk()
                root.title(camera.name)
                image_acquisition_thread = Thread(
                    camera,
                    out_dir=out_dir,
                    filename=filename,
                    num_img=num_img,
                    root=root,
                )
                _ = Canvas(
                    parent=root, image_queue=image_acquisition_thread.get_output_queue()
                )

                camera.frames_per_trigger_zero_for_unlimited = 0
                camera.arm(2)
                camera.issue_software_trigger()

                image_acquisition_thread.start()

                root.mainloop()

                image_acquisition_thread.stop()
                image_acquisition_thread.join()

                try:
                    root.destroy()
                except Exception:
                    pass

    return func


def make_live_rec_func(func):
    """
    func(image): 'raw' image --> displayed image

    Return
    --------------------------------------------
    live, rec: function
    """

    Canvas = makeCanvas(func)
    live = make_cameraFunc(ImageAcquisitionThread, Canvas)
    rec = make_cameraFunc(ImageAcquisitionThread_Save, Canvas)
    return live, rec
