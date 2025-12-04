from sodcffi import lib, ffi
from itertools import product
import numpy as np
from typing import Tuple, Union

class SodImage:
    def __init__(self, cimg) -> None:
        self.cimg = cimg

    @classmethod
    def constant_image(cls, w: int, h: int, c: int, value: Tuple[float, ...]) -> 'SodImage':
        assert isinstance(w, int), "Width must be an integer"
        assert isinstance(h, int), "Height must be an integer"
        assert isinstance(c, int), "Channels must be an integer"
        assert w > 0 and h > 0 and c > 0, "Dimensions must be positive"
        if len(value) != c:
            raise ValueError("Value must have the same number of channels as the image")
        cimg = lib.sod_make_image(w, h, c)
        for x, y in product(range(w), range(h)):
            for ch in range(c):
                assert isinstance(value[ch], (int, float)), f"Value[{ch}] must be numeric"
                lib.sod_img_set_pixel(cimg, x, y, ch, value[ch])
        return cls(cimg)

    @classmethod
    def load(cls, path: str, channels: int = 0) -> 'SodImage':
        assert isinstance(path, str), "Path must be a string"
        assert isinstance(channels, int), "Channels must be an integer"
        cimg = lib.sod_img_load_from_file(path.encode("utf-8"), channels)
        if cimg.data == ffi.NULL:
            raise IOError("Failed to load image from " + path)
        return cls(cimg)

    @classmethod
    def empty_image(cls, w: int, h: int, c: int) -> 'SodImage':
        """Creates an empty image (with uninitialized pixels)."""
        assert isinstance(w, int), "Width must be an integer"
        assert isinstance(h, int), "Height must be an integer"
        assert isinstance(c, int), "Channels must be an integer"
        assert w > 0 and h > 0 and c > 0, "Dimensions must be positive"
        cimg = lib.sod_make_image(w, h, c)
        return cls(cimg)

    def save_png(self, path: str) -> None:
        assert isinstance(path, str), "Path must be a string"
        res = lib.sod_img_save_as_png(self.cimg, path.encode("utf-8"))
        if res != 0:
            raise IOError("Failed to save PNG to " + path)

    def copy(self) -> 'SodImage':
        cp = lib.sod_copy_image(self.cimg)
        return SodImage(cp)

    def crop(self, dx: int, dy: int, w: int, h: int) -> 'SodImage':
        assert isinstance(dx, int), "dx must be an integer"
        assert isinstance(dy, int), "dy must be an integer"
        assert isinstance(w, int), "Width must be an integer"
        assert isinstance(h, int), "Height must be an integer"
        cropped = lib.sod_crop_image(self.cimg, dx, dy, w, h)
        return SodImage(cropped)

    def to_grayscale(self, inplace: bool = False) -> 'SodImage':
        gray = lib.sod_grayscale_image(self.cimg)
        if inplace:
            self.cimg = gray
            return self
        return SodImage(gray)

    def gaussian_blur(self, radius: int, sigma: float, inplace: bool = False) -> 'SodImage':
        assert isinstance(radius, int), "Radius must be an integer"
        assert isinstance(sigma, (int, float)), "Sigma must be numeric"
        blurred = lib.sod_gaussian_blur_image(self.cimg, radius, sigma)
        if blurred.data == ffi.NULL:
            raise IOError("Gaussian blur failed")
        if inplace:
            lib.sod_free_image(self.cimg)
            self.cimg = blurred
            return self
        return SodImage(blurred)

    def threshold(self, channel: int, thresh: float, inplace: bool = False) -> 'SodImage':
        assert isinstance(channel, int), "Channel must be an integer"
        assert isinstance(thresh, (int, float)), "Threshold must be numeric"
        thresh_img = lib.sod_img_get_layer(self.cimg, channel)
        thresh_img = lib.sod_threshold_image(thresh_img, thresh)
        # thresholded = lib.sod_binarize_image(thresholded, 0)
        if inplace:
            self.cimg = thresh_img
            return self
        return SodImage(thresh_img)

    def canny_edge(self, reduce_noise: int = 1, inplace: bool = False) -> 'SodImage':
        assert isinstance(reduce_noise, int), "reduce_noise must be an integer (0 or 1)"
        assert reduce_noise in (0, 1), "reduce_noise must be 0 or 1"
        canny_image = lib.sod_grayscale_image(self.cimg)
        if canny_image.data == ffi.NULL:
            raise IOError("Grayscale conversion failed")
        canny_image = lib.sod_canny_edge_image(canny_image, reduce_noise)
        if canny_image.data == ffi.NULL:
            raise IOError("Canny edge detection failed")
        if inplace:
            lib.sod_free_image(self.cimg)
            self.cimg = canny_image
            return self
        return SodImage(canny_image)

    def rgb_to_hsv(self, inplace: bool = False) -> 'SodImage':
        if inplace:
            lib.sod_img_rgb_to_hsv(self.cimg)
            return self
        new_img = self.copy()
        lib.sod_img_rgb_to_hsv(new_img.cimg)
        return new_img

    def hsv_to_rgb(self, inplace: bool = False) -> 'SodImage':
        if inplace:
            lib.sod_img_hsv_to_rgb(self.cimg)
            return self
        new_img = self.copy()
        lib.sod_img_hsv_to_rgb(new_img.cimg)
        return new_img

    def desaturate(self, ratio: float, inplace: bool = False) -> 'SodImage':
        """Desaturate the image by a given ratio.
        A ratio of 1.0 will keep the image unchanged, while a ratio of 0.0 will
        convert the image to grayscale.
        """
        assert isinstance(ratio, (int, float)), "Ratio must be numeric"
        assert 0.0 <= ratio <= 1.0, "Ratio must be between 0.0 and 1.0"
        desaturated = self.copy()
        desaturated.rgb_to_hsv(inplace=True)
        for x, y in product(range(self.cimg.w), range(self.cimg.h)):
            orig = lib.sod_img_get_pixel(desaturated.cimg, x, y, 1)
            assert isinstance(ratio * orig, (int, float)), "Pixel value must be numeric"
            lib.sod_img_set_pixel(desaturated.cimg, x, y, 1, ratio * orig)
        desaturated.hsv_to_rgb(inplace=True)
        if inplace:
            self.cimg = desaturated.cimg
            return self
        return desaturated

    def get_array(self) -> np.ndarray:
        """
        Returns a NumPy array view of the image.
        Assumes the image uses float data.
        """
        total = self.cimg.h * self.cimg.w * self.cimg.c
        buf = ffi.buffer(self.cimg.data, total * ffi.sizeof("float"))
        arr = np.frombuffer(buf, dtype=np.float32)
        # Ensure the array is writable.
        arr.flags.writeable = True
        return arr.reshape((self.cimg.h, self.cimg.w, self.cimg.c))

    def __getitem__(self, key: Union[Tuple[slice, slice], Tuple[slice, slice, slice]]) -> 'SodImage':
        """
        Supports numpy-like slicing for spatial cropping.
        For 2D slicing, use a tuple of two slices (rows, cols):
            img[dy:ystop, dx:xstop]
        For 3D slicing, use a tuple of three slices (rows, cols, channels):
            img[dy:ystop, dx:xstop, ch_start:ch_stop]
        """
        if isinstance(key, tuple):
            # 2D slicing: spatial crop only.
            if len(key) == 2:
                row_slice, col_slice = key
                if isinstance(row_slice, slice) and isinstance(col_slice, slice):
                    dy = row_slice.start if row_slice.start is not None else 0
                    ystop = row_slice.stop if row_slice.stop is not None else self.cimg.h
                    h = ystop - dy
                    dx = col_slice.start if col_slice.start is not None else 0
                    xstop = col_slice.stop if col_slice.stop is not None else self.cimg.w
                    w = xstop - dx
                    return self.crop(dx, dy, w, h)
            # 3D slicing: spatial crop with channel sub-selection.
            elif len(key) == 3:
                row_slice, col_slice, ch_slice = key
                if (isinstance(row_slice, slice) and isinstance(col_slice, slice)
                        and isinstance(ch_slice, slice)):
                    dy = row_slice.start if row_slice.start is not None else 0
                    ystop = row_slice.stop if row_slice.stop is not None else self.cimg.h
                    new_h = ystop - dy

                    dx = col_slice.start if col_slice.start is not None else 0
                    xstop = col_slice.stop if col_slice.stop is not None else self.cimg.w
                    new_w = xstop - dx

                    ch_start = ch_slice.start if ch_slice.start is not None else 0
                    ch_stop = ch_slice.stop if ch_slice.stop is not None else self.cimg.c
                    new_c = ch_stop - ch_start

                    # Create an empty image with new dimensions.
                    new_img = SodImage.empty_image(new_w, new_h, new_c)
                    orig = self.get_array()
                    # Iterate over the new image pixels and copy selected channels.
                    for i in range(new_h):
                        for j in range(new_w):
                            for k in range(new_c):
                                value = float(orig[dy + i, dx + j, ch_start + k])
                                assert isinstance(value, (int, float)), "Pixel value must be numeric"
                                lib.sod_img_set_pixel(new_img.cimg, j, i, k, value)
                    return new_img
        raise IndexError("Unsupported indexing. Use numpy-style slicing e.g. "
                         "img[dy:ystop, dx:xstop] or img[dy:ystop, dx:xstop, ch_start:ch_stop]")

    def __del__(self) -> None:
        if self.cimg is not None:
            lib.sod_free_image(self.cimg)
            self.cimg = None