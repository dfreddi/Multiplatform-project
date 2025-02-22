from sodcffi import lib, ffi

class SodImage:
    def __init__(self, cimg):
        self.cimg = cimg

    @classmethod
    def load(cls, path, channels=0):
        cimg = lib.sod_img_load_from_file(path.encode("utf-8"), channels)
        if cimg.data == ffi.NULL:
            raise IOError("Failed to load image from " + path)
        return cls(cimg)

    def save_png(self, path):
        res = lib.sod_img_save_as_png(self.cimg, path.encode("utf-8"))
        if res != 0:
            raise IOError("Failed to save PNG to " + path)

    def copy(self):
        cp = lib.sod_copy_image(self.cimg)
        return SodImage(cp)

    def crop(self, dx, dy, w, h):
        cropped = lib.sod_crop_image(self.cimg, dx, dy, w, h)
        return SodImage(cropped)

    def to_grayscale(self, inplace=False):
        gray = lib.sod_grayscale_image(self.cimg)
        if inplace:
            self.cimg = gray
            return self
        return SodImage(gray)

    def gaussian_blur(self, radius, sigma, inplace=False):
        # Call gaussian blur and get a sod_img.
        blurred = lib.sod_gaussian_blur_image(self.cimg, radius, sigma)
        if blurred.data == ffi.NULL:
            raise IOError("Gaussian blur failed")
        if inplace:
            lib.sod_free_image(self.cimg)  # free the old image to avoid a memory leak
            self.cimg = blurred
            return self
        return SodImage(blurred)

    def rgb_to_hsv(self, inplace=False):
        if inplace:
            lib.sod_img_rgb_to_hsv(self.cimg)
            return self
        new_img = self.copy()
        lib.sod_img_rgb_to_hsv(new_img.cimg)
        return new_img

    def hsv_to_rgb(self, inplace=False):
        if inplace:
            lib.sod_img_hsv_to_rgb(self.cimg)
            return self
        new_img = self.copy()
        lib.sod_img_hsv_to_rgb(new_img.cimg)
        return new_img

    def __getitem__(self, key):
        # key expected as (dx, dy, width, height)
        if not isinstance(key, tuple) or len(key) != 4:
            raise IndexError("Index must be a 4-tuple: (dx, dy, w, h)")
        dx, dy, w, h = key
        return self.crop(dx, dy, w, h)

    def __del__(self):
        if self.cimg is not None:
            lib.sod_free_image(self.cimg)
            self.cimg = None