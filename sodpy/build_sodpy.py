import os
from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("""
    typedef struct sod_img {
        int h;
        int w;
        int c;
        union {
            float* data;
            unsigned char* zdata;
        };
    } sod_img;

    // Image creation and destruction.
    sod_img sod_make_image(int w, int h, int c);
    void sod_free_image(sod_img im);
    sod_img sod_copy_image(sod_img im);

    // Image reading and writing.
    sod_img sod_img_load_from_file(const char *zFile, int nChannels);
    int sod_img_save_as_png(sod_img input, const char *zPath);

    // Image processing.
    void sod_img_set_pixel(sod_img m, int x, int y, int c, float val);
    float sod_img_get_pixel(sod_img m, int x, int y, int c);
    sod_img sod_crop_image(sod_img im, int dx, int dy, int w, int h);
    void sod_img_rgb_to_hsv(sod_img im);
    void sod_img_hsv_to_rgb(sod_img im);
    sod_img sod_grayscale_image(sod_img im);
    sod_img sod_gaussian_blur_image(sod_img im, int radius, double sigma);
    sod_img sod_threshold_image(sod_img input, float thresh);
""")

# Compute the absolute path to the folder containing sod_c headers and C sources.
here = os.path.dirname(__file__)
sod_c_dir = os.path.abspath(os.path.join(here, "..", "sod_c"))

ffibuilder.set_source(
    "sodcffi",
    r'''
    #include "sod.h"
    #include "sod_img_reader.h"
    #include "sod_img_writer.h"
    #include "sod.c"
    ''',
    libraries=[],  # Add any libraries if needed
    include_dirs=[sod_c_dir]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

