# Multiplatform Project - Davide Freddi

Toy project for the Multiplatform PhD course. This project is based on a computer vision library in C, which supports both machine learning and traditional image processing. Here, only the latter is adapted, offering a Python interface for the following features:
- SodImage class, adapting the sod_image struct, which is used by most functions (can be loaded from file or constant)
- Image processing functions, which are turned into methods of the SodImage class (either in-place or not)
    - load/create image as class methods
    - rgb <-> hsv conversions
    - copy and save methods
    - desaturation and grayscale conversion
    - gaussian blur method
    - threshold method
    - canny edge detection method
- Image slicing like in OpenCV (which creates a copy of the image and not a view)

## Why this interface is Idiomatic
- It turns a C library with structs and functions into an object-oriented interface, which is more in line with Python-like libraries.
- Methods can act both in-place and/or return the output (typical of a bunch of Python libs)
- It allows slicing, which is popular in most numerical libraries, such as NumPy and all NumPy-based libraries (Torch, OpenCV, ...).

## Building and testing the library

After installing `cffi`, build the library with

```sh
python3 ./sodpy/build_sodpy.py
```

Then, you can test the library with

```sh
python test_lib.py
```

The script uses sodpy to load the test image and process it into some example results, which are saved in the images folder.