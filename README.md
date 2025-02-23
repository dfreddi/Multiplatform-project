# Multiplatform Project - Davide Freddi

Toy project for the multiplatform phd course. This project is based on a computer vision library in C, which does both machine learning and old school image manipulation. Here only the former is adapted, offering a python interface for the following features:
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
It turns a functional library into an object oriented module, which is more in-line with python-like code.
It allows slicing, which is popular in most numerical libraries, such as numpy and all numpy-based libraries (torch, opencv, ...).
Lets the object be modified both inplace and by return, which is popular in many python objects.