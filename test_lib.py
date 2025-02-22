# filepath: /c:/Users/fredd/projects/Multiplatform-project/test_lib.py
from sodpy.sod import SodImage

# Test the gaussian blur function on a test image loaded from file.
i = SodImage.load("test.png", channels=3)
j = i.gaussian_blur(5, 1.9)
j.save_png("test_gaussian.png")

# Test the function for creating a constant image.
k = SodImage.constant_image(100, 100, 3, (0.5, 0.5, 0.5))
k.save_png("test_constant.png")