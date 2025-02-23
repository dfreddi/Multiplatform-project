# filepath: /c:/Users/fredd/projects/Multiplatform-project/test_lib.py
from sodpy.sod import SodImage

# Test the gaussian blur function on a test image loaded from file.
i = SodImage.load("./images/test.png", channels=3)
j = i.gaussian_blur(5, 1.9)
j.save_png("./images/test_gaussian.png")
del j

# Test the function for creating a constant image.
k = SodImage.constant_image(100, 100, 3, (0.5, 0.5, 0.5))
k.save_png("./images/test_constant.png")
del k

# Test slicing
l = i[100:200, 100:200]
l.save_png("./images/test_slice.png")
del l

# Test desaturation
m = i.desaturate(0.0)
m.save_png("./images/test_desaturated.png")
del m

# Test thresholding
n = i.threshold(0, 0.8)
n.save_png("./images/test_thresholded.png")
del n

# Test edge detection
o = i.canny_edge()
o.save_png("./images/test_edges.png")
del o