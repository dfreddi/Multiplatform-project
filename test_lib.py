# filepath: /c:/Users/fredd/projects/Multiplatform-project/test_lib.py
from sodpy.sod import SodImage

i = SodImage.load("test.png", channels=3)
j = i.gaussian_blur(5, 1.9)
j.save_png("test_copy.png")