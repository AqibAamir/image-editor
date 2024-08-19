This code is a Python script for processing images using the PIL library. It includes functions to apply various filters (like blur, contour, etc.), enhance the image by adjusting color, contrast, and brightness, and invert the image colors. The script also provides additional functionality to create and manipulate random images, apply noise, sepia and grayscale filters, generate patterns, and more. The main function allows users to select an effect, apply it to an image, and save the processed image with a specific filename.

![image](https://github.com/user-attachments/assets/6260553e-92bf-4361-86c4-0cc373509816)





this is the original photo:
![image](https://github.com/user-attachments/assets/28cf3684-e4fc-4800-a0de-cb765f3c1934)





After choosing the inverted option, this is how the image looks now:
![image](https://github.com/user-attachments/assets/26b5ba22-fdfd-4ddf-a7ae-e70ac84e3b9b)

to use this program, paste the code into a python ide like visual studio code, and use pip3 to install the following modules:
-from PIL import Image, ImageFilter, ImageOps, ImageEnhance
-import numpy as np
-import random
-import matplotlib.pyplot as plt

Trarget images need to be saved with the name (image.jpg)
