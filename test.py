from PIL import Image
import numpy as np

img_Gray = Image.open('test.png')
img_RGB  = Image.open('test.png').convert('RGB')

img_Gray_array = np.array(img_Gray)
img_RGB_array  = np.array(img_RGB)
print('o')