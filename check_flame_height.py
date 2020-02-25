import numpy as np
import cv2
import os
from tqdm import tqdm

base_path = input('Please enter the path of image directory (or use . to use default setting)')
if base_path == '.':
    base_path = '../C001H001S0003_8bit'

files = os.listdir(base_path)
files = files[1:]


iy_center = 0
ix_center = 512
cnt = 0

print('get iy value at ix = ' + str(ix_center))

for i ,file in enumerate(files):
    
    # averate 10 images in each 10 image sequence
    if i/10 > 10:
        break
    if i%10 != 0: 
        continue

    path = os.path.join(base_path, file)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blue = img[:, :, 2]

    iy_center += np.argmax(img_blue[ix_center, :])
    cnt += 1

iy_center /= cnt

print('iy where the maximum luminosity value: ' + str(int(iy_center)))