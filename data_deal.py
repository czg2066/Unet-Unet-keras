import os, shutil
import numpy as np

root_pth = r'C:\Users\86184\Desktop\U-net_learn\hk\training'
target_x_pth = r'C:\Users\86184\Desktop\U-net_learn\hk\training\X'
target_y_pth = r'C:\Users\86184\Desktop\U-net_learn\hk\training\Y'
is_copy = False

img_pth = os.listdir(root_pth)
img_pth = [p for p in img_pth if '.png' in p]
ori_img_pth = [p for p in img_pth if 'matte' not in p]
anno_img_pth = [p for p in img_pth if 'matte' in p]

seed_img = np.random.seed(2022)
index = np.random.permutation(len(ori_img_pth))
print(np.array(ori_img_pth)[index])
print(np.array(anno_img_pth)[index])
if is_copy:
    for i in range(len(ori_img_pth)):
        shutil.copy(os.path.join(root_pth, ori_img_pth[i]), os.path.join(target_x_pth, ori_img_pth[i]))
        shutil.copy(os.path.join(root_pth, anno_img_pth[i]), os.path.join(target_y_pth, anno_img_pth[i]))
