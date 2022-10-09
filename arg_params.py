import argparse

#####Net#####
net = argparse.ArgumentParser(description='搭建网络的参数')
net.add_argument('--WIDTH', type=int, default=128, help='网络输入图片的宽度')
net.add_argument('--HEIGTH', type=int, default=128, help='网络输入图片的高度')
net.add_argument('--CHANCEL', type=int, default=3, help='网络输入图片的通道')
net.add_argument('--Lr', type=float, default=1e-3, help='Adam的学习率')
net.add_argument('--Lr_decay', type=float, default=1e-5, help='Adam的学习率的下降率')
net.add_argument('--If_add', type=int, default=1, help='是(1)否(0)使用Unet++, 否则Unet')

#####Train######
train = argparse.ArgumentParser(description='训练模型的各种参数')
train.add_argument('--Trainx_pth', type=str, default=r'C:\Users\86184\Desktop\U-net_learn\hk\training\X', help='训练模型输入的图像路径')
train.add_argument('--Trainy_pth', type=str, default=r'C:\Users\86184\Desktop\U-net_learn\hk\training\Y', help='训练模型目标的图像路径')
train.add_argument('--Batch_size', type=int, default=16, help='训练模型输入的图像的数量')
train.add_argument('--Epochs', type=int, default=100, help='训练轮数')
train.add_argument('--Val_rate', type=float, default=0.2, help='验证集占比')
train.add_argument('--Model_pth', type=str, default='./s.h5', help='模型保存路径')
train.add_argument('--Best_model_pth', type=str, default='./oxford_segmentation_2.h5', help='模型保存路径')

######Test######
test = argparse.ArgumentParser(description='模型测试的各种参数')
test.add_argument('--Sperete_rate', type=float, default=0.8, help='每个像素点被划分为上一层的概率阈值')
test.add_argument('--Layers_rate', type=float, default=0.2, help='图层叠加时预测层的占比')
test.add_argument('--Model_pth', type=str, default='./oxford_segmentation_2.h5', help='模型路径')
test.add_argument('--Color', type=int, default=1, help='图层叠加时图层颜色(0:蓝, 1:绿, 2:红)')
test.add_argument('--Device', type=str, default=r'./1.mp4', help='输入图像的方式（数字表示本地摄像头设备，路径是视频路径）')
test.add_argument('--Save_pth_name', type=str, default=r'./res1.mp4', help='保存处理结果的路径（带文件名 -1覆盖原文件）')
test.add_argument('--Save_video_fps', type=int, default=60, help='保存视频的帧率')
test.add_argument('--Save_size', type=int, default=-1, help='保存处理结果的尺寸（-1为原视频尺寸）')
test.add_argument('--If_save', type=int, default=1, help='是否保存处理结果（0：不保存 1：保存）')
