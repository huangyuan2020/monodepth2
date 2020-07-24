# -*- coding: utf-8 -*- 
import os
import random

root_path = "/mnt/beegfs/data/interiornet_dataset/"
train_names = []
cam_path = root_path + "trajectory_0"
for cam in os.listdir(cam_path):
	if cam.startswith('t'):
		continue
		print(cam)
	imgs = os.listdir(cam_path + '/' + cam + '/rgb/')
	for i in range(1, len(imgs) - 1):
		train_names.append("trajectory_0/" + cam + " " + imgs[i].replace(".png", "") +'\n')
val_names = []
cam_path = root_path + "trajectory_1"
for cam in os.listdir(cam_path):
	print(cam)
	if cam.startswith('t'):
		continue
	imgs = os.listdir(cam_path + '/' + cam + '/rgb/')
	for i in range(1, len(imgs) - 1):
		val_names.append("trajectory_1/" + cam + " " + imgs[i].replace(".png", "") + '\n')
val_names = random.sample(val_names, 500)
print('train: ', len(train_names))
print('val: ', len(val_names))
train_file = open("./interior/train_files.txt", 'a')
train_file.writelines(train_names)
train_file.close()
val_file = open("./interior/val_files.txt", 'a')
val_file.writelines(val_names)
val_file.close()
