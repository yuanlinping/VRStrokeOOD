import matplotlib.pyplot as plt 
import numpy as np
import os

def delete_folder(folder_path):
    # 删除文件夹及其内容
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)
    os.rmdir(folder_path)
data_path='./evaluation/shape/shape3/results'
folder_path = "img"
if os.path.exists(folder_path):
    delete_folder(folder_path)
os.makedirs(folder_path)
csvs = os.listdir(data_path)
cnt = 0
for csv in csvs:
    if cnt > 1:
        break
    csv_path = os.path.join(data_path,csv)
    print(csv_path)
    data = np.loadtxt(csv_path,delimiter=' ')
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    out_z = data[:,3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o',label='groudtruth')
    ax.scatter(x, y, out_z, c='r', marker='o',label='prediction')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    # plt.show()
    plt.savefig(f"{folder_path}/{csv}.jpg")