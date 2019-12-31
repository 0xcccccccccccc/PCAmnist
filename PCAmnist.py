
#  Copyright (c) 2019 by MaHaoran. All Rights Reserved.

import os
import struct
import numpy as np
import matplotlib.pyplot as plt


TRAIN_SAMLPES =50000


class PCAResult: #用于保存PCA变换的类
    def __init__(self, eig_vec, middle, average):  # 降维向量(784,n)，中心化均值(Vector784)，平均化数字图像(784*10)
        self.middle = middle
        self.eig_vec = eig_vec
        self.average = average
        self.source = np.dot(average.T - middle, eig_vec)  # 用于对比的降维后数字特征

    def getTargetDeminsion(self): #获得这个变换的目标维度
        return self.eig_vec.shape[1]

    def project(self, mat): #让mat经过这个变换，得到对应维度的新向量
        target = mat - self.middle
        projected = np.dot(target.T, self.eig_vec)
        return projected



def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def drawNumPic(n, data):
    fig, ax = plt.subplots(
        nrows=n // 5 + 1,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(n):
        ax[i].imshow(data[:, i].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()



def train(X:np.ndarray, Y:np.ndarray,targetDim:int):
    mnist_data = [X[Y == i] for i in range(0, 10)]
    # print(len(mnist_data))
    average_data = np.asmatrix([mnist_data[i].mean(axis=0) for i in range(10)]).T
    # drawNumPic(10,average_data)
    m = X.mean(axis=0).T #
    A = X - m #中心化的样本向量
    covarience_matrix = np.dot(A.T, A)  #协方差矩阵
    eig_val, eig_vec = np.linalg.eig(covarience_matrix)
    index = np.argsort(eig_val)[::-1][:targetDim]
    vec = eig_vec[:, index]
    print("主成分贡献率：",np.sum(abs(eig_val[index]))/np.sum(abs(eig_val)))
    # P=np.dot(A,vec)
    # return P,m,A
    #return vec, m, A, average_data
    return PCAResult(eig_vec=vec,middle=m,average=average_data) #训练结果，包含10个平均化的数字图片average_data、用于降维的特征向量、和1个用于中心化的平均数字图像


def recongnise(pca_result:PCAResult, data:np.ndarray):

    Y = pca_result.source #10个降维后的特征数字
    projected = pca_result.project(data) #降维后的待识别数字
    distance = [np.linalg.norm(projected - Y[i]) for i in range(10)] #这个降维后的数字和这10个降维后数字的欧氏距离
    minDis = min(distance)
    minIdx = distance.index(minDis)
    return minIdx


if __name__ == "__main__":
    X = load_mnist("./")[0]
    Y = load_mnist("./")[1]

    X_train = X[:TRAIN_SAMLPES]
    Y_train = Y[:TRAIN_SAMLPES]

    X_test = X[TRAIN_SAMLPES:]
    Y_test = Y[TRAIN_SAMLPES:]
    tdim=60
    while tdim > 0:
        train_result = train(X_train, Y_train,targetDim=tdim)
        right = 0
        sum = 0
        for i in range(10):
            for luckydog in X_test[Y_test == i]:
                if recongnise(train_result, luckydog) == i:
                    right += 1
                sum += 1
        accuracy = right / sum
        print("保留维度：",tdim,"正确率：" , accuracy,"\n")
        tdim+=1
