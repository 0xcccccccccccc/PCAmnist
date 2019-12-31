'''
自己写了十个数字，拿来试试
效果很不理想
'''
import PCAmnist
import numpy as np
import matplotlib.pyplot as plt
import os,sys

if __name__=="__main__":
    root=sys.path[0]
    print(root)
    X = PCAmnist.load_mnist(root)[0]
    Y = PCAmnist.load_mnist(root)[1]
    result = PCAmnist.train(X, Y, 65)
    for i in range(10):
        target = (255 - plt.imread(root+"/myPic/%d.png" % i).mean(axis=2).astype(np.float32).reshape(28 * 28) * 255)

        plt.imshow(target.reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()

        print(PCAmnist.recongnise(result, target))