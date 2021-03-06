
import numpy as np
import os, gzip

def load_data(data_folder):  #加载本地mnist数据集
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
          't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))
        
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
            
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
    
#(train_images, train_labels), (test_images, test_labels) = load_data('/Users/apple/Desktop/dataSet/Mnist_FashionMnist/MNIST/raw/')
(train_images, train_labels), (test_images, test_labels) = load_data('/home/disanda/Desktop/dataSet/MNIST/')
#print("original training data shape:",train_images.shape) # [60000,28,28)]
#print("original testing data shape:",test_images.shape) # [10000,28,28]
train_data=train_images.reshape(60000,784) #变形
test_data=test_images.reshape(10000,784)

#------------------------降维-----------------------
#from sklearn.decomposition import PCA
#reduc = PCA(n_components = 2)
#reduc.fit(train_data) #fit PCA with training data instead of the whole dataset
#train_data = reduc.transform(train_data)
#test_data = reduc.transform(test_data)


# from sklearn.manifold import TSNE
# reduc = TSNE(n_components=3)
# train_data = reduc.fit_transform(train_data)
# test_data = reduc.fit_transform(test_data)

# #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# print("training data shape :",train_data.shape)
# print("testing data shape:",test_data.shape)

#------------------------分类-------------------------
from sklearn.neighbors import KNeighborsClassifier #对降维后的mnist进行KNN分类
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_labels)
y = knn.score(test_data, test_labels) #计算测试得分
print(y)