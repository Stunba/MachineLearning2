import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image
import utils
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_data(input_folder):
    # for each folder (holding a different set of letters)
    X = []
    labels = []
    for directory in os.listdir(input_folder):
        # for each image
        for image in os.listdir(input_folder + '/' + directory):
            # open image and load array data
            try:
                file_path = input_folder + '/' + directory + '/' + image
                img = Image.open(file_path)
                img.load()
                img_data = np.asarray(img, dtype=np.int16)
                # add image to dataset
                X.append(img_data.flatten())
                # add label to labels
                labels.append(directory)
            except:
                None # do nothing if couldn't load file
    
#     N = len(X) # number of images
#     img_size = len(X[0]) # width of image
    return np.asarray(X), labels
#     X = np.asarray(X).reshape(N, img_size, img_size,1) # add our single channel for processing purposes
#     labels = to_categorical(list(map(lambda x: ord(x)-ord('A'), labels)), 10) # convert to one-hot


def prepare_csv(input_folder, output):
    X, labels = load_data(input_folder)
    data = pd.DataFrame(X)
    data['labels'] = labels
    data.to_csv(output)
    

def display_images(X):
    num_rows, num_cols = 3, 5
    fig, axes = plt.subplots(num_rows, num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i, j].imshow(X[num_cols*i + j, :].reshape((28, 28)), cmap='gray')
            axes[i, j].axis('off')
    fig.suptitle('Sample Images from Dataset')
    plt.show()
    

def remove_duplicates(X_train, y_train, X_val, y_val):
    data = []
    for train in zip(X_train, y_train):
        contains = False
        for val in zip(X_val, y_val):
            if np.array_equal(train[0], val[0]) and np.array_equal(train[1], val[1]):
                contains = True
                break
        
        if not contains:
            data.append(train)
        
    X = [x[0] for x in data]
    y = [x[1] for x in data]
    return np.array(X), np.array(y)


def train(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Turn up tolerance for faster convergence
    clf = LogisticRegression(
        C=50. / X_train.shape[0], penalty='l1', solver='saga', tol=0.1, multi_class='multinomial'
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # print('Best C % .4f' % clf.C_)
    
    return score


def plot_accuracy(X_train, y_train, X_test, y_test, sizes=[50, 100, 500, 1000, 2500, 5000, 10000]):
    scores = []
    for size in sizes:
        score = train(X_train[size:], y_train[size:], X_test, y_test)
        scores.append(score)
    
    plt.figure(figsize=(16, 8))
    plt.title('Accuracy depending on train size')
    plt.xlabel('Train size')
    plt.ylabel('Accuracy')
    plt.plot(sizes, scores)
    plt.show()
    
    
input_dir = '../input/notmnist/notMNIST_small/notMNIST_small'
# input_dir = '../input/notmnist/notMNIST_large/notMNIST_large'
# prepare_csv(input_dir, 'notmnist_test.csv')