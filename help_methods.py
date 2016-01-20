import numpy as np
import scipy.linalg as sp
from scipy import ndimage


ALPHA = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
         'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def select_feature(npy, dat):
    """
    Simple method to return a padded list of feature vectors.

    :param npy: The numpy file that contains all the pixel colour values.
    :param dat: The details of the bounding boxes for the indivual letters.
    :return: a list of flattened feature vectors with shape (2000,)
    """

    page = npy
    letter_info = dat
    rows, cols = page.shape
    lis = []

    for letter in letter_info:
        left = int(letter[1])
        bottom = int(letter[2])
        right = int(letter[3])
        top = int(letter[4])
        a = (page[(rows-top):(rows-bottom), left:right])
        a = ndimage.median_filter(a, 3)
        # above below left right
        lis.append(pad_to_size(a).flatten())
    return lis


def pca_item(data, train_data, num):
    """
    Standard PCA method.

    :param data: The actual data that requires PCA'ing
    :param train_data: The training data, this is used for the mean
    :param num: How many dimensions there will be after the pca dimensionality reduction
    :return: The PCA'd data
    """

    # compute data covariance matrix
    covx = np.cov(train_data, rowvar=0)
    # compute first N pca axes
    Norig = covx.shape[0]
    [d, v] = sp.eigh(covx, eigvals=(Norig-num, Norig-1))
    v = np.fliplr(v)
    # compute the mean vector
    data_mean = np.mean(train_data)

    pca_data = np.dot((data - data_mean), v)

    return pca_data


def classify(train, train_labels, test, test_labels, features=None):
    """Nearest neighbour classification
    :param train is the training data matrix
    :param train_labels labels for the training data
    :param test data that is actually going to be classified
    :param test_labels labels for the test data so that a score can be calculated
    :param features list of features that will be a subset of the feature vector values.
    :return score, label. This method returns the percentage score of the classifier and the labels of the classified
    data.
    """

    # Use all feature is no feature parameter has been supplied
    if features is None:
        features = np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]
    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test*test, axis=1))

    modtrain = np.sqrt(np.sum(train*train, axis=1))
    dist = x/np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest = np.argmax(dist, axis=1)
    # mdist = np.max(dist, axis=1)
    label = train_labels[0,nearest]
    score = return_percentage(test_labels[0, :], label)

    return score, label


def pad_to_size(a):
    """
    Very simple padding method.
    :param a: the ndarray which needs padding to (40,50)
    :return: the padded ndarray which is now (40,50) padding was done in black!!
    """

    rows, cols = a.shape
    row_pad = (40-rows)
    col_pad = (50-cols)
    return np.pad(a, ((0, row_pad), (0, col_pad)), mode='constant', constant_values=0)


def char_to_int(b):
    """
    Simple char to int function based on the alphabet at the top of this file.
    :param b: list of characters to be converted to ints
    :return: list of ints that represent characters.
    """
    lis_altered = []

    for i in b:
        lis_altered.append(ALPHA.index(i))

    return lis_altered


def labels_list(lis):
    """
    Creates a list with all the data from the lis of lists passed to it
    :param lis: list of lists
    :return: list of lists that now contains int data changed from chars - using char_to_int
    """

    if len(lis) == 0:
        return []

    lis_of_lis = []
    for i in lis:
        lis_of_lis.append(np.reshape(char_to_int(i), (1, len(i))))
    return lis_of_lis


def labels_to_words(labels):
    """
    Transforms a list of ints to a list of chars based on their values in the alphabet above.
    :param labels: list of ints
    :return: list of chars
    """

    if len(labels)==0:
        return []

    return_lis = []
    for i in labels:
        return_lis.append(ALPHA[i])

    return return_lis


def return_percentage(labels_lis, char_lis):
    """
    Simple return percentage method.
    :param labels_lis: list of the chars that are correct
    :param char_lis: list of the chars from the error correction.
    :return: a percentage score
    """
    leng = float(len(labels_lis))
    score = (100.0 * sum(labels_lis == char_lis))/leng
    return round(score, 2)



