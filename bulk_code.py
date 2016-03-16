import numpy as np
import time
# These are my .py files
import string_manipulation as sm
import help_methods as hm



def main():
    """
    Main method for this class, performs the for loop which makes all the classifications and error corrections.
    """
    with open('wordsEn.txt', 'r') as f:
        all_words = f.read().replace('\r\n', ' ')
    all_words = all_words.split()  # so that its a list of the words
    for j in xrange(2):
        if j == 0:
            print 'THESE ARE 40-d RESULTS'
        else:
            print 'THESE ARE 10-d RESULTS'
        for i in xrange(10):

            # TIME SO IT CAN BE PRINTED AT THE END OF EACH LOOP.
            start = time.time()

            # READING THE DATA
            file_strings_test = ["test1", "test1.1", "test1.2", "test1.3", "test1.4", "test2", "test2.1", "test2.2",
                                 "test2.3", "test2.4"]
            test_data = np.load(file_strings_test[i]+".npy")
            train_1 = np.load("train1.npy")
            train_2 = np.load("train2.npy")
            train_3 = np.load("train3.npy")
            train_4 = np.load("train4.npy")
            if i >= 5:
                train_5 = np.load("test1.npy")
                train_6 = np.load("test1.1.npy")
                train_7 = np.load("test1.2.npy")
                train_8 = np.load("test1.3.npy")
                train_9 = np.load("test1.3.npy")


            # The list below just contains the strings of file names.
            dat_file_names = ["test1", "test2"]

            # This if else is required so that the .dat that is read in changes.
            if i >= 5:
                t1_dtls = np.loadtxt(dat_file_names[1]+".dat", dtype='str')
                tr5_dtls = np.loadtxt(dat_file_names[0]+".dat", dtype='str')
            else:
                t1_dtls = np.loadtxt(dat_file_names[0]+".dat", dtype='str')
            tr1_dtls = np.loadtxt("train1.dat", dtype='str')
            tr2_dtls = np.loadtxt("train2.dat", dtype='str')
            tr3_dtls = np.loadtxt("train3.dat", dtype='str')
            tr4_dtls = np.loadtxt("train4.dat", dtype='str')

            # this takes the labels for each of these.
            t1_labels_lis = t1_dtls[:, 0]
            tr1_labels_lis = tr1_dtls[:, 0]
            tr2_labels_lis = tr2_dtls[:, 0]
            tr3_labels_lis = tr3_dtls[:, 0]
            tr4_labels_lis = tr4_dtls[:, 0]
            if i>=5:
                tr5_labels_lis = tr5_dtls[:, 0]

            # Create a list of all the labels.
            if i>=5:
                labels_lis = hm.labels_list([t1_labels_lis, tr1_labels_lis, tr2_labels_lis, tr3_labels_lis, tr4_labels_lis,tr5_labels_lis])
            else:
                labels_lis = hm.labels_list([t1_labels_lis, tr1_labels_lis, tr2_labels_lis, tr3_labels_lis, tr4_labels_lis])

            # FEATURE EXTRACTION - HOWEVER THIS MAKES IT BIGGER...
            test_flis = hm.select_feature(test_data, t1_dtls)
            train1_flis = hm.select_feature(train_1, tr1_dtls)
            train2_flis = hm.select_feature(train_2, tr2_dtls)
            train3_flis = hm.select_feature(train_3, tr3_dtls)
            train4_flis = hm.select_feature(train_4, tr4_dtls)
            if i>=5:
                train5_flis = hm.select_feature(train_5, tr5_dtls)
                train6_flis = hm.select_feature(train_6, tr5_dtls)
                train7_flis = hm.select_feature(train_7, tr5_dtls)
                train8_flis = hm.select_feature(train_8, tr5_dtls)
                train9_flis = hm.select_feature(train_9, tr5_dtls)

            # CREATE MATRIX OF ALL TRAIN DATA
            if i>=5:
                all_train = np.vstack((train1_flis, train2_flis, train3_flis, train4_flis, train5_flis, train6_flis, train7_flis, train8_flis, train9_flis))
                # CREATE VECTOR OF ALL TRAIN LABELS
                train_labels_vec = np.append(labels_lis[1], np.append(labels_lis[2], np.append(labels_lis[3], np.append(labels_lis[4], np.append(labels_lis[5],np.append(labels_lis[5],np.append(labels_lis[5],np.append(labels_lis[5],labels_lis[5]))))))))
                all_train_labels = np.reshape(train_labels_vec, (1, train_labels_vec.shape[0]))
            else:
                all_train = np.vstack((train1_flis, train2_flis, train3_flis, train4_flis))
                # CREATE VECTOR OF ALL TRAIN LABELS
                train_labels_vec = np.append(labels_lis[1], np.append(labels_lis[2], np.append(labels_lis[3], labels_lis[4])))
                all_train_labels = np.reshape(train_labels_vec, (1, train_labels_vec.shape[0]))


            # DIMENSIONALITY REDUCTION
            features = 40
            pca_train_data = hm.pca_item(all_train, all_train, features)
            pca_test_data = hm.pca_item(test_flis, all_train, features)

            # CLASSIFICATION
            if j == 0:
                score, letters = hm.classify(pca_train_data, all_train_labels, pca_test_data, labels_lis[0])
            else:
                score, letters = hm.classify(pca_train_data, all_train_labels, pca_test_data, labels_lis[0], xrange(10))
            print 'Pre error correction result for', file_strings_test[i], 'is:', score

            # LABELS
            space_labels = list(t1_dtls[:, 5])
            char_labels = hm.labels_to_words(letters)

            # Spaces added to the classified characters.
            word_tester = (''.join(sm.chars_to_words(space_labels, char_labels))).split()
            # This is error correction!
            error_ct_lis = sm.error_correction(all_words, word_tester)
            # print error_ct_lis  # This is a print out of the whole word list, ie the error corrected output.
            score = hm.return_percentage(t1_labels_lis, list(''.join(error_ct_lis)))

            print 'The post error correction result for', file_strings_test[i], 'is:', score
            print 'Time taken was:', round(time.time() - start, 2)



