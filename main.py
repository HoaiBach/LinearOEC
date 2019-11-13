import LinearOEC
from sklearn.svm import LinearSVC
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
import pandas as pd
import scipy
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
import random

if __name__ == '__main__':
    import sys

    dataset = sys.argv[1]
    no_select = int(sys.argv[2])
    run_index = int(sys.argv[3])
    seed = 1617 * run_index
    np.random.seed(seed)
    random.seed = seed

    # load data
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/' + dataset + '.mat')
    X = mat['X']  # data
    X = X.astype(float)
    y = mat['Y']  # label
    y = y[:, 0]

    # load folds
    fold_read = open('/home/nguyenhoai2/Grid/data/FSMathlab_fold/' + dataset, 'r')
    lines = fold_read.readlines()
    no_folds = int(lines[0].split(': ')[1])
    train_indices = []
    test_indices = []
    for l_idx, line in enumerate(lines):
        if 'Fold: ' in line:
            train_line = lines[l_idx + 1]
            train_index = []
            for i_idx in train_line.split(': ')[1].split(', '):
                train_index.append(int(i_idx))
            train_indices.append(train_index)
            test_line = lines[l_idx + 2]
            test_index = []
            for i_idx in test_line.split(': ')[1].split(', '):
                test_index.append(int(i_idx))
            test_indices.append(test_index)
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    assert (n_classes == 2)
    min_class = np.min(count)
    unique_classes = np.unique(y)
    y[y == unique_classes[0]] = 0
    y[y == unique_classes[1]] = 1
    y = np.int8(y)

    no_folds = min(5, min_class)
    # sfold = StratifiedKFold(n_splits=no_folds, shuffle=True, random_state=1617)

    fold_idx = 1
    ave_full_knn = 0.0
    ave_full_svm = 0.0
    ave_sel_knn = 0.0
    ave_sel_svm = 0.0
    ave_time = 0.0
    to_print = 'LOEC\n'

    for train_index, test_index in zip(train_indices, test_indices):
        to_print += '*********** Fold %d ***********\n' % fold_idx
        fold_idx += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_min = np.min(X_train, axis=0)
        X_max = np.max(X_train, axis=0)
        non_dup = np.where(X_min != X_max)[0]
        X_min = X_min[non_dup]
        X_max = X_max[non_dup]
        X_train = X_train[:, non_dup]
        X_test = X_test[:, non_dup]
        X_train = 2 * (X_train - X_min) / (X_max - X_min) - 1
        X_test = 2 * (X_test - X_min) / (X_max - X_min) - 1

        import time

        start_time = time.time()
        classifier = LinearOEC.LinearOEC(X_train.shape[1], optimizer='cmaes', regularization=0.0)
        # classifier = LinearOEC(np.shape(data_array)[1], optimizer='cmaes',regularization=0.0,
        #                        initialization=model.coef_[0], iterations=10)
        classifier.fit(X_train, y_train)
        exe_time = time.time() - start_time
        W = np.abs(classifier.weights)
        f_selected = np.argsort(W)[::-1][:no_select]

        X_train_sel = X_train[:, f_selected]
        X_test_sel = X_test[:, f_selected]

        to_print += 'Time: %f \n' % exe_time

        to_print += 'Selected features: '
        for f_idx in f_selected:
            to_print += str(f_idx) + ', '
        to_print += '\n'

        knn = KNN(metric='euclidean')
        knn.fit(X_train, y_train)
        knn_full_acc = balanced_accuracy_score(y_test, knn.predict(X_test))
        knn_full_train_acc = balanced_accuracy_score(y_train, knn.predict(X_train))
        knn.fit(X_train_sel, y_train)
        knn_sel_acc = balanced_accuracy_score(y_test, knn.predict(X_test_sel))
        knn_sel_train_acc = balanced_accuracy_score(y_train, knn.predict(X_train_sel))
        to_print += 'Full train KNN: %f \n' % knn_full_train_acc
        to_print += 'Sel train KNN: %f \n' % knn_sel_train_acc
        to_print += 'Full KNN: %f \n' % knn_full_acc
        to_print += 'Sel KNN: %f \n' % knn_sel_acc

        clf = svm.LinearSVC(random_state=1617, C=1.0, penalty='l2')
        clf.fit(X_train, y_train)
        svm_full_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
        svm_full_train_acc = balanced_accuracy_score(y_train, clf.predict(X_train))
        clf.fit(X_train_sel, y_train)
        svm_sel_acc = balanced_accuracy_score(y_test, clf.predict(X_test_sel))
        svm_sel_train_acc = balanced_accuracy_score(y_train, clf.predict(X_train_sel))
        to_print += 'Full train SVM: %f \n' % svm_full_train_acc
        to_print += 'Sel train SVM: %f \n' % svm_sel_train_acc
        to_print += 'Full SVM: %f \n' % svm_full_acc
        to_print += 'Sel SVM: %f \n' % svm_sel_acc
        to_print += 'Number of selected features: %d \n' % len(f_selected)
        to_print += 'Time: %f \n' % exe_time

        ave_full_knn += knn_full_acc
        ave_full_svm += svm_full_acc
        ave_sel_knn += knn_sel_acc
        ave_sel_svm += svm_sel_acc
        ave_time += ave_time

    ave_sel_svm /= no_folds
    ave_sel_knn /= no_folds
    ave_full_svm /= no_folds
    ave_full_knn /= no_folds
    ave_time /= no_folds

    to_print += '***********************************Final results*****************************\n'
    to_print += 'Full SVM: %f \n' % ave_full_svm
    to_print += 'Sel SVM: %f \n' % ave_sel_svm
    to_print += 'Full KNN: %f \n' % ave_full_knn
    to_print += 'Sel KNN: %f \n' % ave_sel_knn

    f = open(str(run_index)+'.txt', 'w')
    f.write(to_print)
    f.close()
