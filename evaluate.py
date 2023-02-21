import json
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm


def normalizekm(K):
    v = np.sqrt(np.diag(K));
    nm =  np.outer(v,v)
    Knm = np.power(nm, -1)
    Knm = np.nan_to_num(Knm)
    normalized_K = K * Knm;
    return normalized_K


def eval_10_fold(dataset_name, K, y):
    # Set the seed for uniform parameter distribution
    random.seed(None)
    np.random.seed(None)

    # Number of splits of the data
    splits = 10

    # Normalize kernel matrix
    print("Normalizing kernel matrix...")
    K = normalizekm(K)

    # Load targets
    y = y.reshape((-1,1))
    y = np.ravel(y)

    with open('data_splits/'+dataset_name+'_splits.json','rt') as f:
        for line in f:
            splits = json.loads(line)

    # Size of the dataset
    n = K.shape[0]

    #################################
    # --- SET UP THE PARAMETERS --- #
    #################################
    C_grid = 10. ** np.arange(-7,7,2) / n
    trials = C_grid.size

    ##############################################################
    # --- MAIN CODE: PERMUTE, SPLIT AND EVALUATE PEFORMANCES --- #
    ##############################################################

    correct_pred = []
    val_split = []
    test_split = []

    acc = []
    for it in range(10):
        print("Starting split %d..." % it)
        train_index = splits[it]['model_selection'][0]['train']
        val_index = splits[it]['model_selection'][0]['validation']
        test_index = splits[it]['test']


        # Split the targets
        y_train = y[train_index]
        y_val = y[val_index]
        y_test = y[test_index]

        # Record the performance for each parameter trial
        # respectively on validation and test set
        perf_all_val = np.zeros(trials)
        perf_all_test = np.zeros(trials)

        # Split the kernel matrices
        K_train = K[np.ix_(train_index, train_index)]
        K_val = K[np.ix_(val_index, train_index)]

        #####################################################################
        # --- RUN THE MODEL: FOR A GIVEN SPLIT AND EACH PARAMETER TRIAL --- #
        #####################################################################

        # For each parameter trial
        for i in range(trials):

            #print("\nStarting experiment for trial %d and parameter C = %3f \n\n" % (i, C_grid[i]))

            # Fit classifier1 on training data
            clf = svm.SVC(kernel='precomputed', C = C_grid[i])
            clf.fit(K_train, y_train)

            # predict on validation and test
            y_pred = clf.predict(K_val)

            # accuracy on validation set
            acc = accuracy_score(y_val, y_pred)
            perf_all_val[i] = acc


        #######################################
        # --- FIND THE OPTIMAL PARAMETERS --- #
        #######################################

        # get optimal parameter on validation (argmax accuracy)
        max_idx = np.argmax(perf_all_val)
        C_opt = C_grid[max_idx]

        # performance corresponsing to the optimal parameter on validation
        perf_val_opt = perf_all_val[max_idx]

        clf = svm.SVC(kernel='precomputed', C=C_opt)
        clf.fit(K[np.ix_(train_index, train_index)], y[train_index])
        y_pred = clf.predict(K[np.ix_(test_index, train_index)])
        perf_test_opt = accuracy_score(y_test, y_pred)

        print("The best performance is for parameter C = %3f" % C_opt)
        print("The best performance on the validation set is: %3f" % perf_val_opt)
        print("The corresponding performance on test set is: %3f" % perf_test_opt)
        print()

        # append the best performance on validation at the current split
        val_split.append(perf_val_opt)

        # append the correponding performance on the test set
        test_split.append(perf_test_opt)


    ###############################
    # --- AVERAGE THE RESULTS --- #
    ###############################

    print("\nMean performance on val set for kernel: %3f" % np.mean(np.asarray(val_split)))
    print("With standard deviation: %3f" % np.std(np.asarray(val_split)))
    print("\nMean performance on test set for kernel: %3f" % np.mean(np.asarray(test_split)))
    print("With standard deviation: %3f" % np.std(np.asarray(test_split)))
