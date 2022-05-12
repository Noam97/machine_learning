import numpy as np
import sys

def z_score(train, test):
    m = np.mean(train, axis = 0)
    dev = np.std(train, axis = 0)
    norma_train = (train - m)/ dev
    norma_test = (test - m)/dev
    return norma_train, norma_test

def KNN(k,train_x_arr, train_y_arr, test_x_arr):
    y_hat_lst = []
    for vec in test_x_arr:
        dst_lst = []
        for train in train_x_arr:
            distance = np.linalg.norm(vec - train)
            dst_lst.append(distance)
        y_min_dst = []
        for i in range (k):
            min_distance = min(dst_lst)
            idx_min_distance = dst_lst.index(min_distance)
            y_min_dst.append(train_y_arr[idx_min_distance])
            dst_lst[idx_min_distance] = np.inf

        values, counts = np.unique(y_min_dst, return_counts=True)
        max_count = max(counts)
        idx_max_count = np.where(counts == max_count)[0][0]
        y_hat = values[idx_max_count]
        y_hat_lst.append(y_hat)
    return y_hat_lst

def Perceptron(train_x_arr, train_y_arr, test_x_arr, number_weights, number_dimension):
    weights_arr = np.zeros((number_weights, number_dimension))
    bias = np.zeros(number_weights)
    eta = 0.01
    y_hat_perceptron = []
    for i in range(3):
        for idx_vector, vector in enumerate(train_x_arr):
            guess = -1
            max_update = np.NINF
            for index_w, w in enumerate(weights_arr):
                update = np.dot(w, vector) + bias[index_w]
                if max_update < update:
                    max_update = update
                    guess = index_w

            int_train_y_arr = int(train_y_arr[idx_vector])
            if guess != int_train_y_arr:
                weights_arr[guess] = np.subtract(weights_arr[guess], np.multiply(vector, eta))
                weights_arr[int_train_y_arr] = np.add(weights_arr[int_train_y_arr], np.multiply(vector, eta))
                bias[guess] -= eta
                bias[int_train_y_arr] += eta

    for idx_vector, test_x in enumerate(test_x_arr):
        y_hat = -1
        max_update = np.NINF
        for index_w, w in enumerate(weights_arr):
            update = np.dot(w, test_x) + bias[index_w]
            if max_update < update:
                max_update = update
                y_hat = index_w

        y_hat_perceptron.append(y_hat)
    return y_hat_perceptron


def SVM(train_x_arr, train_y_arr, test_x_arr, number_weights, number_dimension,eta,lam):
    weights_arr = np.zeros((number_weights, number_dimension))
    bias = np.zeros(number_weights)
    for i in range(3):
        for idx_x, vector_x in enumerate(train_x_arr):
            guess = -1
            max_update = np.NINF
            correct = int(train_y_arr[idx_x])
            for idx_w, weight in enumerate(weights_arr):
                if correct == idx_w:
                    continue
                update = np.dot(weight, vector_x) + bias[idx_w]
                if max_update < update:
                    max_update = update
                    guess = idx_w
            loss = 1-(np.dot(weights_arr[correct],vector_x) + bias[correct]) + max_update

            for idx_w in range(len(weights_arr)):
                weights_arr[idx_w] = np.multiply(weights_arr[idx_w],1-(eta*lam))

            if loss > 0:
                weights_arr[guess] = np.subtract(weights_arr[guess], np.multiply(vector_x,eta))
                weights_arr[correct] = np.add(weights_arr[correct], np.multiply(vector_x, eta))
                bias[guess] = bias[guess] - eta
                bias[correct] = bias[correct] + eta

    y_hat_svm = []
    for vector_x in test_x_arr:
        guess = -1
        max_update = np.NINF
        for idx_w, w in enumerate(weights_arr):
            update = np.dot(w, vector_x) + bias[idx_w]
            if max_update < update:
                max_update = update
                guess = idx_w
        y_hat_svm.append(guess)
    return y_hat_svm

def PA(train_x_arr, train_y_arr, test_x_arr, number_weights, number_dimension):
    weights_arr = np.zeros((number_weights, number_dimension))
    bias = np.zeros(number_weights)
    for i in range(15):
        for idx_x, vector_x in enumerate(train_x_arr):
            guess = -1
            max_update = np.NINF
            for idx_w, weights in enumerate(weights_arr):
                update = np.dot(weights, vector_x) + bias[idx_w]
                if max_update < update:
                    max_update = update
                    guess = idx_w

            norma = 2 * pow(np.linalg.norm(vector_x), 2)
            int_train_y_arr = int(train_y_arr[idx_x])
            loss =max(0,1-(np.dot(weights_arr[int_train_y_arr] , vector_x)+ bias[int_train_y_arr])  + max_update)

            if norma!=0:
                tau = loss/norma
                tau_multiply_x = np.dot(tau, vector_x)
                weights_arr[int_train_y_arr] = np.add(tau_multiply_x, weights_arr[int_train_y_arr])
                weights_arr[guess] = np.subtract(weights_arr[guess], tau_multiply_x)
                bias[int_train_y_arr] += tau
                bias[guess] -= tau
    y_hat_pa = []
    for vector_x in test_x_arr:
        guess = -1
        max_update = np.NINF
        for idx_w, weights in enumerate(weights_arr):
            update = np.dot(weights, vector_x) + bias[idx_w]
            if max_update < update:
                max_update = update
                guess = idx_w
        y_hat_pa.append(guess)
    return y_hat_pa

def main():
    eta = 0.1
    lamb = 0.02
    train_x, train_y, test_x, out_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x_arr = np.array(np.loadtxt(train_x,delimiter=','))
    train_y_arr = np.array(np.loadtxt(train_y, delimiter=','))
    test_x_arr = np.array(np.loadtxt(test_x, delimiter=','))

    train_x_arr, test_x_arr = z_score(train_x_arr, test_x_arr)
    y_hat_knn = KNN(5, train_x_arr, train_y_arr, test_x_arr)
    y_hat_perceptron = Perceptron(train_x_arr, train_y_arr, test_x_arr, 3, 5)
    y_hat_pa = PA(train_x_arr, train_y_arr, test_x_arr, 3, 5)
    y_hat_svm = SVM(train_x_arr, train_y_arr, test_x_arr, 3, 5, eta, lamb)

    out_file = open(out_fname, "w")
    for i in range (len(y_hat_knn)):
        out_file.write(f"knn: {int(y_hat_knn[i])}, perceptron: {y_hat_perceptron[i]}, svm: {y_hat_svm[i]}, pa: {y_hat_pa[i]}\n")

main()
