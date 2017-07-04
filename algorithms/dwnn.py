import numpy as np


def weight(base, query, sigma=0.25):
    euclidean = np.array([np.sqrt(np.sum((base[x,] - query)**2)) for x in range(len(base))])
    return np.exp((-euclidean**2)/(2*sigma**2))


def dwnn(base, query, sigma=0.25):
    class_i = base.shape[1] - 1

    X = np.array(base[:, 0:class_i])
    Y = np.array(base[:, [class_i]])

    w = weight(X, query, sigma)
    w = w.reshape(len(w), 1)
    Y_obtained = np.sum(Y*w)/np.sum(w)

    return Y_obtained


# def test_identity(sigma=0.25):
    # data = np.linspace(-10, 10, 100)
    # data = np.column_stack((data, data))

    # sq_error = 0
    # for i in range(0, len(data)):
        # query = data[i, 0]
        # expected = data[i, 1]
        # obtained = dwnn(data, query, sigma)
        # sq_error = sq_error + (expected - obtained)**2

    # sq_error = sq_error/len(data)
    # print(sq_error)


