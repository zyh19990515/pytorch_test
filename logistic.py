import numpy as np

def sigma(y_hat):
    return 1/(1+np.exp(-y_hat))

def logistc_re():
    w = 0
    b = 0
    alpha = 0.05

    # Y = 3 * X + 5

    for i in range(0, 1000):
        print("This is " + str(i) + "times")

        X1 = np.random.randint(30, 50, 1000)
        X2 = np.random.randint(50, 100, 1000)
        # X1.dtype = np.float32
        # X2.dtype = np.float32
        # W = np.zeros(1000000, dtype=np.float32)
        # B = np.zeros(1000000, dtype=np.float32)
        X = np.hstack((X1, X2))
        # print(X)

        Y1 = np.ones(1000)
        Y2 = np.zeros(1000)
        Y = np.hstack((Y1, Y2))

        W = np.random.randint(1, 5, 2000)
        # W.dtype = np.float32
        # print(W)
        B = 0

        Z = np.dot(W.T, X) + B
        print(Z)
        A = sigma(Z)
        #print(A)
        dZ = A - Y
        dw = np.dot(X, dZ.T) / 2000
        db = np.sum(dZ) / 2000

        print("\n")
        w = w - alpha * dw
        b = b - alpha * db
    print(w)
    print(b)
    return w, b
if __name__ == '__main__':
    w, b = logistc_re()
    # yy = sigma(w*(0)+b)
    # print(yy)


