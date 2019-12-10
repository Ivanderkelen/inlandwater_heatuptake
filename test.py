a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c= np.concatenate((a, b), axis=0)

np.concatenate((a, b.T), axis=1)

np.concatenate((a, b), axis=None)
