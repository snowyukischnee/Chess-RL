import numpy as np

a = np.array([[1,2,3], [4,5,3]])
b = a[: np.newaxis]
c = np.expand_dims(a, axis=0)
print(a.shape, b.shape, c.shape)