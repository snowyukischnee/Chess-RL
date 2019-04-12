import numpy as np

a = 1
b = np.ones((4,4), dtype=np.float)
c = np.ones(3, dtype=np.float)
d = (a,b,c)

a1 = 4
b1 = np.zeros((4,4), dtype=np.float)
c1 = np.ones(3, dtype=np.float)
d1 = (a1,b1,c1)


e = [d, d1]

aa, bb, cc = zip(*e)

print(np.asarray(aa).shape)

x = np.add([1,2], [5,6])
print(x, x.shape)