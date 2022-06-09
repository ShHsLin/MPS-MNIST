import numpy as np


x = np.random.random((5,1,3))
y = np.random.random((2,1))

""""
(A,B) @ (C,D,E) -> (C,A,E)


"""


z = y @ x
print(z.shape)

