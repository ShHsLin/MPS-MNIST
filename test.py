import numpy as np


x = np.random.random((6,5,1,3))
y = np.random.random((2,1))

""""
(A,B) @ (F, C,D,E) -> (C,A,E)
(1,1) @ (F, N,L,R) - > (F,N,1,R)

1 = L

"""


z = y @ x
print(z.shape)

