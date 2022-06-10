import numpy as np


def ortho(data):
    print(data.shape)
    qstate = np.array([np.exp(data * 3.j * np.pi / 2) * np.cos(data * np.pi / 2),
                       np.exp(-data * 3.j * np.pi / 2) * np.sin(data * np.pi / 2)])
    return qstate.transpose((1,2,3,0))
