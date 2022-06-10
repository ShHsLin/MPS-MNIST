import numpy as np


def cycle_dims_anticlockwise(ndarray):
    return ndarray.transpose(list(range(1, ndarray.ndim)) + [0])


def ortho(data):
    qstate = np.array([np.exp(data * 3.j * np.pi / 2) * np.cos(data * np.pi / 2),
                       np.exp(-data * 3.j * np.pi / 2) * np.sin(data * np.pi / 2)])
    return cycle_dims_anticlockwise(qstate)


def origin(data):
    """
    Apply map, add physical dimensions to the end:
    (D,N,WxH,P)
    @param data:
    @return:
    """
    qstate = np.array([np.cos(data * np.pi / 2), np.sin(data * np.pi / 2)])
    return cycle_dims_anticlockwise(qstate)
