import numpy as np

def overlap(psi1, psi2):
    physical_contract = np.sum([psi1[:, i].conj() * psi2[:, :, :, i] for i in range(psi1.shape[-1])],
                               axis=0)
    virtual_contract = np.prod(physical_contract, 2)
    overlap = np.sum(virtual_contract, 1)
    overlap = np.abs(overlap)
    return overlap