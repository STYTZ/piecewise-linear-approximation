import numpy as np

def totlen(segments):
    segments = segments.reshape((-1, 6))
    l = segments[:, 3]
    return 0.5*np.sum(l**2)


def totlen_grad(segments):
    segments = segments.reshape((-1, 6))
    l = segments[:, 3]
    
    grad = np.zeros_like(segments)
    grad[:, 3] = l
    return grad.ravel()