import numpy as np

def lenvar(segments):
    segments = segments.reshape((-1, 6))
    l = segments[:, 3]
    return 0.5*np.sum( (l - np.mean(l))**2 )


def lenvar_grad(segments):
    segments = segments.reshape((-1, 6))
    l = segments[:, 3]
    
    grad = np.zeros_like(segments)
    grad[:, 3] = l - np.mean(l)
    return grad.ravel()