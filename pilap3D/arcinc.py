import numpy as np

def arcinc(segments):
    segments = segments.reshape((-1, 6))
    l = segments[:, 3]
    theta = segments[:, 4]
    phi = segments[:, 5]
    t = np.c_[np.sin(theta)*np.cos(phi),
              np.sin(theta)*np.sin(phi),
              np.cos(theta)
             ]
    return np.sum( (1 - np.sum(t[:-1]*t[1:], axis=-1)) * (l[:-1]**2 + l[1:]**2) )


def arcinc_grad(segments):
    segments = segments.reshape((-1, 6))
    l = segments[:, 3]
    theta = segments[:, 4]
    phi = segments[:, 5]
    t = np.c_[np.sin(theta)*np.cos(phi),
              np.sin(theta)*np.sin(phi),
              np.cos(theta)
             ]
    dt_th = np.c_[np.cos(theta)*np.cos(phi),
                 np.cos(theta)*np.sin(phi),
                 -np.sin(theta)
                ]
    dt_ph = np.c_[-np.sin(theta)*np.sin(phi),
                 np.sin(theta)*np.cos(phi),
                 np.zeros_like(theta)
                ]
    
    grad = np.zeros_like(segments)
    
    grad[:-1, 3] += 2 * l[:-1] * (1 - np.sum(t[:-1]*t[1:], axis=-1))
    grad[1:, 3] += 2 * l[1:] * (1 - np.sum(t[:-1]*t[1:], axis=-1))
    
    grad[:-1, 4] += - np.sum(dt_th[:-1]*t[1:], axis=-1) * (l[:-1]**2 + l[1:]**2)
    grad[1:, 4] += - np.sum(t[:-1]*dt_th[1:], axis=-1) * (l[:-1]**2 + l[1:]**2)
    
    grad[:-1, 5] += - np.sum(dt_ph[:-1]*t[1:], axis=-1) * (l[:-1]**2 + l[1:]**2)
    grad[1:, 5] += - np.sum(t[:-1]*dt_ph[1:], axis=-1) * (l[:-1]**2 + l[1:]**2)
    
    return grad.ravel()