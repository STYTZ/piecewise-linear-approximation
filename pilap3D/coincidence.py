import numpy as np

def coincidence(segments):
    segments = segments.reshape((-1, 6))
    c = segments[:, :3]
    l = segments[:, 3]
    theta = segments[:, 4]
    phi = segments[:, 5]
    t = np.c_[np.sin(theta)*np.cos(phi),
              np.sin(theta)*np.sin(phi),
              np.cos(theta)
             ]
    return 0.5*np.sum((np.diff(c, axis=0) - l[:-1, np.newaxis] * t[:-1])**2)


def coincidence_grad(segments):
    segments = segments.reshape((-1, 6))
    c = segments[:, :3]
    dc = np.diff(c, axis=0)
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
    
    grad[:-1, :3] += -(dc - l[:-1, np.newaxis] * t[:-1])
    grad[1:, :3] += (dc - l[:-1, np.newaxis] * t[:-1])
    
    grad[:-1, 3] += l[:-1] - np.sum(dc*t[:-1], axis=-1)
    
    grad[:-1, 4] += -np.sum(dc*l[:-1, np.newaxis]*dt_th[:-1], axis=-1)
    grad[:-1, 5] += -np.sum(dc*l[:-1, np.newaxis]*dt_ph[:-1], axis=-1)
    
    return grad.ravel()