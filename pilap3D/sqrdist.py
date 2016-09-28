import numpy as np

def H(x):
    """Heaviside function with penalty"""
    return 10 * (x > 0)


def sqrdist(segments, nodes):
    segments = segments.reshape((-1, 6))

    c = segments[:, :3]
    l = segments[:, 3]
    theta = segments[:, 4]
    phi = segments[:, 5]
    t = np.c_[np.sin(theta)*np.cos(phi),
              np.sin(theta)*np.sin(phi),
              np.cos(theta)
             ]
    
    n_c = nodes[:, np.newaxis, :] - c[np.newaxis, :, :]
    n_c_t = np.sum(n_c * t[np.newaxis, :, :], axis=-1)
    
    return 0.5*(np.sum(n_c**2, axis=-1)
                - n_c_t**2
                + H(n_c_t - l[np.newaxis, :])*(n_c_t - l[np.newaxis, :])**2
                + H(-n_c_t)*n_c_t**2)


def sqrdist_grad(segments, nodes):
    segments = segments.reshape((-1, 6))

    c = segments[:, :3]
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
        
    n_c = nodes[:, np.newaxis, :] - c[np.newaxis, :, :]
    n_c_t = np.sum(n_c * t[np.newaxis, :, :], axis=-1)
    n_c_dt_th = np.sum(n_c * dt_th[np.newaxis, :, :], axis=-1)
    n_c_dt_ph = np.sum(n_c * dt_ph[np.newaxis, :, :], axis=-1)
    
    grad = np.zeros(nodes.shape[:1] + segments.shape)

    grad[:, :, :3] = (- n_c
                      + n_c_t[:, :, np.newaxis]*t[np.newaxis, :, :]
                      - (H(n_c_t - l[np.newaxis, :])*(n_c_t - l[np.newaxis, :]))[:, :, np.newaxis]*t[np.newaxis, :, :]
                      - (H(-n_c_t)*n_c_t)[:, :, np.newaxis]*t[np.newaxis, :, :]
                     )
    
    grad[:, :, 3] = - (H(n_c_t - l[np.newaxis, :])*(n_c_t - l[np.newaxis, :]))
    
    grad[:, :, 4] = (- n_c_t*n_c_dt_th
                     + H(n_c_t - l[np.newaxis, :])*(n_c_t - l[np.newaxis, :])*n_c_dt_th
                     + H(-n_c_t)*n_c_t*n_c_dt_th)
    grad[:, :, 5] = (- n_c_t*n_c_dt_ph
                     + H(n_c_t - l[np.newaxis, :])*(n_c_t - l[np.newaxis, :])*n_c_dt_ph
                     + H(-n_c_t)*n_c_t*n_c_dt_ph)
    return grad #.ravel()