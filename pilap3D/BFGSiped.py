import numpy as np

def line_search(f, f0, df0, tau=0.5, c1=0.1, alpha=1., amin=1e-20):
    """Backtracking line search
    
    Backtrack alpha = tau*alpha until f(alpha) < f(0) + c1*alpha*df0
    """
    
    if f(alpha) <= f0 + c1*alpha*df0:
        return alpha
    
    while alpha > amin:
        alpha = tau*alpha
        if f(alpha) <= f0 + c1*alpha*df0:
            return alpha
        
    return None
    
    
def BFGSiped(f, fgrad, x0, args=(), gtol=1e-4, maxiter=1500):
    """BFGS method for function optimization."""
    
    xk = x0.copy()
    gk = fgrad(xk, *args)
    I = np.eye(xk.size)
    H = I
    
    iternum = 0
    while np.linalg.norm(gk, ord=np.Inf) > gtol and iternum < maxiter:
        pk = -np.dot(H, gk)
        if np.dot(pk, gk) > 0:
            pk *= -1
        
        alpha = line_search(lambda alpha: f(xk + alpha*pk, *args),
                            f(xk, *args),
                            np.dot(pk, gk)
                           )
                           
        xk1 = xk + alpha*pk
        gk1 = fgrad(xk1, *args)

        sk = xk1 - xk
        yk = gk1 - gk

        rhok = 1.0 / (np.dot(yk, sk))
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        H = np.dot(A1, np.dot(H, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])    

        xk = xk1
        gk = gk1

        iternum += 1
    
    return xk