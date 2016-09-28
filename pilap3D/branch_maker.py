import numpy as np

from .sqrdist import sqrdist, sqrdist_grad
from .totlen import totlen, totlen_grad
from .lenvar import lenvar, lenvar_grad
from .arcinc import arcinc, arcinc_grad
from .coincidence import coincidence, coincidence_grad

from .BFGSiped import BFGSiped

class BranchMaker:
    def __init__(self, nodes):
        self.set_nodes(nodes)
        self.set_penalties()
        
    def set_nodes(self, nodes):
        self.nodes = nodes.copy()
        self.scale_shift = self.nodes.mean(axis=0)
        self.nodes -= self.scale_shift
        self.scale_factor = self.nodes.max()
        self.nodes /= self.scale_factor
    
    def set_penalties(self, totlen_penalty=0.01, lenvar_penalty=0.01, arcinc_penalty=0.0001, coincidence_penalty=10):
        self.totlen_penalty = totlen_penalty
        self.lenvar_penalty = lenvar_penalty
        self.arcinc_penalty = arcinc_penalty    
        self.coincidence_penalty = coincidence_penalty
    
    def _total_cost(self, segments, verbose=False):
        if verbose:
            print('    sqrdist %e\n'
                  '     totlen %e\n'
                  '     lenvar %e\n'
                  '     arcinc %e\n'
                  'coincidence %e' % (np.mean(np.min(sqrdist(segments, self.nodes), axis=1)),
                                      totlen(segments),
                                      lenvar(segments),
                                      arcinc(segments),
                                      coincidence(segments)
                                     )
                 )
    
        return (np.mean(np.min(sqrdist(segments, self.nodes), axis=1))
                + self.totlen_penalty * totlen(segments)
                + self.lenvar_penalty * lenvar(segments)
                + self.arcinc_penalty * arcinc(segments)
                + self.coincidence_penalty * coincidence(segments)
               )
    
    def _total_cost_grad(self, segments):
        min_segm_ind = np.argmin(sqrdist(segments, self.nodes), axis=1)
        segm_ind = np.arange(segments.size/6)
        nonzero = (min_segm_ind[:, np.newaxis] == segm_ind[np.newaxis, :])
        
        return (np.mean(sqrdist_grad(segments, self.nodes)*nonzero[:, :, np.newaxis], axis=0).ravel()
                + self.totlen_penalty * totlen_grad(segments)
                + self.lenvar_penalty * lenvar_grad(segments)
                + self.arcinc_penalty * arcinc_grad(segments)
                + self.coincidence_penalty * coincidence_grad(segments)
               )
    
    def segments2points(self, segments):
        c = segments[:, :3]
        l = segments[:, 3]
        theta = segments[:, 4]
        phi = segments[:, 5]
        t = np.c_[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
        
        return np.vstack((c, c[-1] + l[-1]*t[-1]))
    
    def points2segments(self, points):
        dp = np.diff(points, axis=0)
        c = points[:-1]
        l = np.sqrt(np.sum(dp**2, axis=-1))
        theta = np.arccos(dp[:, 2]/l)
        phi = np.arctan2(dp[:, 1], dp[:, 0])
        
        return np.c_[c, l, theta, phi]
        
    def make_initial_guess(self, numsegments):
        max_range_axis = np.argmax(np.max(self.nodes, axis=0) - np.min(self.nodes, axis=0))

        first = self.nodes[np.argmin(self.nodes[:, max_range_axis])]
        last = self.nodes[np.argmax(self.nodes[:, max_range_axis])]

        x0 = np.c_[np.linspace(first[0], last[0], num=numsegments+1),
                   np.linspace(first[1], last[1], num=numsegments+1),
                   np.linspace(first[2], last[2], num=numsegments+1),
                  ]
        return x0
    
    def fit(self, numsegments, x0=None):
        if x0 is None:
            x0 = self.make_initial_guess(numsegments)       
        else:
            x0 -= self.scale_shift
            x0 /= self.scale_factor
        x0 = self.points2segments(x0).ravel()
        
        x = BFGSiped(self._total_cost, self._total_cost_grad, x0, gtol=1e-6, maxiter=1500)
        min_segm_ind = np.argmin(sqrdist(x, self.nodes), axis=1)
        
        x = self.segments2points(x.reshape((-1, 6)))
        x *= self.scale_factor
        x += self.scale_shift
        
        return x, min_segm_ind