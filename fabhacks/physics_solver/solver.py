"""
Author: Yuxuan Mei, Etienne Vouga
"""

from ..assembly.assembly_R import *
import numpy as np
import scipy.sparse
from typing import NamedTuple
from sksparse.cholmod import *
from .helpers import *

# Represents bounds on some of the DOFs.
#   dofidx:     Index of the DOF to constraint within the configuration vector
#   minval:     Minimum allowable value
#   maxval:     Maximum allowalbe value
#   clamp:      What to do if the bound is violated. If clamp = True, the bounds are enforced as inequality constraints.
#               If clamp = False, the simulation aborts if optimization would cause the bound to be violated.
class DOFBound(NamedTuple):
    dofidx: int
    minval: float
    maxval: float
    clamp:  bool
    
# Options for the solver.
#   constraintWeight:           The penalty weight for all of the soft constraints
#   startingObjectiveWeight:    How much to weigh the user objective (vs. physical potentials) at the start of the optimization
#   maxIters:                   Maximum number of outer solver iterations.
#   tol:                        Convergence tolerance. Solve ends when the force magnitude is below this threshold.

class SolverParams(NamedTuple):
    constraintWeight:           float
    startingObjectiveWeight:    float
    maxIters:                   int
    tol:                        float

# Computes an M x N subset of the identity matrix that projects onto only those degrees of freedom that are *not* pinned.
#   pinned: Boolean list of size N. True entries are DOFs that will be excluded by the projection matrix.
def buildProjectionMatrix(pinned):
    data = []
    ivals = []
    jvals = []
    ndofs = len(pinned)
    
    nrows = 0
    
    for i in range(ndofs):
        if not pinned[i]:
            data.append(1.0)
            ivals.append(nrows)
            jvals.append(i)
            nrows = nrows + 1
    
    mat = scipy.sparse.coo_matrix((data,(ivals,jvals)), (nrows, ndofs))    
    return mat
    
# Computes the maximum step size allowed without violating one of the bound constraints.
# Returns (alpha, b) where alpha (in [0,1]) is the allowed step size and b is the violated DOFBound object.
# b = None if there is no violated constraint.
def boundLineSearch(curconfig, delta, DOFbounds):
    retb = None
    alpha = 1.0
    value = 0
    for b in DOFbounds:
        if delta[b.dofidx] > 0:
            t = (b.maxval - curconfig[b.dofidx])/delta[b.dofidx]
            if t < alpha:
                alpha = max(t, 0.0)
                retb = b
                value = b.maxval
        elif delta[b.dofidx] < 0:
            t = (b.minval - curconfig[b.dofidx])/delta[b.dofidx]
            if t < alpha:
                alpha = max(t, 0.0)
                retb = b            
                value = b.minval
    return (alpha, retb, value)
    
# Relinearizes all of the parts about the current configuration.
# Note that doing so might (and often will) disarticulate the Parts, depending on the configuration variable values.
def setFramesFromConfig(nparts, frames, config):    
    for i in range(nparts):
        partframe = frames[i]
        parttranslation = config[6*i:6*i+3]
        partaxisangle = config[6*i+3:6*i+6]
        
        partrot = homogeneousMatFromAxisAngle(partaxisangle)
        partt = homogeneousMatFromTranslation(parttranslation)
        
        newframe = partframe @ partt @ partrot
        frames[i] = newframe
        config[6*i:6*i+6] = np.zeros(6)
        
def staticSolver(nparts, potentials, constraints, objectives, DOFbounds, frames, guess, params, verbose = False):
    curconfig = guess
    curobjective = float('inf')
    ndofs = guess.size
    
    objweight = params.startingObjectiveWeight
    constraintK = params.constraintWeight
    tol = params.tol
    maxIters = params.maxIters
    
    useCHessians = True    
    pinned = [False]*ndofs
    
    nextobjweight = objweight
    
    for i in range(maxIters):
        setFramesFromConfig(nparts, frames, curconfig)        
        
        oldObj = 0
        objweight = nextobjweight
        
        deriv = np.zeros(ndofs)        
        hess = scipy.sparse.coo_array((ndofs, ndofs))
        curobjective = 0
        
        for p in potentials:
            oldObj = oldObj + p.V(curconfig)
            deriv = deriv + p.dV(curconfig)
            Hp = scipy.sparse.coo_array(p.HV(curconfig), (ndofs, ndofs))
            hess = hess + Hp
        for o in objectives:
            objective = o.V(curconfig)
            curobjective = curobjective + objective
            oldObj = oldObj + objweight*objective
            deriv = deriv + objweight*o.dV(curconfig)
            Hp = scipy.sparse.coo_array(o.HV(curconfig), (ndofs, ndofs))
            hess = hess + objweight*Hp
        
        for c in constraints:
            g = c.g(curconfig)
            oldObj = oldObj + constraintK * np.dot(g, g)
            dg = scipy.sparse.coo_array(c.dg(curconfig), (c.dimension(), ndofs))
            deriv = deriv + 2.0 * constraintK * dg.transpose() @ g
            dcdc = 2.0 * constraintK * (dg.transpose() @ dg)
            if useCHessians:
                Hcdata = []
                Hci = []
                Hcj = []
                coeffs = c.Hg(curconfig)
                for d in range(c.dimension()):
                    nterms = len(coeffs[d][0])
                    for j in range(nterms):
                        Hcdata.append(2.0 * constraintK * g[d] * coeffs[d][0][j])
                        Hci.append(coeffs[d][1][0][j])
                        Hcj.append(coeffs[d][1][1][j])
                Hc = scipy.sparse.coo_array((Hcdata, (Hci,Hcj)), (ndofs, ndofs))
                hess = hess + dcdc + Hc               
            else:
                hess = hess + dcdc
        
        P = buildProjectionMatrix(pinned)
        
        gradnorm = np.linalg.norm(P @ deriv)
        if gradnorm < tol:
            # Update the active set. First unpin any clamped constraints that are now inactive:                
            didunpin = False
            for b in DOFbounds:
                if pinned[b.dofidx] and b.clamp:
                    canunpin = True
                    if curconfig[b.dofidx] == b.maxval and deriv[b.dofidx] <= 0:
                        canunpin = False
                    if curconfig[b.dofidx] == b.minval and deriv[b.dofidx] >= 0:
                        canunpin = False
                    if canunpin:
                        if verbose: print("Unpinning ", b.dofidx, " val = ", curconfig[b.dofidx], " lmult = ", deriv[b.dofidx])
                        pinned[b.dofidx] = False
                        didunpin = True
                        
            if not didunpin:
                # Decrease the objective weight
                if objweight > params.tol:
                    nextobjweight = 0.1 * objweight
                # Next check if any unclamped constraints are still active. If so, the piece falls off
                else:
                    for b in DOFbounds:
                        if pinned[b.dofidx] and not b.clamp:
                            if (curconfig[b.dofidx] == b.maxval and deriv[b.dofidx] > 0) or (curconfig[b.dofidx] == b.minval and deriv[b.dofidx] < 0):
                                message = "Optimization Terminated: Connector Parameter ", b.dofidx - 6 * nparts, " Fell Off!"
                                return (float('inf'), curconfig[6*nparts:], message)
                    # Otherwise, optimization succeeded
                    print("Optimization Succeeded")
                    return (curobjective, curconfig[6*nparts:], "Optimization Succeeded")
            continue
        
        H = P @ hess @ P.transpose()
        reg = 0
        while True:
            ok = True
            try:
                factor = cholesky(H, beta = reg)
                delta = factor(-P @ deriv)
            except:
                ok = False
                
            if ok and np.dot(P.transpose() @ delta, deriv) > 0:
                ok = False
            
            if ok:
                (alpha, violb, value) = boundLineSearch(curconfig, P.transpose() @ delta, DOFbounds)
            
                newconfig = curconfig + alpha * P.transpose() @ delta
                
                newObj = 0
                for p in potentials:
                    newObj = newObj + p.V(newconfig)
                for o in objectives:
                    newObj = newObj + objweight * o.V(newconfig)                
                for c in constraints:
                    g = c.g(newconfig)
                    newObj = newObj + constraintK * np.dot(g, g)
                if newObj <= oldObj:
                    if verbose:
                        print("Old ", oldObj, " new ", newObj, " reg ", reg, " force ", np.linalg.norm(P @ deriv))
                    curconfig = newconfig
                    # Pin the DOF for the newly-active constraint:
                    if violb is not None:
                        pinned[violb.dofidx] = True
                        curconfig[violb.dofidx] = value
                        if verbose: print("Pinning ", violb.dofidx, " at val ", newconfig[violb.dofidx])
                    break
                
            # Step failed, increase regularization
            if reg == 0:
                reg = 1e-6
            else:
                reg = reg * 2
                
            if reg > 1e6:
                print("Optimization Terminated: Stalled")
                return (curobjective, curconfig[6*nparts:], "Optimization Terminated: Stalled")
                
    # max iters
    print("Optimization Terminated: Max Iterations")
    return (curobjective, curconfig[6*nparts:], "Optimization Terminated: Max Iterations")
