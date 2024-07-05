"""
Author: Yuxuan Mei, Etienne Vouga
"""

import numpy as np
from .helpers import *

# One potential in the physical simulation, V(q,s) = 0

class Potential:
       
    # Evaluates the potential function at config. Should return a scalar.
    def V(self, config):
        pass
    
    # The potential derivative (force), an N-vector where N is the dimension of the configuration space.
    def dV(self, config):
        pass
        
    # The potential Hessian, as a sparse matrix.    
    # Returns a (data, (i,j)) tuple of lists.
    def HV(self, config):
        pass

class MultiConnPotential(Potential):
    def __init__(self, mcf1ids, mcf2ids, tdists, dists, nparts):
        self.mcf1ids = mcf1ids
        self.mcf2ids = mcf2ids
        self.tdists = tdists
        self.dists = dists
        self.nparts = nparts

    def V(self, config):
        # we want the actual distance to be larger than min_tdist so to minimize the potential, use min_tdist - actual_dist
        def compute(i):
            return (self.tdists[i] - approx_abs(config[self.mcf2ids[i] + self.nparts * 6] - config[self.mcf1ids[i] + self.nparts * 6])) * self.dists[i]
        return sum([compute(i) for i in range(len(self.mcf1ids))])

    def dV(self, config):
        result = np.zeros(config.size)
        for i in range(len(self.mcf1ids)):
            gradV = approx_abs_grad(config[self.mcf2ids[i] + self.nparts * 6] - config[self.mcf1ids[i] + self.nparts * 6])
            result[self.mcf1ids[i] + self.nparts * 6] = self.dists[i] * gradV
            result[self.mcf2ids[i] + self.nparts * 6] = -self.dists[i] * gradV
        return result

    def HV(self, config):
        data = []
        ivals = []
        jvals = []

        for i in range(len(self.mcf1ids)):
            hessV = self.dists[i] * approx_abs_hess(config[self.mcf2ids[i] + self.nparts * 6] - config[self.mcf1ids[i] + self.nparts * 6])
            data.append(hessV)
            ivals.append(self.mcf2ids[i] + self.nparts * 6)
            jvals.append(self.mcf2ids[i] + self.nparts * 6)
            data.append(-hessV)
            ivals.append(self.mcf2ids[i] + self.nparts * 6)
            jvals.append(self.mcf1ids[i] + self.nparts * 6)
            data.append(-hessV)
            ivals.append(self.mcf1ids[i] + self.nparts * 6)
            jvals.append(self.mcf2ids[i] + self.nparts * 6)
            data.append(hessV)
            ivals.append(self.mcf1ids[i] + self.nparts * 6)
            jvals.append(self.mcf1ids[i] + self.nparts * 6)
        return (data, (ivals, jvals))

# Gravity.
#   parts:  The parts. The masses are read from this.
#   frames: Frames about which all of the parts are linearized.
#   g:      Acceleration of gravity. A 3-vector.    

class GravityPotential(Potential):
    
    def __init__(self, parts, frames, g, id_to_index):
        self.parts = parts
        self.frames = frames
        self.g = g
        self.id_to_index = id_to_index
    
    def V(self, config):
        result = 0
        for id, part in self.parts:
            i = self.id_to_index[id]
            partframe = self.frames[i]
            parttranslation = config[6*i:6*i+3]
            
            partt = homogeneousMatFromTranslation(parttranslation)
        
            pos = (partframe @ partt)[:3,3]
            localV = -part.mass * np.dot(self.g, pos)
            result = result + localV
        return result
        
    def dV(self, config):
        result = np.zeros(config.size)
        for id, part in self.parts:
            i = self.id_to_index[id]
            partframe = self.frames[i]
            
            dpartt = dhomogeneousMatFromTranslation()
            
            term1 = (partframe @ dpartt)[:,:3,3]
            localF1 = -part.mass * term1 @ self.g
            result[6*i:6*i+3] = localF1            
        return result    
        
    def HV(self, config):
        return ([],([], []))
        
# Frame target potential.
#   primitive:  The primitive that should match the target frame.
#   frames:     The frames about which all of the parts are linearized.
#   targetF:    The target frame itself. 4x4 homogeneous matrix.
#   axismask:   Tuple of three bools indicating whether alignment with that frame's axis should be enforced. 
#               (False, False, False) aligns only the center of mass. (True, False, False) aligns the center of mass and x axis, etc.
class TargetPotential(Potential):
    def __init__(self, primitive, frames, targetF, axismask):
        self.primitive = primitive
        self.frames = frames
        self.targetF = targetF
        self.setMaskVector(axismask)
        
    def setMaskVector(self, axismask):
        mv = np.zeros((4,4))
        for i in range(3):
            if axismask[i]:
                for j in range(3):
                    mv[j,i] = 1.0
            mv[i,3] = 1.0
        self.maskVector = mv.reshape(16)        
    
    def V(self, config):
        pidx = self.primitive.parent.id
        partframe = self.frames[pidx]
        parttranslation = config[6*pidx:6*pidx+3]
        partaxisangle = config[6*pidx+3:6*pidx+6]
        
        partrot = homogeneousMatFromAxisAngle(partaxisangle)
        partt = homogeneousMatFromTranslation(parttranslation)        
        F = ((partframe @ partt) @ partrot) @ self.primitive.frame().homogeneous_mat()
        diffVec = np.multiply((F-self.targetF).reshape(16), self.maskVector)
        result = np.dot(diffVec, diffVec)
        return result
        
    def dV(self, config):
        result = np.zeros(config.size)
        pidx = self.primitive.parent.id
        partframe = self.frames[pidx]
        diffVec = np.multiply( (partframe @ self.primitive.frame().homogeneous_mat() - self.targetF).reshape(16), self.maskVector)
        
        dpartrot = dhomogeneousMatFromAxisAngle()
        dpartt = dhomogeneousMatFromTranslation()
        
        dFt = (partframe @ dpartt @ self.primitive.frame().homogeneous_mat()).reshape((3,16))
        for i in range(3):
            dFt[i,:] = np.multiply(dFt[i,:], self.maskVector)
        localFt = 2.0 * dFt @ diffVec        
        result[6*pidx:6*pidx+3] = localFt
                
        dFr = (partframe @ dpartrot @ self.primitive.frame().homogeneous_mat()).reshape((3,16))
        for i in range(3):
            dFr[i,:] = np.multiply(dFr[i,:], self.maskVector)
        localFr = 2.0 * dFr @ diffVec        
        result[6*pidx+3:6*pidx+6] = localFr        
        
        return result    
        
    def HV(self, config):
        data = []
        ivals = []
        jvals = []
        
        pidx = self.primitive.parent.id
        partframe = self.frames[pidx]
        
        diffVec = np.multiply( (partframe @ self.primitive.frame().homogeneous_mat() - self.targetF).reshape(16), self.maskVector)
        
        dpartrot = dhomogeneousMatFromAxisAngle()
        dpartt = dhomogeneousMatFromTranslation()
        
        hpartrot = HhomogeneousMatFromAxisAngle()
        HFr = (partframe @ hpartrot @ self.primitive.frame().homogeneous_mat()).reshape((3,3,16))
        for i in range(3):
            for j in range(3):
                HFr[i,j,:] = np.multiply(HFr[i,j,:], self.maskVector)
        
        localHr = 2.0 * HFr @ diffVec        
        for i in range(3):
            for j in range(3):
                data.append(localHr[i,j])
                ivals.append(6*pidx+3+i)
                jvals.append(6*pidx+3+j)
                        
        dFt = (partframe @ dpartt @ self.primitive.frame().homogeneous_mat()).reshape((3,16))
        dFr = (partframe @ dpartrot @ self.primitive.frame().homogeneous_mat()).reshape((3,16))
        for i in range(3):
            dFt[i,:] = np.multiply(dFt[i,:], self.maskVector)
            dFr[i,:] = np.multiply(dFr[i,:], self.maskVector)
        
        localdtdr = 2.0 * dFt @ dFr.transpose()
        localdrdr = 2.0 * dFr @ dFr.transpose()
        localdtdt = 2.0 * dFt @ dFt.transpose()
        
        for i in range(3):
            for j in range(3):
                data.append(localdtdr[i,j])
                ivals.append(6*pidx+i)
                jvals.append(6*pidx+3+j)
                data.append(localdtdr[i,j])
                ivals.append(6*pidx+3+j)
                jvals.append(6*pidx+i)
                data.append(localdtdt[i,j])
                ivals.append(6*pidx+i)
                jvals.append(6*pidx+j)
                data.append(localdrdr[i,j])
                ivals.append(6*pidx+3+i)
                jvals.append(6*pidx+3+j)
        return (data, (ivals, jvals))