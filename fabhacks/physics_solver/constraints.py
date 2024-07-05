"""
Author: Yuxuan Mei, Etienne Vouga
"""

import numpy as np
from .helpers import *

# One constraint in the physical simulation, g(q,s) = 0
# A single Constraint object can encode multiple constraints (its dim).

class Constraint:
    def __init__(self, dim):
        self.dim = dim
        
    def dimension(self):
        return self.dim
        
    # Evaluates the constraint function at config. Should return a dimension-d vector.
    def g(self, config):
        pass
    
    # The sparse dim x N constraint derivative, where N is the dimension of the configuration space.
    # Returns a (data, (i,j)) tuple of lists.
    # !! Only the connector parameters are used: assumes the part parameters are all zero !!
    def dg(self, config):
        pass
        
    # The constraint Hessian, as a sparse matrix.    
    # Returns a list of dimension (data, (i,j)) tuples of lists.
    def Hg(self, config):
        pass
        
# Constraint that two parts should be connected together.
#   frames:                         The frames about which the parts are linearized.
#   primitive1, primitive2:         The part's primitives that are connected. No state of these primitives is used; just the functions
#                                   for computing the transformation matrices of the connector given the connector's parameters.
#   p1mcfid, p2mcfid:               The MCF ID of the connector on each primitive.
#   connectionmat:                  4 x 4 constant homogeneous matrix that aligns connector 1 to connector 2.
#   part1idx, part2idx:             The indices into `config` where the degrees of freedom of the two parts begin.
#                                   The first part's degrees of freedom are config[part1idx], ..., config[part1idx+6].
#   connector1indices, 
#   connector2indices:              The indices into `config` where the degrees of freedom of the two connectors begin.
#                                   The first connector's parameters begin at config[primitive1], etc.
#   nparams1, nparams2              The number of parameters that should be passed into the connector frame functions.

class ConnectorConstraint(Constraint):
    def __init__(self, id_to_index, frames, primitive1, primitive2, p1mcfid, p2mcfid, connectionmat, part1idx, part2idx, connector1indices, connector2indices, nparams1, nparams2):
        super().__init__(16)
        self.id_to_index = id_to_index
        self.frames = frames
        self.primitive1 = primitive1
        self.primitive2 = primitive2
        self.p1mcfid = p1mcfid
        self.p2mcfid = p2mcfid
        self.connectionmat = connectionmat
        self.part1idx = part1idx
        self.part2idx = part2idx
        self.connector1indices = connector1indices
        self.connector2indices = connector2indices
        self.nparams1 = nparams1
        self.nparams2 = nparams2

    def print_g(self, config, return_str=False):
        partframe1 = self.frames[self.id_to_index[self.primitive1.parent.id]]
        partframe2 = self.frames[self.id_to_index[self.primitive2.parent.id]]
        part1translation = config[self.part1idx:self.part1idx+3]
        part2translation = config[self.part2idx:self.part2idx+3]
        part1axisangle = config[self.part1idx+3:self.part1idx+6]
        part2axisangle = config[self.part2idx+3:self.part2idx+6]
        
        part1rot = homogeneousMatFromAxisAngle(part1axisangle)
        part2rot = homogeneousMatFromAxisAngle(part2axisangle)
        part1t = homogeneousMatFromTranslation(part1translation)
        part2t = homogeneousMatFromTranslation(part2translation)
        
        # WARNING: this is a bit hacky because potentially only one index is -1 but for now I'll use this hack
        cparams1 = [config[i] for i in self.connector1indices] if -1 not in self.connector1indices else self.primitive1.mcf_factory.MCFParams[self.p1mcfid].values()
        cparams2 = [config[i] for i in self.connector2indices] if -1 not in self.connector2indices else self.primitive2.mcf_factory.MCFParams[self.p2mcfid].values()

        connectormat1 = self.primitive1.mcf_factory.connectorFrame(*cparams1)
        connectormat2 = self.primitive2.mcf_factory.connectorFrame(*cparams2)
        
        lhs = partframe1 @ part1t @ part1rot @ connectormat1 @ self.connectionmat
        rhs = partframe2 @ part2t @ part2rot @ connectormat2
        if return_str:
            output = "ConnectorConstraint g()\n"
            output += f"{self.primitive1} {self.p1mcfid} {self.primitive2} {self.p2mcfid}\n"
            output += f"{partframe1 @ part1t @ part1rot @ connectormat1}\n{rhs}\n"
            output += f"partframe1 {partframe1 @ part1t @ part1rot}\n"
            output += f"partframe2 {partframe2 @ part2t @ part2rot}\n"
            output += f"cval {(lhs-rhs).reshape(16)}"
            return output
        else:
            print("ConnectorConstraint g()")
            print(self.primitive1, self.p1mcfid, self.primitive2, self.p2mcfid)
            print(partframe1 @ part1t @ part1rot @ connectormat1)
            print(rhs)
            print("partframe1", partframe1 @ part1t @ part1rot)
            print("partframe2", partframe2 @ part2t @ part2rot)
            print("partframe1")
            print(repr(partframe1))
            print("part1t")
            print(repr(part1t))
            print("part1rot")
            print(repr(part1rot))
            print("connectormat1")
            print(repr(connectormat1))
            return (lhs-rhs).reshape(16)

    def g(self, config):
        partframe1 = self.frames[self.id_to_index[self.primitive1.parent.id]]
        partframe2 = self.frames[self.id_to_index[self.primitive2.parent.id]]
        part1translation = config[self.part1idx:self.part1idx+3]
        part2translation = config[self.part2idx:self.part2idx+3]
        part1axisangle = config[self.part1idx+3:self.part1idx+6]
        part2axisangle = config[self.part2idx+3:self.part2idx+6]
        
        part1rot = homogeneousMatFromAxisAngle(part1axisangle)
        part2rot = homogeneousMatFromAxisAngle(part2axisangle)
        part1t = homogeneousMatFromTranslation(part1translation)
        part2t = homogeneousMatFromTranslation(part2translation)
        
        # WARNING: this is a bit hacky because potentially only one index is -1 but for now I'll use this hack
        cparams1 = [config[i] for i in self.connector1indices] if -1 not in self.connector1indices else self.primitive1.mcf_factory.MCFParams[self.p1mcfid].values()
        cparams2 = [config[i] for i in self.connector2indices] if -1 not in self.connector2indices else self.primitive2.mcf_factory.MCFParams[self.p2mcfid].values()

        connectormat1 = self.primitive1.mcf_factory.connectorFrame(*cparams1)
        connectormat2 = self.primitive2.mcf_factory.connectorFrame(*cparams2)
        
        lhs = partframe1 @ part1t @ part1rot @ connectormat1 @ self.connectionmat
        rhs = partframe2 @ part2t @ part2rot @ connectormat2
        return (lhs-rhs).reshape(16)

    def dg(self, config):
        data = []
        ivalues = []
        jvalues = []
        
        partframe1 = self.frames[self.id_to_index[self.primitive1.parent.id]]
        partframe2 = self.frames[self.id_to_index[self.primitive2.parent.id]]
                
        cparams1 = [config[i] for i in self.connector1indices] if -1 not in self.connector1indices else self.primitive1.mcf_factory.MCFParams[self.p1mcfid].values()
        cparams2 = [config[i] for i in self.connector2indices] if -1 not in self.connector2indices else self.primitive2.mcf_factory.MCFParams[self.p2mcfid].values()

        connectormat1 = self.primitive1.mcf_factory.connectorFrame(*cparams1)
        connectormat2 = self.primitive2.mcf_factory.connectorFrame(*cparams2)
        
        # There are four parts to this derivative: lhs depends on part1's DOFs and connector1's params.
        # rhs depends on the same but for part2 and connector2.
        
        # Part1's terms
        part1term1 = (partframe1 @ dhomogeneousMatFromTranslation() @ connectormat1 @ self.connectionmat).reshape(3, 16).transpose()
        for i in range(16):
            for j in range(3):
                data.append(part1term1[i,j])
                ivalues.append(i)
                jvalues.append(self.part1idx+j)
                
        part1term2 = (partframe1 @ dhomogeneousMatFromAxisAngle() @ connectormat1 @ self.connectionmat).reshape(3, 16).transpose()
        for i in range(16):
            for j in range(3):
                data.append(part1term2[i,j])
                ivalues.append(i)
                jvalues.append(self.part1idx+3+j)
        
        # Part2's terms
        part2term1 = -(partframe2 @ dhomogeneousMatFromTranslation() @ connectormat2).reshape(3, 16).transpose()        
        for i in range(16):
            for j in range(3):
                data.append(part2term1[i,j])
                ivalues.append(i)
                jvalues.append(self.part2idx+j)
        
        part2term2 = -(partframe2 @ dhomogeneousMatFromAxisAngle() @ connectormat2).reshape(3, 16).transpose()
        for i in range(16):
            for j in range(3):
                data.append(part2term2[i,j])
                ivalues.append(i)
                jvalues.append(self.part2idx+3+j)        
        
        # Connector1's terms
        connector1term = (partframe1 @ self.primitive1.mcf_factory.dConnectorFrame(*cparams1) @ self.connectionmat).reshape(self.nparams1, 16).transpose()
        for i in range(16):
            for j in range(self.nparams1):
                if self.connector1indices[j] != -1:
                    data.append(connector1term[i,j])
                    ivalues.append(i)
                    jvalues.append(self.connector1indices[j])

        # Connector2's terms
        connector2term = -(partframe2 @ self.primitive2.mcf_factory.dConnectorFrame(*cparams2)).reshape(self.nparams2, 16).transpose()
        for i in range(16):
            for j in range(self.nparams2):
                if self.connector2indices[j] != -1:
                    data.append(connector2term[i,j])
                    ivalues.append(i)
                    jvalues.append(self.connector2indices[j])

        return (data,(ivalues,jvalues))

    def Hg(self, config):
        
        partframe1 = self.frames[self.id_to_index[self.primitive1.parent.id]]
        partframe2 = self.frames[self.id_to_index[self.primitive2.parent.id]]
        
        cparams1 = [config[i] for i in self.connector1indices] if -1 not in self.connector1indices else self.primitive1.mcf_factory.MCFParams[self.p1mcfid].values()
        cparams2 = [config[i] for i in self.connector2indices] if -1 not in self.connector2indices else self.primitive2.mcf_factory.MCFParams[self.p2mcfid].values()
        
        connectormat1 = self.primitive1.mcf_factory.connectorFrame(*cparams1)
        connectormat2 = self.primitive2.mcf_factory.connectorFrame(*cparams2)
        
        data = [[] for i in range(16)]
        ientries = [[] for i in range(16)]
        jentries = [[] for i in range(16)]
        ## Pure Hessian terms        
        Hrot1 = (partframe1 @ HhomogeneousMatFromAxisAngle() @ connectormat1 @ self.connectionmat).reshape(3, 3, 16)
        for i in range(16):
            for j in range(3):
                for k in range(3):
                    data[i].append(Hrot1[j,k,i])
                    ientries[i].append(self.part1idx+3+j)
                    jentries[i].append(self.part1idx+3+k)
        
        Hrot2 = -(partframe2 @ HhomogeneousMatFromAxisAngle() @ connectormat2).reshape(3, 3, 16)
        for i in range(16):
            for j in range(3):
                for k in range(3):
                    data[i].append(Hrot2[j,k,i])
                    ientries[i].append(self.part2idx+3+j)
                    jentries[i].append(self.part2idx+3+k)
                
        Hcon1 = (partframe1 @ self.primitive1.mcf_factory.HConnectorFrame(*cparams1) @ self.connectionmat).reshape(self.nparams1, self.nparams1, 16)              
        
        for i in range(16):
            for j in range(self.nparams1):
                if self.connector1indices[j] != -1:
                    for k in range(self.nparams1):
                        if self.connector1indices[k] != -1:
                            data[i].append(Hcon1[j,k,i])
                            ientries[i].append(self.connector1indices[j])
                            jentries[i].append(self.connector1indices[k])
        
        Hcon2 = -(partframe2 @ self.primitive2.mcf_factory.HConnectorFrame(*cparams2)).reshape(self.nparams2, self.nparams2, 16)
        for i in range(16):
            for j in range(self.nparams2):
                if self.connector2indices[j] != -1:
                    for k in range(self.nparams2):
                        if self.connector2indices[k] != -1:
                            data[i].append(Hcon2[j,k,i])
                            ientries[i].append(self.connector2indices[j])
                            jentries[i].append(self.connector2indices[k])
                    
        dtdcon1 = np.einsum('ijk,lkm->iljm', dhomogeneousMatFromTranslation(), self.primitive1.mcf_factory.dConnectorFrame(*cparams1))
        Htcon1 = (partframe1 @ dtdcon1 @ self.connectionmat).reshape(3,self.nparams1,16)
        for i in range(16):
            for j in range(3):
                for k in range(self.nparams1):
                    if self.connector1indices[k] != -1:
                        data[i].append(Htcon1[j,k,i])
                        ientries[i].append(self.part1idx+j)
                        jentries[i].append(self.connector1indices[k])
                        data[i].append(Htcon1[j,k,i])
                        jentries[i].append(self.part1idx+j)
                        ientries[i].append(self.connector1indices[k])
                    
        dtdcon2 = np.einsum('ijk,lkm->iljm', dhomogeneousMatFromTranslation(), self.primitive2.mcf_factory.dConnectorFrame(*cparams2))
        Htcon2 = (-partframe2 @ dtdcon2).reshape(3,self.nparams2,16)
        for i in range(16):
            for j in range(3):
                for k in range(self.nparams2):
                    if self.connector2indices[k] != -1:
                        data[i].append(Htcon2[j,k,i])
                        ientries[i].append(self.part2idx+j)
                        jentries[i].append(self.connector2indices[k])
                        data[i].append(Htcon2[j,k,i])
                        jentries[i].append(self.part2idx+j)
                        ientries[i].append(self.connector2indices[k])
                    
        drdcon1 = np.einsum('ijk,lkm->iljm', dhomogeneousMatFromAxisAngle(), self.primitive1.mcf_factory.dConnectorFrame(*cparams1))
        Hrcon1 = (partframe1 @ drdcon1 @ self.connectionmat).reshape(3,self.nparams1,16)
        for i in range(16):
            for j in range(3):
                for k in range(self.nparams1):
                    if self.connector1indices[k] != -1:
                        data[i].append(Hrcon1[j,k,i])
                        ientries[i].append(self.part1idx+3+j)
                        jentries[i].append(self.connector1indices[k])
                        data[i].append(Hrcon1[j,k,i])
                        jentries[i].append(self.part1idx+3+j)
                        ientries[i].append(self.connector1indices[k])
        
        drdcon2 = np.einsum('ijk,lkm->iljm', dhomogeneousMatFromAxisAngle(), self.primitive2.mcf_factory.dConnectorFrame(*cparams2))
        Hrcon2 = (-partframe2 @ drdcon2).reshape(3,self.nparams2,16)
        for i in range(16):
            for j in range(3):
                for k in range(self.nparams2):
                    if self.connector2indices[k] != -1:
                        data[i].append(Hrcon2[j,k,i])
                        ientries[i].append(self.part2idx+3+j)
                        jentries[i].append(self.connector2indices[k])
                        data[i].append(Hrcon2[j,k,i])
                        jentries[i].append(self.part2idx+3+j)
                        ientries[i].append(self.connector2indices[k])
        
        result = []
        for i in range(16):
            result.append((data[i], (ientries[i],jentries[i])))
        return result