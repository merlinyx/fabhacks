"""
Author: Yuxuan Mei, Etienne Vouga
"""

import numpy as np
import math

def approx_abs(x, mu = 0.01):
    return x * math.tanh(x / mu)

def approx_abs_grad(x, mu = 0.01):
    return math.tanh(x / mu) + x * (1 - math.tanh(x / mu) ** 2) / mu

def approx_abs_hess(x, mu = 0.01):
    return (1 - math.tanh(x / mu) ** 2) / mu + (mu - 2*x*math.tanh(x/mu)) * (1 - math.tanh(x / mu) ** 2) / mu ** 2

def homogeneousMatFromTranslation(translation):
    mat = np.zeros((4,4))
    mat[:3,:3] = np.eye(3)
    mat[:3, 3] = translation
    mat[3, 3] = 1
    return mat
    
def dhomogeneousMatFromTranslation():
    mat = np.zeros((3,4,4))    
    mat[:3, :3, 3] = np.eye(3)
    return mat

# Hessian of translation is zero


# Elementary Euler rotation matrices and their derivatives.
# In radians!!

def EulerX(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[1,0,0],[0,c,-s],[0,s,c]])
    
def dEulerX(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[0,0,0],[0,-s,-c],[0,c,-s]])
    
def HEulerX(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[0,0,0],[0,-c,s],[0,-s,-c]])

def EulerY(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[c,0,s],[0,1,0],[-s,0,c]])

def dEulerY(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[-s,0,c],[0,0,0],[-c,0,-s]])
    
def HEulerY(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[-c,0,-s],[0,0,0],[s,0,-c]])

def EulerZ(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[c,-s,0],[s,c,0],[0,0,1]])

def dEulerZ(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[-s,-c,0],[c,-s,0],[0,0,0]])

def HEulerZ(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.matrix([[-c,s,0],[-s,-c,0],[0,0,0]])     

# 4x4 homogeneous rotation matrix about given Euler angles (in 'xyz' order in Tait-Bryan angles).
# Angles are in degrees for some reason.

def homogeneousMatFromEuler(rotation):
    mat = np.zeros((4, 4))
    #rot_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    s = math.pi / 180.0
    mat[:3, :3] = EulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ EulerX(s * rotation[0])
    mat[3, 3] = 1
    return mat

# 3 x 4 x 4 derivative of the above with respect to the angle variables.

def dhomogeneousMatFromEuler(rotation):
    mat = np.zeros((3, 4, 4))
    #rot_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    s = math.pi / 180.0
    mat[2, :3, :3] = s * dEulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ EulerX(s * rotation[0])    
    mat[1, :3, :3] = s * EulerZ(s * rotation[2]) @ dEulerY(s * rotation[1]) @ EulerX(s * rotation[0])    
    mat[0, :3, :3] = s * EulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ dEulerX(s * rotation[0])    
    return mat

# 3 x 3 x 4 x 4 Hessian of the above.
def HhomogeneousMatFromEuler(rotation):
    mat = np.zeros((3, 3, 4, 4))
    #rot_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    s = math.pi / 180.0
    mat[2, 2, :3, :3] = s * s * HEulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ EulerX(s * rotation[0])    
    mat[2, 1, :3, :3] = s * s * dEulerZ(s * rotation[2]) @ dEulerY(s * rotation[1]) @ EulerX(s * rotation[0])
    mat[2, 0, :3, :3] = s * s * dEulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ dEulerX(s * rotation[0])
    mat[1, 2, :3, :3] = s * s * dEulerZ(s * rotation[2]) @ dEulerY(s * rotation[1]) @ EulerX(s * rotation[0])
    mat[1, 1, :3, :3] = s * s * EulerZ(s * rotation[2]) @ HEulerY(s * rotation[1]) @ EulerX(s * rotation[0])
    mat[1, 0, :3, :3] = s * s * EulerZ(s * rotation[2]) @ dEulerY(s * rotation[1]) @ dEulerX(s * rotation[0])
    mat[0, 2, :3, :3] = s * s * dEulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ dEulerX(s * rotation[0])
    mat[0, 1, :3, :3] = s * s * dEulerZ(s * rotation[2]) @ dEulerY(s * rotation[1]) @ EulerX(s * rotation[0])
    mat[0, 0, :3, :3] = s * s * EulerZ(s * rotation[2]) @ EulerY(s * rotation[1]) @ HEulerX(s * rotation[0])
    return mat
    
# Turns a vector into a 3x3 cross product matrix: crossMatrix(v) @ w = v x w.
def crossMatrix(vec):
    mat = np.matrix([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]],[-vec[1],vec[0], 0]])
    return mat     
    
# 4x4 homogeneous rotation matrix given an axis-angle vector

def homogeneousMatFromAxisAngle(axisangle):
    theta = np.linalg.norm(axisangle)
    axis = axisangle/theta
    K = crossMatrix(axis)
    rodrigues = np.eye(3)
    if theta != 0.0:
        rodrigues = rodrigues + math.sin(theta) * K + (1.0 - math.cos(theta)) * K @ K
    mat = np.zeros((4,4))
    mat[0:3, 0:3] = rodrigues
    mat[3,3] = 1.0
    return mat
    
# Derivative of the 4x4 homogeneous rotation matrix representation of an axis-angle vector **at axisangle=0**.
# (The full derivative is *much* more complicated and we don't need it.)
# Returns a 3 x 4 x 4 tensor.

def dhomogeneousMatFromAxisAngle():
    result = np.zeros((3,4,4))
    I = np.eye(3)
    for i in range(3):
        result[i,:3,:3] = crossMatrix(I[i,:])
    return result
    
# Hessian of the 4x4 homogeneous rotation matrix representation of an axis-angle vector **at axisangle=0**.
# Returns a 3 x 3 x 4 x 4 tensor.

def HhomogeneousMatFromAxisAngle():
    result = np.zeros((3,3,4,4))
    I = np.eye(3)
    for i in range(3):
        for j in range(3):
            m1 = crossMatrix(I[i,:])
            m2 = crossMatrix(I[j,:])
            result[i,j,:3,:3] = 0.5*(m1 @ m2 + m2 @ m1)
    return result    