import copy
from ..assembly.frame import Frame
from ..assembly.helpers import *
import numpy as np
from .helpers import *
from ..physics_solver.helpers import *
from enum import Enum, auto

class ParameterFlag(Enum):
    UNBOUNDED = auto() # whether it is periodic
    BOUNDED_AND_CLAMPED = auto() # whether it is bounded and needs to be clamped under physics
    BOUNDED_AND_OPEN = auto() # whether it is bounded but allows falling out of bounds

class Parameter:
    def __init__(self, pid, name, value):
        self.pid = pid
        self.name = name
        self.default_value = value
        self.value = value
        self.index = -1

    def __str__(self):
        return self.name

    def quantized_values(self):
        pass

    def random_value(self):
        pass

    def set_value(self, value):
        self.value = value

    def varname(self):
        return self.name.split("_")[1]

class FloatParameter(Parameter):
    def __init__(self, pid, name, value, min_value, max_value, nquantization=2, is_unbounded=False,\
        fixed_under_physics=False, is_critical_dim=False):
        super().__init__(pid, name, value)
        self.min_value = min_value
        self.max_value = max_value
        self.n = nquantization
        self.fixed_under_physics = fixed_under_physics # if True, the parameter will not be optimized under physics
        self.unbounded = is_unbounded
        self.is_critical_dim = is_critical_dim
        # the flag indicates how the bounds should be handled under physics
        self.flag = ParameterFlag.UNBOUNDED if is_unbounded else ParameterFlag.BOUNDED_AND_OPEN

    def __repr__(self):
        return self.name
        # return f"{self.name}, {self.pid}, {self.value}, {self.default_value}, {self.index}"

    def bounds(self):
        if self.unbounded:
            return (None, None)
        return (self.min_value, self.max_value)

    def set_value(self, value):
        if "_theta_" in self.name or "_phi_" in self.name or "_a_" in self.name or "_b_" in self.name or "_c_" in self.name:
            self.value = clamp_angle(value, self.min_value, self.max_value)
        else:
            self.value = value

    def quantized_values(self):
        if self.min_value == self.max_value:
            return [self.min_value]
        return np.linspace(self.min_value, self.max_value, self.n + 2)[1:-1]

    def random_value(self):
        return np.random.uniform(self.min_value, self.max_value)

class FixedFloatParameter(FloatParameter):
    def __init__(self, pid, name, value, fixed_under_physics=False, is_critical_dim=False):
        super().__init__(pid, name, value, value, value, 0, False, fixed_under_physics, is_critical_dim)

    def bounds(self):
        return (self.value, self.value)

    def quantized_values(self):
        return [self.value]

    def random_value(self):
        return self.value    

class MCFParameters:
    def __init__(self):
        self.params = []
    
    def add(self, p):
        self.params.append(p)

    def indices(self, start_index=0):
        return [start_index + p.index if p.index != -1 else -1 for p in self.params]

    def values(self):
        return [p.value for p in self.params]

    def slider_param_indices(self):
        return [p.index for p in self.params if isinstance(p, FloatParameter) and not isinstance(p, FixedFloatParameter)]

    def slider_param_values(self):
        return [p.value for p in self.params if isinstance(p, FloatParameter) and not isinstance(p, FixedFloatParameter)]

    def slider_parameters(self):
        return [p for p in self.params if isinstance(p, FloatParameter) and not isinstance(p, FixedFloatParameter)]

    def critical_dim_slider_parameters(self):
        return [p for p in self.params if p.is_critical_dim and isinstance(p, FloatParameter) and not isinstance(p, FixedFloatParameter)]

    def reset_slider_param_indices(self):
        for p in self.slider_parameters():
            p.index = -1

    def reset_slider_param_values(self):
        for p in self.slider_parameters():
            p.value = p.default_value

    def set_slider_param_indices(self, start_index):
        newly_set_indices = []
        for i, p in enumerate(self.slider_parameters()):
            if p.index == -1:
                p.index = start_index + i
                newly_set_indices.append(i)
        return newly_set_indices
    
    def set_slider_parameters(self, new_values):
        for i, p in enumerate(self.slider_parameters()):
            p.value = new_values[i]

# A class that knows how to create a new mate connector frame
# based on frame parameter(s) (and previous MCFs).
class MateConnectorFrame:
    def __init__(self, primitive_id):
        self.name = None
        self.id = primitive_id
        self.base_frame = None
        self.prefix_chains = {}
        # only store frame parameter(s); reconstruct frame on demand
        self.MCFParams = []

    def parametric_frame(self, *params):
        pass

    # def parametric_frame_tframe(self, *params):
    #     pass

    def get_mcf_range(self):
        min_values = []
        max_values = []
        for p in self.MCFParams[0].params:
            min_values.append(p.min_value)
            max_values.append(p.max_value)
        min_eval_mcf = np.array(self.parametric_frame(*min_values).pos)
        max_eval_mcf = np.array(self.parametric_frame(*max_values).pos)
        return (0, np.linalg.norm(max_eval_mcf - min_eval_mcf))

    def generate_mcf_samples(self, sample_size):
        param_quantized_values_list = [[]] * self.numParams()
        for i, p in enumerate(self.MCFParams[0].params):
            p.n = 16 if self.numParams() > 1 else 100 # quantization
            p.index = i
            param_quantized_values_list[i] = p.quantized_values()
        values = []
        if sample_size == None:
            values = itertools.product(*param_quantized_values_list)
        else:
            values = reservoir_sample(itertools.product(*param_quantized_values_list), sample_size)
        mcf_samples = np.array([self.parametric_frame(*v).pos for v in values]).astype(np.float32)
        return mcf_samples

    def mcf_params_values(self, i):
        return self.MCFParams[i].values()

    def eval_mcf(self, i):
        if len(self.MCFParams) == 0 or len(self.MCFParams[i].params) == 0:
            return self.base_frame
        return self.parametric_frame(*self.MCFParams[i].values())

    def eval_cf(self, i):
        if len(self.MCFParams) == 0 or len(self.MCFParams[i].params) == 0:
            return self.base_frame
        return self.connectorFrame(*self.MCFParams[i].values())

    # def eval_mcft(self, i):
    #     if len(self.MCFParams) == 0 or len(self.MCFParams[i].params) == 0:
    #         return self.base_frame
    #     return self.parametric_frame_tframe(*self.MCFParams[i].values())
        
    # Number of parameters used by the connector
    def numParams(self):
        pass
        
    # As in eval_mcf above, but stateless.
    # Returns the 4x4 frame matrix (relative to the part) given the parameter values
    def connectorFrame(self, *params):
        pass
    
    # The gradient of connectorFrame with respect to the parameters. n x 4 x 4 tensor.
    def dConnectorFrame(self, *params):
        pass
    
    # The Hessian of the connectorFrame. n x n x 4 x 4 tensor.
    def HConnectorFrame(self, *params):
        pass           

    def new_mcf(self, use_base_frame=False, ranges=None):
        return 0

    def distribute_slider_parameters(self):
        if len(self.MCFParams) > 1:
            for i, mcfparams in enumerate(self.MCFParams):
                for p in mcfparams.slider_parameters():
                    p.value = (p.max_value - p.min_value) / float(len(self.MCFParams) + 1) * float(i + 1) + p.min_value

    def set_fixed_under_physics(self, mcfid, fixed):
        if mcfid == -1:
            for mcfparams in self.MCFParams:
                for p in mcfparams.params:
                    p.fixed_under_physics = fixed
        else:
            mcfparams = self.MCFParams[mcfid]
            for p in mcfparams.params:
                p.fixed_under_physics = fixed

    def is_fixed_under_physics(self, mcfid=-1):
        hasfalse = False
        hasparam = False
        if mcfid == -1:
            for mcfparams in self.MCFParams:
                for p in mcfparams.params:
                    if not p.fixed_under_physics:
                        hasfalse = True
                    hasparam = True
        else:
            mcfparams = self.MCFParams[mcfid]
            for p in mcfparams.params:
                if not p.fixed_under_physics:
                    hasfalse = True
                hasparam = True
        return hasparam and not hasfalse

###################################################################
# Below are specific implementations of MateConnectorFrame
# for each primitive.
###################################################################

class RodMCF(MateConnectorFrame):
    def __init__(self, primitive_id, radius, length):
        super().__init__(primitive_id)
        self.radius = radius
        self.length = length
    
    def parametric_frame(self, t, phi):
        t = clamp(t)
        phi = clamp(phi, 45, 135)
        tf = Frame([0, self.radius, (t - 0.5) * self.length], [-90., 0., 0.])
        rf = Frame([0, 0, 0], [phi-90., 0., 0.])
        return self.base_frame.transform_frame(tf.transform_frame(rf))

    def numParams(self):
        return 2
        
    def connectorFrame(self, t, phi):
        tmat = homogeneousMatFromTranslation([0, self.radius, (t - 0.5) * self.length])
        rmat = homogeneousMatFromEuler([-90., 0., 0.])
        rmat2 = homogeneousMatFromEuler([phi-90., 0., 0.])
        return self.base_frame.homogeneous_mat() @ tmat @ rmat @ rmat2

    def dConnectorFrame(self, t, phi):
        tmat = homogeneousMatFromTranslation([0, self.radius, (t - 0.5) * self.length])
        rmat = homogeneousMatFromEuler([-90., 0., 0.])
        rmat2 = homogeneousMatFromEuler([phi-90., 0., 0.])
        dtmat = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        drmat2 = np.tensordot([1, 0, 0], dhomogeneousMatFromEuler([phi-90., 0., 0.]), 1)
        d = np.zeros((2,4,4))
        d[0,:,:] = dtmat @ rmat @ rmat2
        d[1,:,:] = tmat @ rmat @ drmat2
        return (self.base_frame.homogeneous_mat() @ d)

    def HConnectorFrame(self, t, phi):
        h = np.zeros((2,2,4,4))
        tmat = homogeneousMatFromTranslation([0, self.radius, (t - 0.5) * self.length])
        rmat = homogeneousMatFromEuler([-90., 0., 0.])
        # rmat2 = homogeneousMatFromEuler([0., 0., phi-90.])
        dtmat = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        drmat2 = np.tensordot([1, 0, 0], dhomogeneousMatFromEuler([phi-90., 0., 0.]), 1)
        hrmat2 = np.tensordot([1, 0, 0], np.tensordot([1, 0, 0], HhomogeneousMatFromEuler([phi-90., 0., 0.]), 1), 1)
        h[0, 1, :, :] = dtmat @ rmat @ drmat2
        h[1, 0, :, :] = dtmat @ rmat @ drmat2
        h[1, 1, :, :] = tmat @ rmat @ hrmat2
        return (self.base_frame.homogeneous_mat() @ h)

    def new_mcf(self, use_base_frame=False, ranges=None, t=0.5, phi=90.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, is_critical_dim=True))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi))
        else:
            t = clamp(t)
            phi = clamp(phi, 45, 135)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, 0., 1., is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, 45., 135.))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, ranges[0][0], ranges[0][1], is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, ranges[1][0], ranges[1][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class RodCtlMCF(MateConnectorFrame):
    def __init__(self, primitive_id, radius, length):
        super().__init__(primitive_id)
        self.radius = radius
        self.length = length

    def parametric_frame(self, t):
        t = clamp(t)
        tf = Frame([0, 0, (t - 0.5) * self.length], [-90., 0., 0.])
        return self.base_frame.transform_frame(tf)
        
    def numParams(self):
        return 1
        
    def connectorFrame(self, t):
        tmat = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat = homogeneousMatFromEuler([-90., 0., 0.])        
        return self.base_frame.homogeneous_mat() @ tmat @ rmat
        
    def dConnectorFrame(self, t):
        tmat = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        rmat = homogeneousMatFromEuler([-90., 0., 0.])        
        return (self.base_frame.homogeneous_mat() @ tmat @ rmat).reshape((1,4,4))
    
    def HConnectorFrame(self, t):
        return np.zeros((1,1,4,4))

    def new_mcf(self, use_base_frame=False, ranges=None, t=0.5):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, is_critical_dim=True))
        else:
            t = clamp(t)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, 0., 1., is_critical_dim=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, ranges[0][0], ranges[0][1], is_critical_dim=True))
        self.MCFParams.append(new_params)
        return mcf_id

class TubeMCF(MateConnectorFrame):
    def __init__(self, primitive_id, inner_radius, thickness, length):
        super().__init__(primitive_id)
        self.inner_radius = inner_radius
        self.thickness = thickness
        self.length = length

    def parametric_frame(self, t, theta):
        t = clamp(t)
        y = self.inner_radius + self.thickness
        theta = clamp_angle(theta, 0., 360.)
        trad = np.radians(theta)
        tf = Frame([0, 0, (t - 0.5) * self.length], [-90., 0, 0])
        rf = Frame([-y * np.sin(trad), 0., -y * np.cos(trad)], [0., theta, 0.])
        return self.base_frame.transform_frame(tf.transform_frame(rf))

    def numParams(self):
        return 2
        
    def connectorFrame(self, t, theta):
        y = self.inner_radius + self.thickness
        trad = np.radians(theta)
        tmat1 = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat1 = homogeneousMatFromEuler([-90., 0., 0.])
        tmat2 = homogeneousMatFromTranslation([-y * np.sin(trad), 0., -y * np.cos(trad)])
        rmat2 = homogeneousMatFromEuler([0., theta, 0.])
        return self.base_frame.homogeneous_mat() @ tmat1 @ rmat1 @ tmat2 @ rmat2

    def dConnectorFrame(self, t, theta):
        trad = np.radians(theta)
        y = self.inner_radius + self.thickness
        tmat1 = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat1 = homogeneousMatFromEuler([-90., 0., 0.])
        tmat2 = homogeneousMatFromTranslation([-y * np.sin(trad), 0., -y * np.cos(trad)])
        rmat2 = homogeneousMatFromEuler([0., theta, 0.])

        s = math.pi / 180.0
        dtmat1 = np.zeros((2,4,4))
        dtmat1[0,:,:] = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        dtmat2 = np.zeros((2,4,4))
        dtmat2[1,:,:] = np.tensordot([-y * s * math.cos(trad), 0, y * s * math.sin(trad)], dhomogeneousMatFromTranslation(), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 1, 0], dhomogeneousMatFromEuler([0., theta, 0.]), 1)
        return self.base_frame.homogeneous_mat() @ (dtmat1 @ rmat1 @ tmat2 @ rmat2 + tmat1 @ rmat1 @ dtmat2 @ rmat2 + tmat1 @ rmat1 @ tmat2 @ drmat2)
    
    def HConnectorFrame(self, t, theta):
        trad = np.radians(theta)
        y = self.inner_radius + self.thickness
        tmat1 = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat1 = homogeneousMatFromEuler([-90., 0., 0.])
        tmat2 = homogeneousMatFromTranslation([-y * np.sin(trad), 0., -y * np.cos(trad)])
        rmat2 = homogeneousMatFromEuler([0., theta, 0.])

        s = math.pi / 180.0
        dtmat1 = np.zeros((2,4,4))
        dtmat1[0,:,:] = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        dtmat2 = np.zeros((2,4,4))
        dtmat2[1,:,:] = np.tensordot([-y * s * math.cos(trad), 0, y * s * math.sin(trad)], dhomogeneousMatFromTranslation(), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 1, 0], dhomogeneousMatFromEuler([0., theta, 0.]), 1)

        htmat2 = np.zeros((2,2,4,4))
        htmat2[1,1,:,:] = np.tensordot(
            [y * s * s * math.sin(trad), 0, y * s * s * math.cos(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        hrmat2 = np.zeros((2,2,4,4))
        hrmat2[1,1,:,:] = np.tensordot([0, 1, 0], np.tensordot([0, 1, 0], HhomogeneousMatFromEuler([0., theta, 0.]), 1), 1)
        return self.base_frame.homogeneous_mat() @ (2. * dtmat1 @ rmat1 @ dtmat2 @ rmat2 + 2. * dtmat1 @ rmat1 @ tmat2 @ drmat2 + 2. * tmat1 @ rmat1 @ dtmat2 @ drmat2\
                                                    + tmat1 @ rmat1 @ htmat2 @ rmat2 + tmat1 @ rmat1 @ tmat2 @ hrmat2)

    def new_mcf(self, use_base_frame=False, ranges=None, t=0.5, theta=180.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, is_critical_dim=True))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, is_unbounded=True))
        else:
            if ranges is None:
                t = clamp(t)
                theta = clamp_angle(theta, 0., 360.)
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, 0., 1., is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
            else:
                t = (ranges[0][0] + ranges[0][1]) / 2.
                theta = (ranges[1][0] + ranges[1][1]) / 2.
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, ranges[0][0], ranges[0][1], is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[1][0], ranges[1][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class TubeCtlMCF(MateConnectorFrame):
    def __init__(self, primitive_id, inner_radius, thickness, length):
        super().__init__(primitive_id)
        self.inner_radius = inner_radius
        self.thickness = thickness
        self.length = length

    def parametric_frame(self, t, theta):
        t = clamp(t)
        theta = clamp_angle(theta, 0., 360.)
        trad = np.radians(theta)
        tf = Frame([0, 0, (t - 0.5) * self.length], [-90, 0., 0.])
        rf = Frame([-self.inner_radius * np.sin(trad), 0., -self.inner_radius * np.cos(trad)], [0., theta, 0.])
        return self.base_frame.transform_frame(tf.transform_frame(rf))

    def numParams(self):
        return 2

    def connectorFrame(self, t, theta):
        trad = np.radians(theta)
        tmat1 = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat1 = homogeneousMatFromEuler([-90., 0., 0.])
        tmat2 = homogeneousMatFromTranslation([-self.inner_radius * np.sin(trad), 0., -self.inner_radius * np.cos(trad)])
        rmat2 = homogeneousMatFromEuler([0., theta, 0.])
        return self.base_frame.homogeneous_mat() @ tmat1 @ rmat1 @ tmat2 @ rmat2

    def dConnectorFrame(self, t, theta):
        trad = np.radians(theta)
        tmat1 = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat1 = homogeneousMatFromEuler([-90., 0., 0.])
        tmat2 = homogeneousMatFromTranslation([-self.inner_radius * np.sin(trad), 0., -self.inner_radius * np.cos(trad)])
        rmat2 = homogeneousMatFromEuler([0., theta, 0.])

        s = math.pi / 180.0
        dtmat1 = np.zeros((2,4,4))
        dtmat1[0,:,:] = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        dtmat2 = np.zeros((2,4,4))
        dtmat2[1,:,:] = np.tensordot([-self.inner_radius * s * math.cos(trad), 0, self.inner_radius * s * math.sin(trad)], dhomogeneousMatFromTranslation(), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 1, 0], dhomogeneousMatFromEuler([0., theta, 0.]), 1)
        return self.base_frame.homogeneous_mat() @ (dtmat1 @ rmat1 @ tmat2 @ rmat2 + tmat1 @ rmat1 @ dtmat2 @ rmat2 + tmat1 @ rmat1 @ tmat2 @ drmat2)

    def HConnectorFrame(self, t, theta):
        trad = np.radians(theta)
        tmat1 = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.length])
        rmat1 = homogeneousMatFromEuler([-90., 0., 0.])
        tmat2 = homogeneousMatFromTranslation([-self.inner_radius * np.sin(trad), 0., -self.inner_radius * np.cos(trad)])
        rmat2 = homogeneousMatFromEuler([0., theta, 0.])

        s = math.pi / 180.0
        dtmat1 = np.zeros((2,4,4))
        dtmat1[0,:,:] = np.tensordot([0, 0, self.length], dhomogeneousMatFromTranslation(), 1)
        dtmat2 = np.zeros((2,4,4))
        dtmat2[1,:,:] = np.tensordot([-self.inner_radius * s * math.cos(trad), 0, self.inner_radius * s * math.sin(trad)], dhomogeneousMatFromTranslation(), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 1, 0], dhomogeneousMatFromEuler([0., theta, 0.]), 1)

        htmat2 = np.zeros((2,2,4,4))
        htmat2[1,1,:,:] = np.tensordot(
            [self.inner_radius * s * s * math.sin(trad), 0, self.inner_radius * s * s * math.cos(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        hrmat2 = np.zeros((2,2,4,4))
        hrmat2[1,1,:,:] = np.tensordot([0, 1, 0], np.tensordot([0, 1, 0], HhomogeneousMatFromEuler([0., theta, 0.]), 1), 1)
        return self.base_frame.homogeneous_mat() @ (2. * dtmat1 @ rmat1 @ dtmat2 @ rmat2 + 2. * dtmat1 @ rmat1 @ tmat2 @ drmat2 + 2. * tmat1 @ rmat1 @ dtmat2 @ drmat2\
                                                    + tmat1 @ rmat1 @ htmat2 @ rmat2 + tmat1 @ rmat1 @ tmat2 @ hrmat2)

    def new_mcf(self, use_base_frame=False, ranges=None, t=0.5, theta=180.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, is_critical_dim=True))
            new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
        else:
            if ranges is None:
                t = clamp(t)
                theta = clamp_angle(theta, 0., 360.)
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, 0., 1., is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
            else:
                t = (ranges[0][0] + ranges[0][1]) / 2.
                theta = (ranges[1][0] + ranges[1][1]) / 2.
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, ranges[0][0], ranges[0][1], is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[1][0], ranges[1][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class HookFlexMCF(MateConnectorFrame):
    def __init__(self, primitive_id, arc_radius, arc_angle, thickness, ranges, phi_ranges=[45, 135]):
        super().__init__(primitive_id)
        self.arc_radius = arc_radius
        self.arc_angle = arc_angle
        self.thickness = thickness
        self.ranges = ranges
        self.phi_ranges = phi_ranges

    def parametric_frame(self, theta, phi):
        theta = clamp_angle(theta, 0., self.arc_angle)
        phi = clamp_angle(phi, self.phi_ranges[0], self.phi_ranges[1])
        trad = np.radians(theta)
        r = self.arc_radius - self.thickness
        tf = Frame([0, -r * np.sin(trad), -r * np.cos(trad)], [-theta, 0., 0.])
        rf = Frame([0, 0, 0], [0, 0, phi-90.])
        return self.base_frame.transform_frame(tf.transform_frame(rf))

    def numParams(self):
        return 2
        
    def connectorFrame(self, theta, phi):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])
        rmat2 = homogeneousMatFromEuler([0, 0, phi-90.])      
        return self.base_frame.homogeneous_mat() @ tmat @ rmat @ rmat2

    def dConnectorFrame(self, theta, phi):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])                
        rmat2 = homogeneousMatFromEuler([0, 0., phi-90.])
        dtmat = np.zeros((2,4,4))
        dtmat[0,:,:] = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        drmat = np.zeros((2,4,4))
        drmat[0,:,:] = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 0, 1], dhomogeneousMatFromEuler([0, 0., phi-90.]), 1)
        return self.base_frame.homogeneous_mat() @ (tmat @ drmat @ rmat2 + tmat @ rmat @ drmat2 + dtmat @ rmat @ rmat2)
   
    def HConnectorFrame(self, theta, phi):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])
        rmat2 = homogeneousMatFromEuler([0, 0., phi-90.])
        dtmat = np.zeros((2,4,4))
        dtmat[0,:,:] = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        drmat = np.zeros((2,4,4))
        drmat[0,:,:] = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 0, 1], dhomogeneousMatFromEuler([0, 0., phi-90.]), 1)
        Htmat = np.zeros((2,2,4,4))
        Htmat[0,0,:,:] = np.tensordot(
            [0, r * s * s * math.sin(s*theta), r * s * s * math.cos(s*theta)],
            dhomogeneousMatFromTranslation(),
            1)
        Hrmat = np.zeros((2,2,4,4))
        Hrmat[0,0,:,:] = np.tensordot([-1, 0, 0], np.tensordot([-1, 0, 0], HhomogeneousMatFromEuler([-theta, 0., 0.]), 1), 1)
        Hrmat2 = np.zeros((2,2,4,4))
        Hrmat2[1,1,:,:] = np.tensordot([0, 0, 1], np.tensordot([0, 0, 1], HhomogeneousMatFromEuler([0, 0., phi-90.]), 1), 1)
        return self.base_frame.homogeneous_mat() @ (\
            Htmat @ rmat @ rmat2 + tmat @ Hrmat @ rmat2 + tmat @ rmat @ Hrmat2\
            + 2 * np.einsum('ijk,lkm->iljm', dtmat, drmat @ rmat2) + 2 * np.einsum('ijk,lkm->iljm', dtmat, rmat @ drmat)\
            + 2 * np.einsum('ijk,lkm->iljm', tmat @ drmat, drmat2))

    def new_mcf(self, use_base_frame=False, ranges=None, theta=None, phi=None):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        theta = self.arc_angle / 2. if theta is None else clamp_angle(theta, 0., self.arc_angle)
        phi = 90. if phi is None else clamp_angle(phi, self.phi_ranges[0], self.phi_ranges[1])
        if use_base_frame:
            # if ranges is None:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta))
            # else:
            #     new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi))
        else:
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, self.ranges[0], self.ranges[1], is_unbounded=(self.arc_angle == 360.)))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, self.phi_ranges[0], self.phi_ranges[1]))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, ranges[1][0], ranges[1][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class HookMCF(MateConnectorFrame):
    def __init__(self, primitive_id, arc_radius, arc_angle, thickness, ranges):
        super().__init__(primitive_id)
        self.arc_radius = arc_radius
        self.arc_angle = arc_angle
        self.thickness = thickness
        self.ranges = ranges
    
    def parametric_frame(self, theta):
        theta = clamp_angle(theta, 0., self.arc_angle)
        trad = np.radians(theta)
        r = self.arc_radius - self.thickness
        tf = Frame([0, -r * np.sin(trad), -r * np.cos(trad)], [-theta, 0., 0.])
        return self.base_frame.transform_frame(tf)
        
    def numParams(self):
        return 1
        
    def connectorFrame(self, theta):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])        
        return self.base_frame.homogeneous_mat() @ tmat @ rmat
        
    def dConnectorFrame(self, theta):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])                
        tmatderiv = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        rmatderiv = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        return (self.base_frame.homogeneous_mat() @ (tmat @ rmatderiv + tmatderiv @ rmat)).reshape((1,4,4))
    
    def HConnectorFrame(self, theta):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])                
        tmatderiv = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        rmatderiv = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        tmathess = np.tensordot(
            [0, r * s * s * math.sin(s*theta), r * s * s * math.cos(s*theta)],
            dhomogeneousMatFromTranslation(),
            1)
        rmathess = np.tensordot([-1, 0, 0], np.tensordot([-1, 0, 0], HhomogeneousMatFromEuler([-theta, 0., 0.]), 1), 1)
        return (self.base_frame.homogeneous_mat() @ (tmathess @ rmat + 2.0 * tmatderiv @ rmatderiv + tmat @ rmathess)).reshape((1,4,4))
    
    def new_mcf(self, use_base_frame=False, ranges=None, theta=None):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        theta = self.arc_angle / 2. if theta is None else clamp_angle(theta, 0., self.arc_angle)
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta))
        else:
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, self.ranges[0], self.ranges[1], is_unbounded=(self.arc_angle == 360.)))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class HoleMCF(MateConnectorFrame):
    def __init__(self, primitive_id, arc_radius, thickness):
        super().__init__(primitive_id)
        self.arc_radius = arc_radius
        self.thickness = thickness

    def parametric_frame(self, theta):
        theta = clamp_angle(theta, 0., 360.)
        trad = np.radians(theta)
        r = self.arc_radius - self.thickness
        tf = Frame([0, -r * np.sin(trad), -r * np.cos(trad)], [-theta, 0., 0.])
        return self.base_frame.transform_frame(tf)
        
    def numParams(self):
        return 1
        
    def connectorFrame(self, theta):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])        
        return self.base_frame.homogeneous_mat() @ tmat @ rmat
        
    def dConnectorFrame(self, theta):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])                
        tmatderiv = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        rmatderiv = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        return (self.base_frame.homogeneous_mat() @ (tmat @ rmatderiv + tmatderiv @ rmat)).reshape((1,4,4))
    
    def HConnectorFrame(self, theta):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])
        tmatderiv = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        rmatderiv = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        tmathess = np.tensordot(
            [0, r * s * s * math.sin(s*theta), r * s * s * math.cos(s*theta)],
            dhomogeneousMatFromTranslation(),
            1)
        rmathess = np.tensordot([-1, 0, 0], np.tensordot([-1, 0, 0], HhomogeneousMatFromEuler([-theta, 0., 0.]), 1), 1)
        return (self.base_frame.homogeneous_mat() @ (tmathess @ rmat + 2.0 * tmatderiv @ rmatderiv + tmat @ rmathess)).reshape((1,4,4))

    def new_mcf(self, use_base_frame=False, ranges=None, theta=180.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta))
        else:
            theta = clamp_angle(theta, 0., 360.)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class HoleFlexMCF(MateConnectorFrame):
    def __init__(self, primitive_id, arc_radius, thickness):
        super().__init__(primitive_id)
        self.arc_radius = arc_radius
        self.thickness = thickness

    def parametric_frame(self, theta, phi):
        theta = clamp_angle(theta, 0., 360.)
        phi = clamp_angle(phi, 30., 150.)
        trad = np.radians(theta)
        r = self.arc_radius - self.thickness
        tf = Frame([0, -r * np.sin(trad), -r * np.cos(trad)], [-theta, 0., 0.])
        rf = Frame([0, 0, 0], [0, 0, phi-90.])
        return self.base_frame.transform_frame(tf.transform_frame(rf))

    def numParams(self):
        return 2
        
    def connectorFrame(self, theta, phi):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])
        rmat2 = homogeneousMatFromEuler([0, 0., phi-90.])      
        return self.base_frame.homogeneous_mat() @ tmat @ rmat @ rmat2
        
    def dConnectorFrame(self, theta, phi):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])                
        rmat2 = homogeneousMatFromEuler([0, 0., phi-90.])
        dtmat = np.zeros((2,4,4))
        dtmat[0,:,:] = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        drmat = np.zeros((2,4,4))
        drmat[0,:,:] = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 0, 1], dhomogeneousMatFromEuler([0, 0., phi-90.]), 1)
        return self.base_frame.homogeneous_mat() @ (tmat @ drmat @ rmat2 + tmat @ rmat @ drmat2 + dtmat @ rmat @ rmat2)

    def HConnectorFrame(self, theta, phi):
        s = math.pi / 180.0
        r = self.arc_radius - self.thickness       
        tmat = homogeneousMatFromTranslation([0, -r * math.sin(s * theta), -r * math.cos(s * theta)])
        rmat = homogeneousMatFromEuler([-theta, 0., 0.])
        rmat2 = homogeneousMatFromEuler([0, 0., phi-90.])
        dtmat = np.zeros((2,4,4))
        dtmat[0,:,:] = np.tensordot([0, -r * s * math.cos(s*theta), r * s * math.sin(s*theta)], dhomogeneousMatFromTranslation(), 1)
        drmat = np.zeros((2,4,4))
        drmat[0,:,:] = np.tensordot([-1, 0, 0], dhomogeneousMatFromEuler([-theta, 0., 0.]), 1)
        drmat2 = np.zeros((2,4,4))
        drmat2[1,:,:] = np.tensordot([0, 0, 1], dhomogeneousMatFromEuler([0, 0., phi-90.]), 1)
        Htmat = np.zeros((2,2,4,4))
        Htmat[0,0,:,:] = np.tensordot(
            [0, r * s * s * math.sin(s*theta), r * s * s * math.cos(s*theta)],
            dhomogeneousMatFromTranslation(),
            1)
        Hrmat = np.zeros((2,2,4,4))
        Hrmat[0,0,:,:] = np.tensordot([-1, 0, 0], np.tensordot([-1, 0, 0], HhomogeneousMatFromEuler([-theta, 0., 0.]), 1), 1)
        Hrmat2 = np.zeros((2,2,4,4))
        Hrmat2[1,1,:,:] = np.tensordot([0, 0, 1], np.tensordot([0, 0, 1], HhomogeneousMatFromEuler([0, 0., phi-90.]), 1), 1)
        return self.base_frame.homogeneous_mat() @ (\
            Htmat @ rmat @ rmat2 + tmat @ Hrmat @ rmat2 + tmat @ rmat @ Hrmat2\
            + 2 * np.einsum('ijk,lkm->iljm', dtmat, drmat @ rmat2) + 2 * np.einsum('ijk,lkm->iljm', dtmat, rmat @ drmat)\
            + 2 * np.einsum('ijk,lkm->iljm', tmat @ drmat, drmat2))

    def new_mcf(self, use_base_frame=False, ranges=None, theta=180., phi=None):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        theta = clamp_angle(theta, 0., 360.)
        phi = 90. if phi is None else clamp_angle(phi, 30., 150.)
        if use_base_frame:
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi))
        else:
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, 30., 150.))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, ranges[1][0], ranges[1][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class HoleCtlMCF(MateConnectorFrame):
    def __init__(self, primitive_id, arc_radius, thickness):
        super().__init__(primitive_id)
        self.arc_radius = arc_radius
        self.thickness = thickness

    def parametric_frame(self, theta, phi):
        theta = clamp_angle(theta, 0., 360.)
        phi = clamp_angle(phi, 0., 360.)
        tf = Frame([0, 0, 0], [phi, theta, 0])
        return self.base_frame.transform_frame(tf)

    def numParams(self):
        return 2
        
    def connectorFrame(self, theta, phi):
        rmat = homogeneousMatFromEuler([phi, theta, 0.])        
        return self.base_frame.homogeneous_mat() @ rmat
        
    def dConnectorFrame(self, theta, phi):
        rmat = np.zeros((2,4,4))
        rmat[0,:,:] = np.tensordot([0, 1, 0], dhomogeneousMatFromEuler([phi, theta, 0.]), 1)
        rmat[1,:,:] = np.tensordot([1, 0, 0], dhomogeneousMatFromEuler([phi, theta, 0.]), 1)
        return (self.base_frame.homogeneous_mat() @ rmat)
    
    def HConnectorFrame(self, theta, phi):
        rmat = np.zeros((2,2,4,4))
        rmat[0, 0, :, :] = np.tensordot([0, 1, 0], np.tensordot([0, 1, 0], HhomogeneousMatFromEuler([phi, theta, 0.]), 1), 1)
        rmat[0, 1, :, :] = np.tensordot([0, 1, 0], np.tensordot([1, 0, 0], HhomogeneousMatFromEuler([phi, theta, 0.]), 1), 1)
        rmat[1, 0, :, :] = np.tensordot([1, 0, 0], np.tensordot([0, 1, 0], HhomogeneousMatFromEuler([phi, theta, 0.]), 1), 1)
        rmat[1, 1, :, :] = np.tensordot([1, 0, 0], np.tensordot([1, 0, 0], HhomogeneousMatFromEuler([phi, theta, 0.]), 1), 1)
        return (self.base_frame.homogeneous_mat() @ rmat)

    def new_mcf(self, use_base_frame=False, ranges=None, theta=180., phi=0.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi))
        else:
            phi = clamp_angle(phi, 0., 360.)
            theta = clamp_angle(theta, 0., 360.)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, 0., 360., is_unbounded=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 0., 360., is_unbounded=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, ranges[0][0], ranges[0][1]))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[1][0], ranges[1][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class ClipMCF(MateConnectorFrame):
    def __init__(self, primitive_id, height):
        super().__init__(primitive_id)
        self.height = height
    
    def parametric_frame(self, t):
        t = clamp(t)
        tf = Frame([0, 0, (t - 0.5) * self.height], [0., 0., 0.])
        return self.base_frame.transform_frame(tf)
        
    def numParams(self):
        return 1
        
    def connectorFrame(self, t):
        tmat = homogeneousMatFromTranslation([0, 0, (t - 0.5) * self.height])
        return self.base_frame.homogeneous_mat() @ tmat
        
    def dConnectorFrame(self, t):
        tmat = np.tensordot([0, 0, self.height], dhomogeneousMatFromTranslation(), 1)
        return (self.base_frame.homogeneous_mat() @ tmat).reshape((1,4,4))
    
    def HConnectorFrame(self, t):
        return np.zeros((1,1,4,4))

    def new_mcf(self, use_base_frame=False, ranges=None, t=0.5):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, fixed_under_physics=True))
        else:
            t = clamp(t)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, 0., 1., fixed_under_physics=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, ranges[0][0], ranges[0][1], fixed_under_physics=True))
        self.MCFParams.append(new_params)
        return mcf_id

class EdgeMCF(MateConnectorFrame):
    def __init__(self, primitive_id, width, length, height):
        super().__init__(primitive_id)
        self.width = width
        self.length = length
        self.height = height
    
    def parametric_frame(self, t):
        t = clamp(t)
        f = copy.deepcopy(self.base_frame)
        f.pos += 0.5 * self.width * f.axis(1)
        f.pos += (t - 0.5) * self.length * f.axis(0)
        return f
        
    def numParams(self):
        return 1
        
    def connectorFrame(self, t):               
        tmat = homogeneousMatFromTranslation([(t-0.5) * self.length, 0.5 * self.width, 0])
        return self.base_frame.homogeneous_mat() @ tmat
        
    def dConnectorFrame(self, t):
        tmat = self.length * np.tensordot([1,0,0], dhomogeneousMatFromTranslation(), 1)
        return (self.base_frame.homogeneous_mat() @ tmat).reshape((1,4,4))
    
    def HConnectorFrame(self, t):
        return np.zeros((1,1,4,4))

    def new_mcf(self, use_base_frame=False, ranges=None, t=0.5):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, is_critical_dim=True))
        else:
            t = clamp(t)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, 0., 1., is_critical_dim=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t_{mcf_id}", t, ranges[0][0], ranges[0][1], is_critical_dim=True))
        self.MCFParams.append(new_params)
        return mcf_id

class SurfaceMCF(MateConnectorFrame):
    def __init__(self, primitive_id, width, length):
        super().__init__(primitive_id)
        self.width = width
        self.length = length
        self.height = 0.1

    def parametric_frame(self, t1, t2, spin):
        t1 = clamp(t1)
        t2 = clamp(t2)
        spin = clamp_angle(spin, 0., 360.)
        tf = Frame([(t2 - 0.5) * self.length, (0.5 - t1) * self.width, 0.5 * self.height])
        rf = Frame([0, 0, 0], [0, 0, spin])
        return self.base_frame.transform_frame(tf.transform_frame(rf))

    # def parametric_frame_tframe(self, t1, t2, spin):
    #     t1 = clamp(t1)
    #     t2 = clamp(t2)
    #     spin = clamp_angle(spin, 0., 360.)
    #     tf = Frame([(t2 - 0.5) * self.length, (0.5 - t1) * self.width, 0.5 * self.height])
    #     rf = Frame([0, 0, 0], [0, 0, spin])
    #     return tf.transform_frame(rf)

    def numParams(self):
        return 3

    def connectorFrame(self, t1, t2, spin):
        xlate = [(t2 - 0.5) * self.length, (0.5 - t1) * self.width, 0.5 * self.height]
        tmat = homogeneousMatFromTranslation(xlate)
        rmat = homogeneousMatFromEuler([0., 0., spin])
        return self.base_frame.homogeneous_mat() @ tmat @ rmat

    def dConnectorFrame(self, t1, t2, spin):
        xlate = [(t2 - 0.5) * self.length, (0.5 - t1) * self.width, 0.5 * self.height]
        tmat = homogeneousMatFromTranslation(xlate)
        rmat = homogeneousMatFromEuler([0., 0., spin])

        nparams = self.numParams()
        dtmat = np.zeros((nparams,4,4))
        dtmat[0,:,:] = self.width * np.tensordot([0,-1,0], dhomogeneousMatFromTranslation(), 1)
        dtmat[1,:,:] = self.length * np.tensordot([1,0,0], dhomogeneousMatFromTranslation(), 1)
        drmat = np.zeros((nparams,4,4))
        drmat[2:nparams,:,:] = np.tensordot([0, 0, 1], dhomogeneousMatFromEuler([0., 0., spin]), 1)
        return self.base_frame.homogeneous_mat() @ (dtmat @ rmat + tmat @ drmat)
    
    def HConnectorFrame(self, t1, t2, spin):
        xlate = [(t2 - 0.5) * self.length, (0.5 - t1) * self.width, 0.5 * self.height]
        tmat = homogeneousMatFromTranslation(xlate)
        rmat = homogeneousMatFromEuler([0., 0., spin])

        nparams = self.numParams()
        dtmat = np.zeros((nparams,4,4))
        dtmat[0,:,:] = self.width * np.tensordot([0,-1,0], dhomogeneousMatFromTranslation(), 1)
        dtmat[1,:,:] = self.length * np.tensordot([1,0,0], dhomogeneousMatFromTranslation(), 1)
        drmat = np.zeros((nparams,4,4))
        drmat[2:nparams,:,:] = np.tensordot([0, 0, 1], dhomogeneousMatFromEuler([0., 0., spin]), 1)
        Htmat = np.zeros((nparams,nparams,4,4))
        Hrmat = np.zeros((nparams,nparams,4,4))
        Hrmat[2:nparams,2:nparams,:,:] = np.tensordot([0, 0, 1], np.tensordot([0, 0, 1], HhomogeneousMatFromEuler([0., 0., spin]), 1), 1)
        return self.base_frame.homogeneous_mat() @ (Htmat @ rmat + 2.0 * dtmat @ drmat + tmat @ Hrmat)

    def new_mcf(self, use_base_frame=False, ranges=None, t1=0.5, t2=0.5, spin=0.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t1_{mcf_id}", t1, is_critical_dim=True))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_t2_{mcf_id}", t2, is_critical_dim=True))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_s_{mcf_id}", spin))
        else:
            t1 = clamp(t1)
            t2 = clamp(t2)
            spin = clamp_angle(spin, 0., 360.)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t1_{mcf_id}", t1, 0., 1., is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t2_{mcf_id}", t2, 0., 1., is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_s_{mcf_id}", spin, 0., 360., is_unbounded=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t1_{mcf_id}", t1, ranges[0][0], ranges[0][1], is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_t2_{mcf_id}", t2, ranges[1][0], ranges[1][1], is_critical_dim=True))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_s_{mcf_id}", spin, ranges[2][0], ranges[2][1]))
        self.MCFParams.append(new_params)
        return mcf_id

class HemisphereMCF(MateConnectorFrame):
    def __init__(self, primitive_id, radius):
        super().__init__(primitive_id)
        self.radius = radius

    def parametric_frame(self, theta, phi, spin):
        theta = clamp_angle(theta, 90., 270.)
        phi = clamp_angle(phi, 90., 270.)
        spin = clamp_angle(spin, 0., 360.)
        prad = np.radians(phi)
        trad = np.radians(theta)
        tf = Frame([self.radius*math.sin(prad), self.radius*math.cos(prad)*math.sin(trad), -self.radius*math.cos(prad)*math.cos(trad)], [0, 0, 0])
        rf = Frame([0, 0, 0], [-theta, phi, 0])
        rf2 = Frame([0, 0, 0], [0, 0, spin])
        return self.base_frame.transform_frame(tf.transform_frame(rf.transform_frame(rf2)))

    def numParams(self):
        return 3
        
    def connectorFrame(self, theta, phi, spin):
        prad = np.radians(phi)
        trad = np.radians(theta)
        xlate = [self.radius*math.sin(prad), self.radius*math.cos(prad)*math.sin(trad), -self.radius*math.cos(prad)*math.cos(trad)]
        tmat = homogeneousMatFromTranslation(xlate)
        rmat1 = homogeneousMatFromEuler([-theta, phi, 0.])
        rmat2 = homogeneousMatFromEuler([0, 0, spin])
        return self.base_frame.homogeneous_mat() @ tmat @ rmat1 @ rmat2
        
    def dConnectorFrame(self, theta, phi, spin):
        prad = np.radians(phi)
        trad = np.radians(theta)
        xlate = [self.radius*math.sin(prad), self.radius*math.cos(prad)*math.sin(trad), -self.radius*math.cos(prad)*math.cos(trad)]
        tmat = homogeneousMatFromTranslation(xlate)
        rmat1 = homogeneousMatFromEuler([-theta, phi, 0.])
        rmat2 = homogeneousMatFromEuler([0, 0, spin])

        nparams = self.numParams()
        s = math.pi / 180.0
        dtmat = np.zeros((nparams,4,4))
        dtmat[0,:,:] = np.tensordot(
            [0, s * self.radius * math.cos(prad) * math.cos(trad), s * self.radius * math.cos(prad) * math.sin(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        dtmat[1,:,:] = np.tensordot(
            [s * self.radius * math.cos(prad), -s * self.radius * math.sin(prad) * math.sin(trad), s * self.radius * math.sin(prad) * math.cos(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        drmat1 = np.zeros((nparams,4,4))
        drmat1[0,:,:] = np.tensordot([-1,0,0], dhomogeneousMatFromEuler([-theta, phi, 0.]), 1)
        drmat1[1,:,:] = np.tensordot([0,1,0], dhomogeneousMatFromEuler([-theta, phi, 0.]), 1)
        drmat2 = np.zeros((nparams,4,4))
        drmat2[2:nparams,:,:] = np.tensordot([0,0,1], dhomogeneousMatFromEuler([0, 0, spin]), 1)
        return self.base_frame.homogeneous_mat() @ ((dtmat @ rmat1 @ rmat2) + (tmat @ drmat1 @ rmat2) + (tmat @ rmat1 @ drmat2))
    
    def HConnectorFrame(self, theta, phi, spin):
        prad = np.radians(phi)
        trad = np.radians(theta)
        xlate = [self.radius*math.sin(prad), self.radius*math.cos(prad)*math.sin(trad), -self.radius*math.cos(prad)*math.cos(trad)]
        tmat = homogeneousMatFromTranslation(xlate)
        rmat1 = homogeneousMatFromEuler([-theta, phi, 0.])
        rmat2 = homogeneousMatFromEuler([0, 0, spin])

        nparams = self.numParams()
        s = math.pi / 180.0
        dtmat = np.zeros((nparams,4,4))
        dtmat[0,:,:] = np.tensordot(
            [0, s * self.radius * math.cos(prad) * math.cos(trad), s * self.radius * math.cos(prad) * math.sin(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        dtmat[1,:,:] = np.tensordot(
            [s * self.radius * math.cos(prad), -s * self.radius * math.sin(prad) * math.sin(trad), s * self.radius * math.sin(prad) * math.cos(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        drmat1 = np.zeros((nparams,4,4))
        drmat1[0,:,:] = np.tensordot([-1,0,0],  dhomogeneousMatFromEuler([-theta, phi, 0.]), 1)
        drmat1[1,:,:] = np.tensordot([0,1,0], dhomogeneousMatFromEuler([-theta, phi, 0.]), 1)
        drmat2 = np.zeros((nparams,4,4))
        drmat2[2:nparams,:,:] = np.tensordot([0,0,1], dhomogeneousMatFromEuler([0, 0, spin]), 1)

        Htmat = np.zeros((nparams,nparams,4,4))
        Htmat[0,0,:,:] = np.tensordot(
            [0, -s * s * self.radius * self.radius * math.cos(prad) * math.sin(trad), s * s * self.radius * self.radius * math.cos(prad) * math.cos(trad)],
            dhomogeneousMatFromTranslation(), 
            1)
        Htmat[0,1,:,:] = np.tensordot(
            [0, -s * s * self.radius * self.radius * math.sin(prad) * math.cos(trad), -s * s * self.radius * self.radius * math.sin(prad) * math.sin(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        Htmat[1,0,:,:] = np.tensordot(
            [0, -s * s * self.radius * self.radius * math.sin(prad) * math.cos(trad), -s * s * self.radius * self.radius * math.sin(prad) * math.sin(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        Htmat[1,1,:,:] = np.tensordot(
            [-s * s * self.radius * self.radius * math.sin(prad), -s * s * self.radius * self.radius * math.cos(prad) * math.sin(trad), s * s * self.radius * self.radius * math.cos(prad) * math.cos(trad)],
            dhomogeneousMatFromTranslation(),
            1)
        Hrmat1 = np.zeros((nparams,nparams,4,4))
        Hrmat1[0,0,:,:] = np.tensordot([-1,0,0], np.tensordot([-1,0,0], HhomogeneousMatFromEuler([-theta, phi, 0.]), 1), 1)
        Hrmat1[0,1,:,:] = np.tensordot([0,1,0], np.tensordot([-1,0,0], HhomogeneousMatFromEuler([-theta, phi, 0.]), 1), 1)
        Hrmat1[1,0,:,:] = np.tensordot([-1,0,0], np.tensordot([0,1,0], HhomogeneousMatFromEuler([-theta, phi, 0.]), 1), 1)
        Hrmat1[1,1,:,:] = np.tensordot([0,1,0], np.tensordot([0,1,0], HhomogeneousMatFromEuler([-theta, phi, 0.]), 1), 1)
        
        Hrmat2 = np.zeros((nparams,nparams,4,4))
        Hrmat2[2:nparams,2:nparams,:,:] = np.tensordot([0,0,1], np.tensordot([0,0,1], HhomogeneousMatFromEuler([0, 0, spin]), 1), 1)
        return self.base_frame.homogeneous_mat() @ (\
            (Htmat @ rmat1 @ rmat2) + (tmat @ Hrmat1 @ rmat2) + (tmat @ rmat1 @ Hrmat2)\
            + 2 * np.einsum('ijk,lkm->iljm', dtmat, drmat1 @ rmat2) + 2 * np.einsum('ijk,lkm->iljm', dtmat, rmat1 @ drmat1)\
            + 2 * np.einsum('ijk,lkm->iljm', tmat @ drmat1, drmat2))

    def new_mcf(self, use_base_frame=False, ranges=None, theta=180., phi=180., spin=0.):
        mcf_id = len(self.MCFParams)
        new_params = MCFParameters()
        if use_base_frame:
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi))
            new_params.add(FixedFloatParameter(self.id, f"{self.name}_{self.id}_s_{mcf_id}", spin))
        else:
            theta = clamp_angle(theta, 90., 270.)
            phi = clamp_angle(phi, 90., 270.)
            spin = clamp_angle(spin, 0., 360.)
            if ranges is None:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, 90., 270.))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, 90., 270.))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_s_{mcf_id}", spin, 0., 360., is_unbounded=True))
            else:
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_theta_{mcf_id}", theta, ranges[0][0], ranges[0][1]))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_phi_{mcf_id}", phi, ranges[1][0], ranges[1][1]))
                new_params.add(FloatParameter(self.id, f"{self.name}_{self.id}_s_{mcf_id}", spin, ranges[2][0], ranges[2][1]))
        self.MCFParams.append(new_params)
        return mcf_id
