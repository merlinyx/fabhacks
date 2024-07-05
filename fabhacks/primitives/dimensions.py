from .clip import Clip
from .edge import Edge
from .ring import Ring
from .hook import Hook
from .rod import Rod
from .surface import Surface
from .tube import Tube
from .hemisphere import Hemisphere
import numpy as np

def rod_hook_dim_constraint(p1, p2, assembly=None):
    # assuming the rod's length is not shorter than the hook's thickness
    if isinstance(p1, Rod) and isinstance(p2, Hook):
        if (not p1.is_open_for_connect) and (not p2.is_open_for_connect):
            return False
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        return p1.radius < p2.arc_radius and cd1 > p2.thickness * 2
    if isinstance(p1, Hook) and isinstance(p2, Rod):
        if (not p2.is_open_for_connect) and (not p1.is_open_for_connect):
            return False
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p2.radius < p1.arc_radius and cd2 > p1.thickness * 2
    return False

def rod_ring_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Rod) and isinstance(p2, Ring):
        return p1.is_open_for_connect and p1.radius < p2.arc_radius
    if isinstance(p1, Ring) and isinstance(p2, Rod):
        return p2.is_open_for_connect and p2.radius < p1.arc_radius
    return False

def rod_clip_dim_constraint(p1, p2, assembly=None):
    # assuming the rod's length is not shorter than the clip's length, so no critical dim check
    if isinstance(p1, Rod) and isinstance(p2, Clip):
        return 2. * p1.radius < p2.open_gap
    if isinstance(p1, Clip) and isinstance(p2, Rod):
        return 2. * p2.radius < p1.open_gap
    return False

def rod_tube_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Rod) and isinstance(p2, Tube):
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        return p1.is_open_for_connect and p1.generalized_radius() < p2.inner_radius and cd1 > p2.length
    if isinstance(p1, Tube) and isinstance(p2, Rod):
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p2.is_open_for_connect and p2.generalized_radius() < p1.inner_radius and cd2 > p1.length
    return False

# def rod_surface_dim_constraint(p1, p2, assembly=None):
#     return False

def hook_hook_dim_constraint(p1, p2, assembly=None):
    if (not p1.is_open_for_connect) and (not p2.is_open_for_connect):
        return False
    if isinstance(p1, Hook) and isinstance(p2, Hook):
        return p1.thickness < p2.arc_radius and p2.thickness < p1.arc_radius
    return False

def hook_ring_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Hook) and not p1.is_open_for_connect:
        return False
    if isinstance(p2, Hook) and not p2.is_open_for_connect:
        return False
    if (isinstance(p1, Hook) and isinstance(p2, Ring)) or (isinstance(p1, Ring) and isinstance(p2, Hook)):
        return p1.thickness < p2.arc_radius and p2.thickness < p1.arc_radius
    return False

def hook_tube_dim_constraint(p1, p2, assembly=None):
    # assuming the tube's length is not shorter than the hook's thickness
    if isinstance(p1, Hook) and isinstance(p2, Tube):
        if (not p2.is_open) and (not p1.is_open_for_connect):
            return False
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p1.arc_radius > p2.inner_radius + p2.thickness and cd2 > p1.thickness * 2
    if isinstance(p2, Hook) and isinstance(p1, Tube):
        if (not p1.is_open) and (not p2.is_open_for_connect):
            return False
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        return p2.arc_radius > p1.inner_radius + p1.thickness and cd1 > p2.thickness * 2
    return False

# def hook_edge_dim_constraint(p1, p2, assembly=None):
#     # assuming the edge's length/width is not shorter than the hook's thickness
#     if isinstance(p1, Hook) and isinstance(p2, Edge):
#         return p1.arc_radius > p2.height
#     if isinstance(p2, Hook) and isinstance(p1, Edge):
#         return p2.arc_radius > p1.height
#     return False

def ring_tube_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Ring) and isinstance(p2, Tube):
        return p1.arc_radius > p2.inner_radius + p2.thickness and p2.is_open
    if isinstance(p2, Ring) and isinstance(p1, Tube):
        return p2.arc_radius > p1.inner_radius + p1.thickness and p1.is_open
    return False

# def ring_surface_dim_constraint(p1, p2, assembly=None):
#     return False

def clip_tube_dim_constraint(p1, p2, assembly=None):
    # assuming the tube's length is not shorter than the clip's length
    if isinstance(p1, Clip) and isinstance(p2, Tube):
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p1.open_gap > p2.inner_radius + p2.thickness and cd2 > p1.widt
    if isinstance(p2, Clip) and isinstance(p1, Tube):
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        return p2.open_gap > p1.inner_radius + p1.thickness and cd1 > p2.width
    return False

def clip_edge_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Clip) and isinstance(p2, Edge):
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p1.open_gap > p2.height and cd2 > p1.width
    if isinstance(p2, Clip) and isinstance(p1, Edge):
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        return p2.open_gap > p1.height and cd1 > p2.width
    return False

def tube_tube_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Tube) and isinstance(p2, Tube):
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p1.is_open and p1.generalized_radius() < p2.inner_radius and cd1 > p2.length\
            or p2.is_open and p2.generalized_radius() < p1.inner_radius and cd2 > p1.length
    return False

def surface_surface_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Surface) and isinstance(p2, Surface):
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        if not(cd1[0] < cd2[0] and cd1[1] < cd2[1] or cd2[0] < cd1[0] and cd2[1] < cd1[1]):
            return False
        return p1.width < p2.width and p1.length < p2.length or p2.width < p1.width and p2.length < p1.length
    return False

def surface_hemisphere_dim_constraint(p1, p2, assembly=None):
    if isinstance(p1, Surface) and isinstance(p2, Hemisphere):
        cd1 = p1.critical_dim()
        if p1.id in assembly.primitive_graph.nodes:
            cd1 = assembly.primitive_graph.nodes[p1.id]["critical_dim"]
        return p1.width > np.pi * p2.radius and p1.length > np.pi * p2.radius and cd1[0] > p2.radius * 2 and cd1[1] > p2.radius
    if isinstance(p2, Surface) and isinstance(p1, Hemisphere):
        cd2 = p2.critical_dim()
        if p2.id in assembly.primitive_graph.nodes:
            cd2 = assembly.primitive_graph.nodes[p2.id]["critical_dim"]
        return p2.width > np.pi * p1.radius and p2.length > np.pi * p1.radius and cd2[0] > p1.radius * 2 and cd2[1] > p1.radius
    return False

# ===============================================================================
def get_joint_type(p1, p2, is_fixed_joint):
    if is_fixed_joint:
        return "fixed"
    if isinstance(p1, Rod) and isinstance(p2, Hook):
        return "rodhook"
    if isinstance(p1, Hook) and isinstance(p2, Hook):
        return "hookhook"
    if isinstance(p1, Rod) and isinstance(p2, Ring):
        return "revolute"
    if isinstance(p1, Hook) and isinstance(p2, Ring):
        return "revolute"
    return "unknown"
