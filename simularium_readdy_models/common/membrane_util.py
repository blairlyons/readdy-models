#!/usr/bin/env python

from sys import float_info
import math

import numpy as np

from ..common import ReaddyUtil


def add_membrane_particle_types(system, particle_radius, temperature_K, viscosity):
    """
    Add particle and topology types for membrane particles
    to the ReaDDy system.
    """
    diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
        particle_radius, viscosity, temperature_K
    )  # nm^2/s
    system.topologies.add_type("Membrane")
    system.add_topology_species("membrane#outer", diffCoeff)
    system.add_topology_species("membrane#inner", diffCoeff)
    for n in range(1, 5):
        system.add_topology_species(f"membrane#outer_edge_{n}", diffCoeff)
        system.add_topology_species(f"membrane#inner_edge_{n}", diffCoeff)
    system.add_topology_species("membrane#outer_edge_4_1", diffCoeff)
    system.add_topology_species("membrane#outer_edge_2_3", diffCoeff)
    system.add_topology_species("membrane#inner_edge_4_1", diffCoeff)
    system.add_topology_species("membrane#inner_edge_2_3", diffCoeff)
        
        
def add_weak_interaction(types1, types2, force_const, bond_length, depth, cutoff, system):
    """
    Adds a weak interaction piecewise harmonic bond to the system
        from each type in types1
        to each type in types2
        with force constant force_const
        and length bond_length [nm]
        and with depth and cutoff.
    """
    for t1 in types1:
        for t2 in types2:
            system.potentials.add_weak_interaction_piecewise_harmonic(
                t1, t2, force_const, bond_length, depth, cutoff
            )


def add_box_potential(particle_types, origin, extent, force_constant, system):
    """
    Add a box potential to keep the given particle types
    inside a box centered at origin with extent.
    """
    for particle_type in particle_types:
        system.potentials.add_box(
            particle_type=particle_type,
            force_constant=force_constant,
            origin=origin,
            extent=extent,
        )
        
        
def leaflet_types(side):
    """
    Get all the types for one leaflet side. side = "inner" or "outer"
    """
    result = [f"membrane#{side}"]
    for n in range(1, 5):
        result.append(f"membrane#{side}_edge_{n}")
    result.append(f"membrane#{side}_edge_4_1")
    result.append(f"membrane#{side}_edge_2_3")
    return result
    
    
def calculate_lattice(size, particle_radius):
    """
    Calculate the x and y lattice coordinates of the membrane
    and find the plane dimension with zero size.
    """
    plane_dim = -1
    width = [0, 0]
    ix = 0
    for dim in range(3):
        if size[dim] < float_info.epsilon:
            plane_dim = dim
            continue
        width[ix] = round(size[dim])
        ix += 1
    if plane_dim < 0 or width[1] < float_info.epsilon:
        raise Exception("The membrane size must be zero in one and only one dimension.")
    result = [
        np.arange(0, width[0], 2. * particle_radius), 
        np.arange(0, width[1], 2. * particle_radius * np.sqrt(3))
    ]
    return result, plane_dim
        
        
def calculate_box_potentials(center, size, particle_radius, box_size):
    """
    Get the origin and extent for each of the four box potentials 
    constraining the edges of the membrane.
    """
    coords, plane_dim = calculate_lattice(size, particle_radius)
    box_origins = np.zeros((4, 3))
    box_extents = np.zeros((4, 3))
    lattice_dim = 0
    for dim in range(3):
        if dim == plane_dim:
            box_origins[:, dim] = -box_size[dim]
            box_extents[:, dim] = box_size[dim]
            continue
        values = coords[lattice_dim] - 0.5 * size[dim] + center[dim]
        offset = particle_radius * (1 if lattice_dim < 1 else np.sqrt(3))
        if lattice_dim < 1:
            box_origins[0][dim] = values[0]
            box_extents[0][dim] = values[-1]
            box_origins[1][dim] = values[-1] + offset
            box_extents[1][dim] = values[-1] + offset
            box_origins[2][dim] = values[0] + offset
            box_extents[2][dim] = values[-1] + offset
            box_origins[3][dim] = values[0]
            box_extents[3][dim] = values[0]
        else:
            box_origins[0][dim] = values[0]
            box_extents[0][dim] = values[0]
            box_origins[1][dim] = values[0] + offset
            box_extents[1][dim] = values[-1] + offset
            box_origins[2][dim] = values[-1] + offset
            box_extents[2][dim] = values[-1] + offset
            box_origins[3][dim] = values[0]
            box_extents[3][dim] = values[-1]
        lattice_dim += 1
    return box_origins, box_extents


def add_membrane_constraints(system, center, size, particle_radius, box_size):
    """
    Add bond, angle, and box constraints for membrane particles to the ReaDDy system.
    """
    util = ReaddyUtil()
    inner_types = leaflet_types("inner")
    outer_types = leaflet_types("outer")
    # weak interaction between particles in the same leaflet
    add_weak_interaction(
        inner_types, 
        inner_types, 
        force_const=250., bond_length=2. * particle_radius, 
        depth=7., cutoff=2.5 * 2. * particle_radius, system=system
    )
    add_weak_interaction(
        outer_types, 
        outer_types, 
        force_const=250., bond_length=2. * particle_radius, 
        depth=7., cutoff=2.5 * 2. * particle_radius, system=system
    )
    # (very weak) bond to pass ReaDDy requirement in order to define edges
    util.add_bond(
        inner_types, 
        inner_types, 
        force_const=1e-10, bond_length=2. * particle_radius, system=system
    )
    util.add_bond(
        outer_types, 
        outer_types, 
        force_const=1e-10, bond_length=2. * particle_radius, system=system
    )
    # bonds between pairs of inner and outer particles
    util.add_bond(
        inner_types, 
        outer_types, 
        force_const=250., bond_length=2. * particle_radius, system=system
    )
    # angles between inner-outer pairs and their neighbors on the sheet
    util.add_angle(
        inner_types, 
        outer_types, 
        outer_types, 
        force_const=1000., angle=0.5 * np.pi, system=system
    )
    util.add_angle(
        inner_types, 
        inner_types, 
        outer_types, 
        force_const=1000., angle=0.5 * np.pi, system=system
    )
    # box potentials for edges TODO
    # box_origins, box_extents = calculate_box_potentials(center, size, particle_radius, box_size)
    # corner_suffixes = ["4_1", "2_3"]
    # for n in range(1, 5):
    #     box_types = [f"membrane#outer_edge_{n}", f"membrane#inner_edge_{n}"]
    #     for suffix in corner_suffixes:
    #         if f"{n}" in suffix:
    #             box_types += [f"membrane#outer_edge_{suffix}", f"membrane#inner_edge_{suffix}"]
    #             break
    #     add_box_potential(
    #         box_types, 
    #         origin=box_origins[n - 1] - particle_radius, 
    #         extent=box_extents[n - 1] + particle_radius, 
    #         force_constant=250., system=system
    #     )


def init_membrane(simulation, center, size, particle_radius, box_size):
    """
    Add initial membrane particles to the ReaDDy simulation.
    """
    coords, plane_dim = calculate_lattice(size, particle_radius)
    cols = coords[0].shape[0]
    rows = coords[1].shape[0]
    n_lattice_points = cols * (2 * rows)
    positions = np.zeros((2 * n_lattice_points, 3))
    types = np.array(2 * n_lattice_points * ["membrane#side-_init_0_0"])
    lattice_dim = 0
    for dim in range(3):
        
        if dim == plane_dim:
            positions[:n_lattice_points, dim] = center[dim] - particle_radius
            positions[n_lattice_points:, dim] = center[dim] + particle_radius
            continue
        
        values = coords[lattice_dim] - 0.5 * size[dim] + center[dim]
        offset = particle_radius * (1 if lattice_dim < 1 else np.sqrt(3))
        
        # positions and types
        for i in range(rows):
            
            p = values if lattice_dim < 1 else values[i]
            
            start_ix = 2 * i * cols
            positions[start_ix:start_ix + cols, dim] = p
            if i < 1:
                types[start_ix] = "membrane#outer_edge_4_1"
                types[start_ix + 1:start_ix + cols] = "membrane#outer_edge_1"
            else:
                types[start_ix] = "membrane#outer_edge_4"
                types[start_ix + 1:start_ix + cols] = "membrane#outer"
                
            start_ix += n_lattice_points
            positions[start_ix:start_ix + cols, dim] = p
            if i < 1:
                types[start_ix] = "membrane#inner_edge_4_1"
                types[start_ix + 1:start_ix + cols] = "membrane#inner_edge_1"
            else:
                types[start_ix] = "membrane#inner_edge_4"
                types[start_ix + 1:start_ix + cols] = "membrane#inner"
            
            start_ix = (2 * i + 1) * cols
            positions[start_ix:start_ix + cols, dim] = p + offset
            if i < rows - 1:
                types[start_ix:start_ix + cols - 1] = "membrane#outer"
                types[start_ix + cols - 1] = "membrane#outer_edge_2"
            else:
                types[start_ix:start_ix + cols - 1] = "membrane#outer_edge_3"
                types[start_ix + cols - 1] = "membrane#outer_edge_2_3"
            
            start_ix += n_lattice_points
            positions[start_ix:start_ix + cols, dim] = p + offset
            if i < rows - 1:
                types[start_ix:start_ix + cols - 1] = "membrane#inner"
                types[start_ix + cols - 1] = "membrane#inner_edge_2"
            else:
                types[start_ix:start_ix + cols - 1] = "membrane#inner_edge_3"
                types[start_ix + cols - 1] = "membrane#inner_edge_2_3"
            
        lattice_dim += 1
    
    membrane = simulation.add_topology("Membrane", types.tolist(), positions)
    
    # edges
    for n in range(n_lattice_points):
        other_n = n + n_lattice_points
        membrane.get_graph().add_edge(n, other_n)
        if n % cols != cols - 1:
            membrane.get_graph().add_edge(n, n + 1)
            membrane.get_graph().add_edge(other_n, other_n + 1)
        if math.ceil((n + 1) / cols) >= 2 * rows:
            continue
        if (n % (2 * cols) != (2 * cols) - 1):
            membrane.get_graph().add_edge(n, n + cols)
            membrane.get_graph().add_edge(other_n, other_n + cols)
        if (n % (2 * cols) != 0):
            membrane.get_graph().add_edge(n, n + cols - 1)
            membrane.get_graph().add_edge(other_n, other_n + cols - 1)
