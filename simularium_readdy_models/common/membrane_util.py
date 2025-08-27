#!/usr/bin/env python

from sys import float_info
import math
import uuid

import numpy as np

from ..common import ReaddyUtil, ParticleData


def _leaflet_types(side):
    """
    Get all the types for one leaflet side. side = "inner" or "outer"
    """
    result = [f"membrane#{side}"]
    for n in range(1, 5):
        result.append(f"membrane#{side}_edge_{n}")
    result.append(f"membrane#{side}_edge_4_1")
    result.append(f"membrane#{side}_edge_2_3")
    return result


def all_membrane_particle_types():
    """
    Get all the membrane particle types.
    """
    return _leaflet_types("outer") + _leaflet_types("inner")


def add_membrane_particle_types(system, particle_radius, temperature_K, viscosity):
    """
    Add particle and topology types for membrane particles
    to the ReaDDy system.
    """
    diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
        particle_radius, viscosity, temperature_K
    )  # nm^2/s
    system.topologies.add_type("Membrane")
    particle_types = all_membrane_particle_types()
    for t in particle_types:
        system.add_topology_species(t, diffCoeff)


def _add_weak_interaction(
    types1, types2, force_const, bond_length, depth, cutoff, system
):
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


def _add_box_potential(particle_types, origin, extent, force_constant, system):
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


def _calculate_lattice(size, particle_radius):
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
        np.arange(0, width[0], 2.0 * particle_radius),
        np.arange(0, width[1], 2.0 * particle_radius * np.sqrt(3)),
    ]
    return result, plane_dim


def _calculate_box_potentials(center, size, particle_radius, box_size):
    """
    Get the origin and extent for each of the four box potentials
    constraining the edges of the membrane.
    """
    coords, plane_dim = _calculate_lattice(size, particle_radius)
    box_origins = np.zeros((4, 3))
    box_extents = np.zeros((4, 3))
    lattice_dim = 0
    for dim in range(3):
        if dim == plane_dim:
            box_origins[:, dim] = -0.5 * box_size[dim] + 3
            box_extents[:, dim] = box_size[dim] - 6
            continue
        values = coords[lattice_dim] - 0.5 * size[dim] + center[dim]
        offset = particle_radius * (1 if lattice_dim < 1 else np.sqrt(3))
        if lattice_dim < 1:
            box_origins[0][dim] = values[0]
            box_extents[0][dim] = values[-1] - values[0]
            box_origins[1][dim] = values[-1] + offset - particle_radius
            box_extents[1][dim] = 2 * particle_radius
            box_origins[2][dim] = values[0] + offset
            box_extents[2][dim] = values[-1] - values[0]
            box_origins[3][dim] = values[0] - particle_radius
            box_extents[3][dim] = 2 * particle_radius
        else:
            box_origins[0][dim] = values[0] - particle_radius
            box_extents[0][dim] = 2 * particle_radius
            box_origins[1][dim] = values[0] + offset
            box_extents[1][dim] = values[-1] - values[0]
            box_origins[2][dim] = values[-1] + offset - particle_radius
            box_extents[2][dim] = 2 * particle_radius
            box_origins[3][dim] = values[0]
            box_extents[3][dim] = values[-1] - values[0]
        lattice_dim += 1
    return box_origins, box_extents


def add_membrane_constraints(system, center, size, particle_radius, box_size):
    """
    Add bond, angle, and box constraints for membrane particles to the ReaDDy system.
    """
    util = ReaddyUtil()
    inner_types = _leaflet_types("inner")
    outer_types = _leaflet_types("outer")
    # weak interaction between particles in the same leaflet
    _add_weak_interaction(
        inner_types,
        inner_types,
        force_const=250.0,
        bond_length=2.0 * particle_radius,
        depth=7.0,
        cutoff=2.5 * 2.0 * particle_radius,
        system=system,
    )
    _add_weak_interaction(
        outer_types,
        outer_types,
        force_const=250.0,
        bond_length=2.0 * particle_radius,
        depth=7.0,
        cutoff=2.5 * 2.0 * particle_radius,
        system=system,
    )
    # (very weak) bond to pass ReaDDy requirement in order to define edges
    util.add_bond(
        inner_types,
        inner_types,
        force_const=1e-10,
        bond_length=2.0 * particle_radius,
        system=system,
    )
    util.add_bond(
        outer_types,
        outer_types,
        force_const=1e-10,
        bond_length=2.0 * particle_radius,
        system=system,
    )
    # bonds between pairs of inner and outer particles
    util.add_bond(
        inner_types,
        outer_types,
        force_const=250.0,
        bond_length=2.0 * particle_radius,
        system=system,
    )
    # angles between inner-outer pairs and their neighbors on the sheet
    util.add_angle(
        inner_types,
        outer_types,
        outer_types,
        force_const=1000.0,
        angle=0.5 * np.pi,
        system=system,
    )
    util.add_angle(
        inner_types,
        inner_types,
        outer_types,
        force_const=1000.0,
        angle=0.5 * np.pi,
        system=system,
    )
    # box potentials for edges
    box_origins, box_extents = _calculate_box_potentials(
        center, size, particle_radius, box_size
    )
    corner_suffixes = ["4_1", "2_3"]
    for n in range(1, 5):
        box_types = [f"membrane#outer_edge_{n}", f"membrane#inner_edge_{n}"]
        for suffix in corner_suffixes:
            if f"{n}" in suffix:
                box_types += [
                    f"membrane#outer_edge_{suffix}",
                    f"membrane#inner_edge_{suffix}",
                ]
                break
        _add_box_potential(
            box_types,
            origin=box_origins[n - 1],
            extent=box_extents[n - 1],
            force_constant=250.0,
            system=system,
        )


def get_membrane_monomers(center, size, particle_radius, start_particle_id=0, top_id=0):
    """
    get all the monomer data for the membrane patch
    defined by center size and particle radius.
    """
    coords, plane_dim = _calculate_lattice(size, particle_radius)
    cols = coords[0].shape[0]
    rows = coords[1].shape[0]
    n_lattice_points = cols * (2 * rows)
    positions = np.zeros((2 * n_lattice_points, 3))
    types = np.array(2 * n_lattice_points * ["membrane#side-_init_0_0"])
    lattice_dim = 0
    for dim in range(3):

        if dim == plane_dim:
            positions[:n_lattice_points, dim] = center[dim] + particle_radius
            positions[n_lattice_points:, dim] = center[dim] - particle_radius
            continue

        values = coords[lattice_dim] - 0.5 * size[dim] + center[dim]
        offset = particle_radius * (1 if lattice_dim < 1 else np.sqrt(3))

        # positions and types
        for i in range(rows):

            p = values if lattice_dim < 1 else values[i]

            start_ix = 2 * i * cols
            positions[start_ix : start_ix + cols, dim] = p
            if i < 1:
                types[start_ix] = "membrane#outer_edge_4_1"
                types[start_ix + 1 : start_ix + cols] = "membrane#outer_edge_1"
            else:
                types[start_ix] = "membrane#outer_edge_4"
                types[start_ix + 1 : start_ix + cols] = "membrane#outer"

            start_ix += n_lattice_points
            positions[start_ix : start_ix + cols, dim] = p
            if i < 1:
                types[start_ix] = "membrane#inner_edge_4_1"
                types[start_ix + 1 : start_ix + cols] = "membrane#inner_edge_1"
            else:
                types[start_ix] = "membrane#inner_edge_4"
                types[start_ix + 1 : start_ix + cols] = "membrane#inner"

            start_ix = (2 * i + 1) * cols
            positions[start_ix : start_ix + cols, dim] = p + offset
            if i < rows - 1:
                types[start_ix : start_ix + cols - 1] = "membrane#outer"
                types[start_ix + cols - 1] = "membrane#outer_edge_2"
            else:
                types[start_ix : start_ix + cols - 1] = "membrane#outer_edge_3"
                types[start_ix + cols - 1] = "membrane#outer_edge_2_3"

            start_ix += n_lattice_points
            positions[start_ix : start_ix + cols, dim] = p + offset
            if i < rows - 1:
                types[start_ix : start_ix + cols - 1] = "membrane#inner"
                types[start_ix + cols - 1] = "membrane#inner_edge_2"
            else:
                types[start_ix : start_ix + cols - 1] = "membrane#inner_edge_3"
                types[start_ix + cols - 1] = "membrane#inner_edge_2_3"

        lattice_dim += 1

    particles = {}
    for p in range(2 * n_lattice_points):
        particles[p + start_particle_id] = ParticleData(
            unique_id=p,
            type_name=types[p],
            position=positions[p],
            neighbor_ids=[],
        )

    # edges
    for ix in range(n_lattice_points):
        p_ix = ix + start_particle_id
        other_ix = ix + n_lattice_points + start_particle_id
        particles[p_ix].neighbor_ids.append(other_ix)
        particles[other_ix].neighbor_ids.append(p_ix)
        if ix % cols != cols - 1:
            particles[p_ix].neighbor_ids.append(p_ix + 1)
            particles[p_ix + 1].neighbor_ids.append(p_ix)
            particles[other_ix].neighbor_ids.append(other_ix + 1)
            particles[other_ix + 1].neighbor_ids.append(other_ix)
        if math.ceil((ix + 1) / cols) >= 2 * rows:
            continue
        if ix % (2 * cols) != 2 * cols - 1:
            if ix % (2 * cols) < cols:
                particles[p_ix].neighbor_ids.append(p_ix + cols)
                particles[p_ix + cols].neighbor_ids.append(p_ix)
                particles[other_ix].neighbor_ids.append(other_ix + cols)
                particles[other_ix + cols].neighbor_ids.append(other_ix)
            else:
                particles[p_ix].neighbor_ids.append(p_ix + cols + 1)
                particles[p_ix + cols + 1].neighbor_ids.append(p_ix)
                particles[other_ix].neighbor_ids.append(other_ix + cols + 1)
                particles[other_ix + cols + 1].neighbor_ids.append(other_ix)
        if ix % (2 * cols) != cols - 1:
            if ix % (2 * cols) < cols - 1:
                particles[p_ix + 1].neighbor_ids.append(p_ix + cols)
                particles[p_ix + cols].neighbor_ids.append(p_ix + 1)
                particles[other_ix + 1].neighbor_ids.append(other_ix + cols)
                particles[other_ix + cols].neighbor_ids.append(other_ix + 1)
            else:
                particles[p_ix].neighbor_ids.append(p_ix + cols)
                particles[p_ix + cols].neighbor_ids.append(p_ix)
                particles[other_ix].neighbor_ids.append(other_ix + cols)
                particles[other_ix + cols].neighbor_ids.append(other_ix)

    result = {
        "topologies": {
            top_id: {
                "type_name": "Membrane",
                "particle_ids": (
                    np.arange(2 * n_lattice_points) + start_particle_id
                ).tolist(),
            }
        },
        "particles": {},
    }
    particles = {p_id: dict(particle_data) for p_id, particle_data in particles.items()}
    result["particles"] = particles
    return result
