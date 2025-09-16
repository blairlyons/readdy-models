#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import numpy as np

from simularium_readdy_models.microtubules import (
    MicrotubulesSimulation
)
from simularium_readdy_models import ReaddyUtil
from ..conftest import monomer_state_to_str


default_microtubule_parameters = {
    "name": "",
    "total_steps": 10,
    "time_step": 0.1,
    "box_size": [300.0, 300.0, 300.0],  # nm
    "temperature_C": 37.0,  # from Pollard experiments
    "viscosity": 8.1,  # cP, viscosity in cytoplasm
    "force_constant": 75.0,
    "grow_reaction_distance": 1.0,  # nm
    "attach_reaction_distance": 1.0,  # nm
    "n_cpu": 4,
    "tubulin_radius": 2.,
    "protofilament_growth_GTP_rate": 0.,  # 1/ns
    "protofilament_growth_GDP_rate": 0.,  # 1/ns
    "protofilament_shrink_GTP_rate": 0.,  # 1/ns
    "protofilament_shrink_GDP_rate": 0.,  # 1/ns
    "ring_attach_GTP_rate": 0.,  # 1/ns
    "ring_attach_GDP_rate": 0.,  # 1/ns
    "ring_detach_GTP_rate": 0.,  # 1/ns
    "ring_detach_GDP_rate": 0.,  # 1/ns
    "hydrolyze_rate": 0.,  # 1/ns
    "verbose": True,
}


def tubulin_is_bent(ring_ix, filament_ix, ring_connections):
    """
    Is the tubulin in bent state?
    Bent means it is missing both ring edges
    """
    return (
        True not in ring_connections[ring_ix + 1, filament_ix:filament_ix + 2]
    )


def site_can_attach(ring_ix, filament_ix, site_ix, ring_connections):
    """
    Can the site do an attach rxn?
    True if the ring edge is not attached yet
    """
    if site_ix == 1:
        return not ring_connections[ring_ix + 1, filament_ix + 1]
    if site_ix == 2:
        return not ring_connections[ring_ix + 1, filament_ix]
    return False


def tubulin_has_sites(ring_ix, filament_ix, ring_connections):
    """
    Does the tubulin need to have sites?
    Based on whether it and its neighbors toward + and - ends
    are fully crosslinked to their lateral neighbors
    """
    return (
        False in ring_connections[ring_ix:ring_ix + 3, filament_ix:filament_ix + 2]
    )


def add_edge(pid1, pid2, edges):
    """
    Add particle ids for the edge
    
    ReaDDy seems to sometimes not add edges or to duplicate them:
    - if the first int is less than the second: 
        add (and duplicate if already added)
    - if the first int is greater than the second: 
        ignore
    """
    # edges.append((pid1, pid2))
    edge = (pid1, pid2)
    if pid1 > pid2:
        edge = (pid2, pid1)
    if edge not in edges:
        edges.append(edge)


def add_init_microtubule_state(readdy_simulation, ring_connections, topology_type, new_edge_tub_ix):
    """
    Create patch of tubulins with given types and connections
    """
    particle_dims = ring_connections.shape 
    n_rings = particle_dims[0] - 2  # since there's two less rings than filament edges
    n_filaments = particle_dims[1] - 1  # since there's one less filament than ring edges
    n_tubulin = n_rings * n_filaments
    types = []
    site_types = []
    positions = []
    site_positions = []
    edges = []
    d_ring = np.array([-4., 0., 0.])
    d_filament = np.array([0., 0., 4.])
    out_string = "\n"
    new_edge_site_ix = [0, 0]
    for filament_ix in range(n_filaments):
        for ring_ix in range(n_rings):
            # each tubulin
            tub_type = "A" if ring_ix % 2 == 0 else "B"
            bent_type = "bent_" if tubulin_is_bent(ring_ix, filament_ix, ring_connections) else ""
            number1 = ring_ix % 3 + 1
            number2 = (filament_ix + math.floor(ring_ix / 3)) % 3 + 1
            has_sites = tubulin_has_sites(ring_ix, filament_ix, ring_connections)
            out_string += f"tub [{ring_ix}, {filament_ix}] : {tub_type}, {number1}_{number2}, sites = {has_sites}\n"
            types.append(f"tubulin{tub_type}#GTP_{bent_type}{number1}_{number2}")
            positions.append(ring_ix * d_ring + filament_ix * d_filament)
            tubulin_id = filament_ix * n_rings + ring_ix
            # bond to minus end tubulin
            if ring_ix > 0:
                minus_id = filament_ix * n_rings + ring_ix - 1
                add_edge(tubulin_id, minus_id, edges)
            # bond to plus end tubulin
            if ring_ix < n_rings - 1:
                plus_id = filament_ix * n_rings + ring_ix + 1
                add_edge(tubulin_id, plus_id, edges)
            # bond to prev ring tubulin
            if filament_ix > 0 and ring_connections[ring_ix + 1, filament_ix]:
                prev_id = (filament_ix - 1) * n_rings + ring_ix
                add_edge(tubulin_id, prev_id, edges)
            # bond to next ring tubulin
            if filament_ix < n_filaments - 1 and ring_connections[ring_ix + 1, filament_ix + 1]:
                next_id = (filament_ix + 1) * n_rings + ring_ix
                add_edge(tubulin_id, next_id, edges)
            # sites
            if has_sites:
                print(f"Adding sites for tubulin {tubulin_id}")
                tublin_position = positions[len(positions) - 1]
                site_out_id = n_tubulin + len(site_types)
                for site_ix in range(5):
                    site_id = site_out_id + site_ix
                    add_edge(site_id, tubulin_id, edges) # edge to tubulin
                    if site_ix == 0:
                        d_site = np.array([0., 2., 0.])
                        site_types.append("site#out")
                    else:
                        d_site = 0.5 * (d_ring if site_ix > 2 else d_filament)
                        d_site *= -1 if site_ix % 2 == 1 else 1
                        site_type = f"site#{site_ix}"
                        add_edge(site_id, site_out_id, edges) # edge to site#out
                        if site_ix > 2:  # diagonal edges between sites
                            add_edge(site_id, site_id - (site_ix - 2), edges)
                            add_edge(site_id, site_id - (site_ix - 1), edges)
                        is_new_edge_site = False
                        if new_edge_tub_ix is not None:
                            if tubulin_id == new_edge_tub_ix[0] and site_ix == 2:
                                new_edge_site_ix[1] = site_id
                                is_new_edge_site = True
                            if tubulin_id == new_edge_tub_ix[1] and site_ix == 1:
                                new_edge_site_ix[0] = site_id
                                is_new_edge_site = True
                        if not is_new_edge_site and site_can_attach(ring_ix, filament_ix, site_ix, ring_connections):
                            site_type += "_GTP"
                        site_types.append(site_type)
                    site_positions.append(tublin_position + d_site)
                # site bonds to minus end tubulin sites
                if ring_ix > 0 and tubulin_has_sites(ring_ix - 1, filament_ix, ring_connections):
                    minus_site_out_id = site_out_id - 5
                    add_edge(site_out_id, minus_site_out_id, edges) # sites out
                    add_edge(site_out_id + 1, minus_site_out_id + 1, edges) # sites 1
                    add_edge(site_out_id + 2, minus_site_out_id + 2, edges) # sites 2
                    add_edge(site_out_id + 3, minus_site_out_id + 4, edges) # sites 3 and 4
                # site bonds to plus end tubulin sites
                if ring_ix < n_rings - 1 and tubulin_has_sites(ring_ix + 1, filament_ix, ring_connections):
                    plus_site_out_id = site_out_id + 5
                    add_edge(site_out_id, plus_site_out_id, edges) # sites out
                    add_edge(site_out_id + 1, plus_site_out_id + 1, edges) # sites 1
                    add_edge(site_out_id + 2, plus_site_out_id + 2, edges) # sites 2
                    add_edge(site_out_id + 4, plus_site_out_id + 3, edges) # sites 3 and 4
    if new_edge_tub_ix is not None:
        # temporary edge between attaching tubulin sites
        add_edge(new_edge_site_ix[0], new_edge_site_ix[1], edges)
    microtubule = readdy_simulation.add_topology(
        topology_type, types + site_types, np.array(positions + site_positions)
    )
    for edge in edges:
        microtubule.get_graph().add_edge(edge[0], edge[1])
        
        
def create_microtubules_simulation(parameters, ring_connections, topology_type, new_edge_tub_ix):
    """
    create simulation and add initial particles
    """
    mt_simulation = MicrotubulesSimulation(parameters, just_bonds=True)
    add_init_microtubule_state(mt_simulation.simulation, ring_connections, topology_type, new_edge_tub_ix)
    return mt_simulation

def check_for_attach_spatial_reaction(expected_n_GTP_site_tags, simulation):
    test_monomers = ReaddyUtil.get_current_monomers(simulation.simulation.current_topologies)
    raise Exception(monomer_state_to_str(test_monomers))
    n_GTP_site_tags = 0
    for particle_id in test_monomers["particles"]:
        particle = test_monomers["particles"][particle_id]
        if "site" in particle["type_name"]:
            n_GTP_site_tags += 1 if "_G" in particle["type_name"] else 0
    assert n_GTP_site_tags == expected_n_GTP_site_tags, monomer_state_to_str(test_monomers)