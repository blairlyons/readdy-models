#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simularium_readdy_models.actin import ActinStructure


actin_parameters_rxns_off = {
    "name": "actin",
    "total_steps": 1e3,
    "time_step": 0.0000001,
    "internal_timestep": 0.1,  # ns
    "box_size": 1000.0,  # nm
    "temperature_C": 22.0,  # from Pollard experiments
    "viscosity": 8.1,  # cP, viscosity in cytoplasm
    "force_constant": 250.0,
    "reaction_distance": 1.0,  # nm
    "n_cpu": 4,
    "actin_concentration": 200.0,  # uM
    "arp23_concentration": 10.0,  # uM
    "cap_concentration": 0.0,  # uM
    "n_fibers": 0,
    "fiber_length": 0.0,
    "actin_radius": 2.0,  # nm
    "arp23_radius": 2.0,  # nm
    "cap_radius": 3.0,  # nm
    "dimerize_rate": 1e-30,  # 1/ns
    "dimerize_reverse_rate": 1e-30,  # 1/ns
    "trimerize_rate": 1e-30,  # 1/ns
    "trimerize_reverse_rate": 1e-30,  # 1/ns
    "pointed_growth_ATP_rate": 1e-30,  # 1/ns
    "pointed_growth_ADP_rate": 1e-30,  # 1/ns
    "pointed_shrink_ATP_rate": 1e-30,  # 1/ns
    "pointed_shrink_ADP_rate": 1e-30,  # 1/ns
    "barbed_growth_ATP_rate": 1e-30,  # 1/ns
    "barbed_growth_ADP_rate": 1e-30,  # 1/ns
    "nucleate_ATP_rate": 1e-30,  # 1/ns
    "nucleate_ADP_rate": 1e-30,  # 1/ns
    "barbed_shrink_ATP_rate": 1e-30,  # 1/ns
    "barbed_shrink_ADP_rate": 1e-30,  # 1/ns
    "arp_bind_ATP_rate": 1e-30,  # 1/ns
    "arp_bind_ADP_rate": 1e-30,  # 1/ns
    "arp_unbind_ATP_rate": 1e-30,  # 1/ns
    "arp_unbind_ADP_rate": 1e-30,  # 1/ns
    "barbed_growth_branch_ATP_rate": 1e-30,  # 1/ns
    "barbed_growth_branch_ADP_rate": 1e-30,  # 1/ns
    "debranching_ATP_rate": 1e-30,  # 1/ns
    "debranching_ADP_rate": 1e-30,  # 1/ns
    "cap_bind_rate": 1e-30,  # 1/ns
    "cap_unbind_rate": 1e-30,  # 1/ns
    "hydrolysis_actin_rate": 1e-30,  # 1/ns
    "hydrolysis_arp_rate": 1e-30,  # 1/ns
    "nucleotide_exchange_actin_rate": 1e-30,  # 1/ns
    "nucleotide_exchange_arp_rate": 1e-30,  # 1/ns
    "use_box_actin": False,
    "use_box_arp": False,
    "use_box_cap": False,
    "verbose": False,
    "periodic_boundary": True,
    "obstacle_radius": 1.0,
}


def actin_monomer():
    return {
        "topologies": {
            0: {
                "type_name": "Actin-Monomer",
                "particle_ids": [
                    0,
                ],
            }
        },
        "particles": {
            0: {
                "unique_id": 0,
                "type_name": "actin#free_ATP",
                "position": np.array(ActinStructure.mother_positions[0]),
                "neighbor_ids": [],
            },
        },
    }


def actin_dimer():
    return {
        "topologies": {
            0: {
                "type_name": "Actin-Dimer",
                "particle_ids": [
                    0,
                    1,
                ],
            }
        },
        "particles": {
            0: {
                "unique_id": 0,
                "type_name": "actin#pointed_ATP_1",
                "position": np.array(ActinStructure.mother_positions[0]),
                "neighbor_ids": [1],
            },
            1: {
                "unique_id": 1,
                "type_name": "actin#barbed_ATP_2",
                "position": np.array(ActinStructure.mother_positions[1]),
                "neighbor_ids": [0],
            },
        },
    }
