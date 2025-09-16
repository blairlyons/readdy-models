#!/usr/bin/env python

import numpy as np

from simularium_readdy_models import ReaddyUtil


def run_readdy(total_steps, system, simulation):
    # setup readdy functions
    timestep = 0.1
    readdy_actions = simulation._actions
    init = readdy_actions.initialize_kernel()
    diffuse = readdy_actions.integrator_euler_brownian_dynamics(timestep)
    calculate_forces = readdy_actions.calculate_forces()
    create_nl = readdy_actions.create_neighbor_list(
        system.calculate_max_cutoff().magnitude
    )
    update_nl = readdy_actions.update_neighbor_list()
    react = readdy_actions.reaction_handler_uncontrolled_approximation(timestep)
    react_top = readdy_actions.topology_reaction_handler(timestep)
    observe = readdy_actions.evaluate_observables()
    # init simulation
    init()
    create_nl()
    calculate_forces()
    observe(0)
    for t in range(total_steps):
        print(f"Running timestep t = {t}")
        diffuse()  # results in NaN positions (???)
        update_nl()
        react()
        react_top()
        update_nl()
        calculate_forces()
        observe(t + 1)


def get_min_id(monomers):
    """
    pytest caches something that causes particle IDs
    to not always start at 0 in separate tests,
    so use min ID to offset test values to match expected.
    """
    particle_ids = list(monomers["particles"].keys())
    return np.amin(np.array(particle_ids))


def monomer_state_to_str(monomers):
    result = "topologies:\n"
    for top_id in monomers["topologies"]:
        top = monomers["topologies"][top_id]
        type_name = top["type_name"]
        particles = top["particle_ids"].copy()
        particles.sort()
        result += f"  {top_id} : {type_name} {particles}\n"
    result += "particles:\n"
    min_id = get_min_id(monomers)
    particle_ids = list(monomers["particles"].keys())
    particle_ids.sort()
    for particle_id in particle_ids:
        particle = monomers["particles"][particle_id]
        type_name = particle["type_name"]
        position = particle["position"]
        neighbor_ids = [nid - min_id for nid in particle["neighbor_ids"]]
        result += (
            f"  {particle_id - min_id} : {type_name}, {position}, {neighbor_ids}\n"
        )
    return result


def check_readdy_state(simulation, expected_monomers, ignore_extra_spatial_rxn=False):
    test_monomers = ReaddyUtil.get_current_monomers(
        simulation.simulation.current_topologies
    )
    # if ignore_extra_spatial_rxn:
    # raise Exception(monomer_state_to_str(test_monomers))
    assert_monomers_equal(
        test_monomers,
        expected_monomers,
        ignore_extra_spatial_rxn=ignore_extra_spatial_rxn,
        test_position=False,
    )


def assert_monomers_equal(
    test_monomers,
    expected_monomers,
    ignore_extra_spatial_rxn=False,
    test_position=False,
):
    """
    Assert two topologies (in monomer form) are equivalent.
    """
    # check topology has the correct type_name
    # and contains the correct particle_ids (in any order, starting at any index)
    test_top_id = list(test_monomers["topologies"].keys())[0]
    exp_top_id = list(expected_monomers["topologies"].keys())[0]
    assert (
        "SpatialRxnResult" not in test_monomers["topologies"][test_top_id]["type_name"]
    )
    if not ignore_extra_spatial_rxn:
        assert (
            test_monomers["topologies"][test_top_id]["type_name"]
            == expected_monomers["topologies"][exp_top_id]["type_name"]
        )
    test_particle_ids = test_monomers["topologies"][test_top_id]["particle_ids"]
    # pytest caches something that causes particle IDs to not always start at 0
    min_id = get_min_id(test_monomers)
    if ignore_extra_spatial_rxn:
        assert len(test_particle_ids) >= len(
            expected_monomers["topologies"][exp_top_id]["particle_ids"]
        )
    else:
        assert len(test_particle_ids) == len(
            expected_monomers["topologies"][exp_top_id]["particle_ids"]
        )
        for particle_id in test_particle_ids:
            assert (
                particle_id - min_id
                in expected_monomers["topologies"][exp_top_id]["particle_ids"]
            ), f"top particle IDs = {test_particle_ids}"
    for particle_id in expected_monomers["topologies"][exp_top_id]["particle_ids"]:
        assert particle_id + min_id in test_particle_ids
    # check the particle types, positions (optionally), and neighbors
    for particle_id in test_monomers["particles"]:
        test_particle = test_monomers["particles"][particle_id]
        exp_particle = expected_monomers["particles"][particle_id - min_id]
        if not ignore_extra_spatial_rxn:
            assert test_particle["type_name"] == exp_particle["type_name"]
        test_neighbor_ids = test_particle["neighbor_ids"].copy()
        test_neighbor_ids.sort()
        exp_neighbor_ids = exp_particle["neighbor_ids"].copy()
        exp_neighbor_ids.sort()
        exp_neighbor_ids = [nid + min_id for nid in exp_neighbor_ids]
        if not ignore_extra_spatial_rxn:
            assert (
                test_neighbor_ids == exp_neighbor_ids
            ), f"Neighbors don't match for particle ID {particle_id - min_id} : {test_neighbor_ids} != {exp_neighbor_ids}"
        else:
            for exp_neighbor_id in exp_neighbor_ids:
                assert exp_neighbor_id in test_particle["neighbor_ids"]
        if test_position:
            np.testing.assert_almost_equal(
                test_particle["position"], exp_particle["position"], decimal=2
            )


def assert_fibers_equal(topology_fibers1, topology_fibers2, test_position=False):
    """
    Assert two topologies (in fiber form) are equivalent.
    """
    # check topology has the correct type_name
    # and contains the correct points (in order)
    assert len(topology_fibers1) == len(topology_fibers2)
    for f in range(len(topology_fibers1)):
        assert topology_fibers1[f].type_name == topology_fibers2[f].type_name
        assert len(topology_fibers1[f].points) == len(topology_fibers2[f].points)
        for p in range(len(topology_fibers1[f].points)):
            np.testing.assert_allclose(
                topology_fibers1[f].points[p], topology_fibers2[f].points[p]
            )
