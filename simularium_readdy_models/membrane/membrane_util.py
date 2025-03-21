#!/usr/bin/env python

from sys import float_info

import numpy as np

from ..common import ReaddyUtil


class MembraneUtil:

    @staticmethod
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
            system.add_topology_species(f"membrane#edge{n}", diffCoeff)
        
    @staticmethod
    def add_membrane_constraints(system):
        """
        Add bond, angle, and box constraints for membrane particles
        """
        # TODO
        
    @staticmethod
    def lattice_positions(center, extent, particle_radius):
        """
        Calculate lattice positions for membrane particles.
        """
        plane_dim = -1
        width = [0, 0]
        ix = 0
        for dim in range(3):
            if extent[dim] < float_info.epsilon:
                plane_dim = dim
                continue
            width[ix] = round(2. * extent[dim])
            ix += 1
        coords = [
            np.arange(0, width[0], particle_radius), 
            np.arange(0, width[1], particle_radius * np.sqrt(3))
        ]
        if plane_dim < 0 or width[1] < float_info.epsilon:
            raise Exception("The membrane extent must be zero in only one dimension.")
        cols = coords[0].shape[0]
        rows = coords[1].shape[0]
        n_lattice_points = cols * (2 * coords[1].shape[0])
        positions = np.zeros((2 * n_lattice_points, 3))
        lattice_dim = 0
        for dim in range(3):
            if dim == plane_dim:
                positions[:n_lattice_points, dim] = center[dim] - 0.5 * particle_radius
                positions[n_lattice_points:, dim] = center[dim] + 0.5 * particle_radius
                continue
            values = coords[lattice_dim] - extent[dim] + center[dim]
            offset = 0.5 * particle_radius * (1 if lattice_dim < 1 else np.sqrt(3))
            for i in range(rows):
                start_ix = 2 * i * cols
                positions[start_ix:start_ix + cols, dim] = values if lattice_dim < 1 else values[i]
                start_ix = (2 * i + 1) * cols
                positions[start_ix:start_ix + cols, dim] = (values if lattice_dim < 1 else values[i]) + offset
                start_ix = 2 * i * cols + n_lattice_points
                positions[start_ix:start_ix + cols, dim] = values if lattice_dim < 1 else values[i]
                start_ix = (2 * i + 1) * cols + n_lattice_points
                positions[start_ix:start_ix + cols, dim] = (values if lattice_dim < 1 else values[i]) + offset
            lattice_dim += 1
            
        # testing
        import matplotlib.pyplot as plt
        plt.scatter(positions[:n_lattice_points, 0], positions[:n_lattice_points, 2], color='blue')
        plt.axis('equal')
        plt.show()
        
        return positions

    @staticmethod
    def add_membrane(system, simulation, center, extent, particle_radius, temperature_K, viscosity):
        """
        add a membrane (and necessary types and constraints) to the system and simulation.
        """
        MembraneUtil.add_membrane_particle_types(system, particle_radius, temperature_K, viscosity)
        MembraneUtil.add_membrane_constraints(system)
        positions = MembraneUtil.lattice_positions(center, extent, particle_radius)
        types = []  # TODO
        membrane = simulation.add_topology("Membrane", types, positions)
        for i in range(1, 3):
            membrane.get_graph().add_edge(0, i)  # TODO
