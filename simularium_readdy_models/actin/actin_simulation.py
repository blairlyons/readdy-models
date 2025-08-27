#!/usr/bin/env python


import numpy as np
import readdy

from ..common import (
    ReaddyUtil,
    add_membrane_particle_types,
    add_membrane_constraints,
    get_membrane_monomers,
    all_membrane_particle_types,
)
from .actin_structure import ActinStructure
from .actin_util import ActinUtil


class ActinSimulation:
    def __init__(
        self,
        parameters,
        record=False,
        save_checkpoints=False,
        readdy_system=None,
    ):
        """
        Creates a ReaDDy branched actin simulation.

        Ref: http://jcb.rupress.org/content/jcb/180/5/887.full.pdf

        Params = Dict[str, float]
        keys:
        total_steps, timestep, box_size, temperature_C, viscosity,
        force_constant, reaction_distance, n_cpu, actin_concentration,
        arp23_concentration, cap_concentration, seed_n_fibers, seed_fiber_length,
        actin_radius, arp23_radius, cap_radius, dimerize_rate, dimerize_reverse_rate,
        trimerize_rate, trimerize_reverse_rate, pointed_growth_ATP_rate,
        pointed_growth_ADP_rate, pointed_shrink_ATP_rate,
        pointed_shrink_ADP_rate, barbed_growth_ATP_rate,
        barbed_growth_ADP_rate, nucleate_ATP_rate, nucleate_ADP_rate,
        barbed_shrink_ATP_rate, barbed_shrink_ADP_rate, arp_bind_ATP_rate,
        arp_bind_ADP_rate, arp_unbind_ATP_rate, arp_unbind_ADP_rate,
        barbed_growth_branch_ATP_rate, barbed_growth_branch_ADP_rate,
        debranching_ATP_rate, debranching_ADP_rate, cap_bind_rate,
        cap_unbind_rate, hydrolysis_actin_rate, hydrolysis_arp_rate,
        nucleotide_exchange_actin_rate, nucleotide_exchange_arp_rate, verbose
        """
        self.parameters = parameters
        self.set_constant_parameters()
        self.actin_util = ActinUtil(
            self.parameters, self.get_pointed_end_displacements()
        )
        self.create_actin_system(readdy_system)
        self.simulation = ReaddyUtil.create_readdy_simulation(
            self.system,
            self._parameter("n_cpu"),
            self._parameter("name"),
            self._parameter("total_steps"),
            record,
            save_checkpoints,
        )

    def set_constant_parameters(self):
        """
        Set values for "parameters" that never change.
        """
        self.parameters["internal_timestep"] = 0.1
        self.parameters["temperature_C"] = 22.0
        self.parameters["viscosity"] = 8.1  # cP
        self.parameters["actin_radius"] = 2.0
        self.parameters["arp23_radius"] = 2.0
        self.parameters["cap_radius"] = 3.0

    def _parameter(self, parameter_name):
        """
        Safely get a parameter using defaults when possible.
        """
        if parameter_name in self.parameters:
            return self.parameters[parameter_name]
        if parameter_name in ActinUtil.DEFAULT_PARAMETERS:
            return ActinUtil.DEFAULT_PARAMETERS[parameter_name]
        raise Exception(f"Parameter {parameter_name} is required but was not provided.")

    def create_actin_system(self, readdy_system):
        """
        Create the ReaDDy system for actin
        including particle types, constraints, and reactions.
        """
        self.parameters["temperature_K"] = self._parameter("temperature_C") + 273.15
        if readdy_system is None:
            self.system = readdy.ReactionDiffusionSystem(
                box_size=self._parameter("box_size"),
                periodic_boundary_conditions=[
                    bool(self._parameter("periodic_boundary"))
                ]
                * 3,
            )
            self.system.temperature = self.parameters["temperature_K"]
            self.add_particle_types()
            ActinUtil.check_add_global_box_potential(self.system)
            self.add_constraints()
            self.add_reactions()
        else:
            self.system = readdy_system

    def add_particle_types(self):
        """
        Add particle and topology types for actin particles
        to the ReaDDy system.
        """
        temperature = self._parameter("temperature_K")
        viscosity = self._parameter("viscosity")
        actin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self._parameter("actin_radius"), viscosity, temperature
        )  # nm^2/s
        arp23_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self._parameter("arp23_radius"), viscosity, temperature
        )  # nm^2/s
        cap_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self._parameter("cap_radius"), viscosity, temperature
        )  # nm^2/s
        self.actin_util.add_actin_types(self.system, actin_diffCoeff)
        self.actin_util.add_arp23_types(self.system, arp23_diffCoeff)
        self.actin_util.add_cap_types(self.system, cap_diffCoeff)
        self.system.topologies.add_type("Obstacle")
        self.system.add_topology_species(
            "obstacle", self._parameter("obstacle_diff_coeff")
        )
        if self._parameter("add_membrane"):
            add_membrane_particle_types(
                self.system,
                self._parameter("membrane_particle_radius"),
                temperature,
                viscosity,
            )
        if self._parameter("barbed_binding_site"):
            self.actin_util.add_binding_site_types(self.system, actin_diffCoeff)

    def add_constraints(self):
        """
        Add geometric constraints for connected actin particles,
        including bonds, angles, and repulsions, to the ReaDDy system.
        """
        util = ReaddyUtil()
        longitudinal_bonds = bool(self._parameter("longitudinal_bonds"))
        only_linear_actin = bool(self._parameter("only_linear_actin_constraints"))
        actin_constraints = bool(self._parameter("actin_constraints"))
        # force constants
        actin_angle_force_constant = float(self._parameter("angles_force_constant"))
        actin_dihedral_force_constant = float(
            self._parameter("dihedrals_force_constant")
        )
        # linear actin
        self.actin_util.add_bonds_between_actins(
            self.system,
            util,
            longitudinal_bonds,
            float(self._parameter("bonds_force_multiplier")),
        )
        if actin_constraints:
            self.actin_util.add_filament_twist_angles(
                actin_angle_force_constant, self.system, util, longitudinal_bonds
            )
            self.actin_util.add_filament_twist_dihedrals(
                actin_dihedral_force_constant,
                self.system,
                util,
                longitudinal_bonds,
                only_linear_actin,
            )
        if not only_linear_actin:
            # branch junction
            self.actin_util.add_branch_bonds(self.system, util)
            self.actin_util.add_branch_angles(
                2.0 * ActinUtil.DEFAULT_FORCE_CONSTANT, self.system, util
            )
            self.actin_util.add_branch_dihedrals(
                ActinUtil.DEFAULT_FORCE_CONSTANT, self.system, util
            )
            # capping protein
            self.actin_util.add_cap_bonds(self.system, util)
            self.actin_util.add_cap_angles(
                2.0 * ActinUtil.DEFAULT_FORCE_CONSTANT, self.system, util
            )
            self.actin_util.add_cap_dihedrals(
                ActinUtil.DEFAULT_FORCE_CONSTANT, self.system, util
            )
        # repulsions
        self.actin_util.add_repulsions(
            self._parameter("arp23_radius"),
            self._parameter("cap_radius"),
            self._parameter("obstacle_radius"),
            ActinUtil.DEFAULT_FORCE_CONSTANT,
            self.system,
            util,
            actin_actin_repulsion_potentials=True,
            longitudinal_bonds=longitudinal_bonds,
        )
        self.actin_util.add_repulsions_with_actin(
            ["obstacle"],
            self._parameter("obstacle_radius"),
            ActinUtil.DEFAULT_FORCE_CONSTANT,
            self.system,
            util,
        )
        # box potentials
        self.actin_util.add_monomer_box_potentials(self.system)
        self.actin_util.add_obstacle_box_potential(self.system)
        self.actin_util.add_extra_box(self.system)
        # membrane
        if self._parameter("add_membrane"):
            add_membrane_constraints(
                self.system,
                np.array(
                    [
                        float(self._parameter("membrane_center_x")),
                        float(self._parameter("membrane_center_y")),
                        float(self._parameter("membrane_center_z")),
                    ]
                ),
                np.array(
                    [
                        float(self._parameter("membrane_size_x")),
                        float(self._parameter("membrane_size_y")),
                        float(self._parameter("membrane_size_z")),
                    ]
                ),
                self._parameter("membrane_particle_radius"),
                self._parameter("box_size"),
            )
            self.actin_util.add_repulsions_with_actin(
                all_membrane_particle_types(),
                self._parameter("membrane_particle_radius"),
                ActinUtil.DEFAULT_FORCE_CONSTANT,
                self.system,
                util,
            )

    def add_reactions(self):
        """
        Add reactions to the ReaDDy system.
        """
        if bool(self._parameter("reactions")):
            self.actin_util.add_dimerize_reaction(self.system)
            self.actin_util.add_trimerize_reaction(self.system)
            self.actin_util.add_nucleate_reaction(self.system)
            self.actin_util.add_pointed_growth_reaction(self.system)
            self.actin_util.add_barbed_growth_reaction(self.system)
            self.actin_util.add_nucleate_branch_reaction(self.system)
            self.actin_util.add_arp23_bind_reaction(self.system)
            self.actin_util.add_cap_bind_reaction(self.system)
            self.actin_util.add_dimerize_reverse_reaction(self.system)
            self.actin_util.add_trimerize_reverse_reaction(self.system)
            self.actin_util.add_pointed_shrink_reaction(self.system)
            self.actin_util.add_barbed_shrink_reaction(self.system)
            self.actin_util.add_hydrolyze_reaction(self.system)
            self.actin_util.add_actin_nucleotide_exchange_reaction(self.system)
            self.actin_util.add_arp23_nucleotide_exchange_reaction(self.system)
            self.actin_util.add_arp23_unbind_reaction(self.system)
            self.actin_util.add_debranch_reaction(self.system)
            self.actin_util.add_cap_unbind_reaction(self.system)
        if self.do_pointed_end_translation():
            self.actin_util.add_translate_reaction(self.system)
        if self._parameter("position_obstacle_stride") > 0:
            self.actin_util.add_position_obstacle_reaction(self.system)

    def do_pointed_end_translation(self):
        result = self._parameter("displace_pointed_end_tangent") or self._parameter(
            "displace_pointed_end_radial"
        )
        if result and (
            not self._parameter("orthogonal_seed")
            or int(self._parameter("n_fixed_monomers_pointed")) < 1
        ):
            raise Exception(
                "Pointed end translation requires orthogonal seed "
                "and non-zero number of fixed monomers at the pointed end."
            )
        return result

    def get_pointed_end_displacements(self):
        """
        Get parameters for translation of the pointed end of an orthogonal seed.
        """
        if not self.do_pointed_end_translation():
            return {}
        if self._parameter("displace_pointed_end_tangent") and self._parameter(
            "displace_pointed_end_radial"
        ):
            raise Exception(
                "Cannot apply tangent and radial displacements simultaneously"
            )
        if self._parameter("displace_pointed_end_tangent"):
            displacement = {
                "get_translation": ActinUtil.get_position_for_tangent_translation,
                "parameters": {
                    "tangent_displace_speed_um_s": self._parameter(
                        "tangent_displace_speed_um_s"
                    ),
                },
            }
        if self._parameter("displace_pointed_end_radial"):
            displacement = {
                "get_translation": ActinUtil.get_position_for_radial_translation,
                "parameters": {
                    "radius_nm": self._parameter("radial_displacement_radius_nm"),
                    "theta_init_radians": np.pi,
                    "theta_final_radians": np.pi
                    + np.deg2rad(self._parameter("radial_displacement_angle_deg")),
                    "total_steps": float(self._parameter("total_steps")),
                },
            }
        result = {
            "displace_stride": int(self._parameter("displace_stride")),
        }
        for monomer_index in range(int(self._parameter("n_fixed_monomers_pointed"))):
            result[monomer_index] = displacement
        return result

    def add_random_monomers(self):
        """
        Add randomly distributed actin monomers, Arp2/3 dimers,
        and capping protein according to concentrations and box size.
        """
        box_size = self._parameter("box_size")
        self.actin_util.add_actin_monomers(
            ReaddyUtil.calculate_nParticles(
                self._parameter("actin_concentration"), box_size
            ),
            self.simulation,
        )
        self.actin_util.add_arp23_dimers(
            ReaddyUtil.calculate_nParticles(
                self._parameter("arp23_concentration"), box_size
            ),
            self.simulation,
        )
        self.actin_util.add_capping_protein(
            ReaddyUtil.calculate_nParticles(
                self._parameter("cap_concentration"), box_size
            ),
            self.simulation,
        )

    def add_random_linear_fibers(self, use_uuids=True, longitudinal_bonds=True):
        """
        Add randomly distributed and oriented linear fibers.
        """
        seed_n_fibers = int(self._parameter("seed_n_fibers"))
        if seed_n_fibers < 1:
            return
        self.actin_util.add_random_linear_fibers(
            self.simulation,
            seed_n_fibers,
            self._parameter("seed_fiber_length"),
            -1 if use_uuids else 0,
            longitudinal_bonds,
        )

    def add_fibers_from_data(self, fibers_data, use_uuids=True):
        """
        Add fibers specified in a list of FiberData.

        fiber_data: List[FiberData]
        (FiberData for mother fibers only, which should have
        their daughters' FiberData attached to their nucleated arps)
        """
        self.actin_util.add_fibers_from_data(self.simulation, fibers_data, use_uuids)

    def add_obstacles(self):
        """
        Add obstacle particles.
        """
        if not self._parameter("add_obstacles"):
            return
        n = 0
        while f"obstacle{n}_position_x" in self.parameters:
            self.simulation.add_topology(
                "Obstacle",
                ["obstacle"],
                np.array(
                    [
                        [
                            float(self._parameter(f"obstacle{n}_position_x")),
                            float(self._parameter(f"obstacle{n}_position_y")),
                            float(self._parameter(f"obstacle{n}_position_z")),
                        ]
                    ]
                ),
            )
            n += 1
        if n > 0:
            print(f"Added {n} obstacle(s).")

    def add_membrane(self):
        """
        Add membrane types and particles.
        """
        if not self._parameter("add_membrane"):
            return
        ReaddyUtil.add_monomers_from_data(
            self.simulation,
            get_membrane_monomers(
                np.array(
                    [
                        float(self._parameter("membrane_center_x")),
                        float(self._parameter("membrane_center_y")),
                        float(self._parameter("membrane_center_z")),
                    ]
                ),
                np.array(
                    [
                        float(self._parameter("membrane_size_x")),
                        float(self._parameter("membrane_size_y")),
                        float(self._parameter("membrane_size_z")),
                    ]
                ),
                self._parameter("membrane_particle_radius"),
            ),
        )

    def add_crystal_structure_monomers(self):
        """
        Add monomers exactly from the branched actin crystal structure.
        """
        type_names = [
            "actin#pointed_ATP_1",
            "actin#ATP_2",
            "actin#ATP_3",
            "actin#ATP_1",
            "actin#ATP_2",
            "actin#ATP_3",
            "actin#ATP_1",
            "actin#barbed_ATP_2",
            "arp2#branched",
            "arp3#ATP",
            "actin#branch_ATP_1",
            "actin#ATP_2",
            "actin#barbed_ATP_3",
        ]
        positions = np.zeros((13, 3))
        positions[:8, :] = ActinStructure.mother_positions
        positions[8, :] = ActinStructure.arp2_position
        positions[9, :] = ActinStructure.arp3_position
        positions[10:, :] = ActinStructure.daughter_positions
        neighbor_ids = [
            [1],
            [0, 2],
            [1, 3],
            [2, 4, 8],
            [3, 5, 9],
            [4, 6],
            [5, 7],
            [6],
            [3, 9, 10],
            [4, 8],
            [8, 11],
            [10, 12],
            [11],
        ]
        monomer_data = {
            "topologies": {
                0: {
                    "type_name": "Actin-Polymer",
                    "particle_ids": [],
                }
            },
            "particles": {},
        }
        for index in range(len(type_names)):
            monomer_data["topologies"][0]["particle_ids"].append(index)
            monomer_data["particles"][index] = {
                "type_name": type_names[index],
                "position": np.array(positions[index]),
                "neighbor_ids": neighbor_ids[index],
            }
        ReaddyUtil.add_monomers_from_data(self.simulation, monomer_data)

    def simulate(self, d_time):
        """
        Simulate in ReaDDy for the given d_time seconds.
        """

        def loop():
            readdy_actions = self.simulation._actions
            init = readdy_actions.initialize_kernel()
            diffuse = readdy_actions.integrator_euler_brownian_dynamics(
                self._parameter("internal_timestep")
            )
            calculate_forces = readdy_actions.calculate_forces()
            create_nl = readdy_actions.create_neighbor_list(
                self.system.calculate_max_cutoff().magnitude
            )
            update_nl = readdy_actions.update_neighbor_list()
            react = readdy_actions.reaction_handler_uncontrolled_approximation(
                self._parameter("internal_timestep")
            )
            observe = readdy_actions.evaluate_observables()
            init()
            create_nl()
            calculate_forces()
            update_nl()
            observe(0)
            n_steps = int(d_time * 1e9 / self._parameter("internal_timestep"))
            for t in range(1, n_steps + 1):
                diffuse()
                update_nl()
                react()
                update_nl()
                calculate_forces()
                observe(t)

        self.simulation._run_custom_loop(loop)

    def get_current_monomers(self):
        """
        During a running simulation,
        get data for topologies of particles
        from readdy.simulation.current_topologies
        as monomers.
        """
        return ReaddyUtil.get_current_monomers(self.simulation.current_topologies)
