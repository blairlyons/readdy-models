#!/usr/bin/env python

import math
from typing import Dict

import numpy as np

from ..common import ReaddyUtil
from .actin_structure import ActinStructure
from .actin_util import ActinUtil


class ActinAnalyzer:
    """
    Analyze data from a ReaDDy actin trajectory.
    """

    @staticmethod
    def bond_energy_constants(temp_c: float) -> Dict[int, float]:
        """
        Get k for lateral (offset = 1)
        and longitudinal (offset = 2) bonds.
        """
        KT_NA = 2.479  # kJ / mol ??
        THEO_TEMP_K = 298.0
        temp_k = temp_c + 273.15
        KT_NA = temp_k * KT_NA / THEO_TEMP_K  # ???
        k_long = 250.0  # kJ / mol * nm^2
        k_lat = 250.0  # kJ / mol * nm^2
        k_long = k_long / KT_NA
        k_lat = k_lat / KT_NA
        return {
            1: k_lat,
            2: k_long,
        }

    @staticmethod
    def ideal_bond_lengths() -> Dict[int, float]:
        """
        Get ideal bond lengths for lateral (offset = 1)
        and longitudinal (offset = 2) bonds.
        """
        ideal_length_lat = np.linalg.norm(
            ActinStructure.mother_positions[1] - ActinStructure.mother_positions[0]
        )
        ideal_length_long = np.linalg.norm(
            ActinStructure.mother_positions[2] - ActinStructure.mother_positions[0]
        )
        return {
            1: ideal_length_lat,
            2: ideal_length_long,
        }

    @staticmethod
    def analyze_reaction_rate_over_time(reactions, time_inc_s, reaction_name):
        """
        Get a list of the reaction rate per second
        at each analyzed timestep of the given reaction.
        """
        if reaction_name not in reactions:
            print(f"Couldn't find reaction: {reaction_name}")
            return None
        return np.insert(reactions[reaction_name].to_numpy() / time_inc_s, 0, 0.0)

    @staticmethod
    def analyze_average_for_series(data):
        """
        Get a list of the average per item of the given 2D data.
        """
        result = []
        for t in range(len(data)):
            frame_sum = 0.0
            n = 0
            for d in data[t]:
                frame_sum += float(d)
                n += 1
            result.append(frame_sum / float(n) if n > 0 else 0.0)
        return np.array(result)

    @staticmethod
    def analyze_stddev_for_series(data):
        """
        Get a list of the std deviation per item of the given 2D data.
        """
        result = []
        for t in range(len(data)):
            data_arr = np.array(data[t])
            result.append(np.std(data_arr) if data_arr.shape[0] > 0 else 0.0)
        return np.array(result)

    @staticmethod
    def _free_actin_types():
        """
        Get all the types for actins freely diffusing.
        """
        return ["actin#free", "actin#free_ATP"]

    @staticmethod
    def _pointed_actin_types():
        """
        Get all the types for actins at the pointed end of a filament.
        """
        result = []
        for i in range(1, 6):
            result.append(f"actin#pointed_{i}")
            result.append(f"actin#pointed_ATP_{i}")
            result.append(f"actin#pointed_fixed_{i}")
            result.append(f"actin#pointed_fixed_ATP_{i}")
        return result

    @staticmethod
    def _actin_chain_types():
        """
        Get all the types for actins along a filament from pointed to barbed,
        not including pointed or branch actins that start a new filament.
        """
        result = []
        for i in ActinUtil.polymer_number_range():
            result.append(
                [
                    f"actin#{i}",
                    f"actin#ATP_{i}",
                    f"actin#mid_{i}",
                    f"actin#mid_ATP_{i}",
                    f"actin#fixed_{i}",
                    f"actin#fixed_ATP_{i}",
                    f"actin#mid_fixed_{i}",
                    f"actin#mid_fixed_ATP_{i}",
                    f"actin#barbed_{i}",
                    f"actin#barbed_ATP_{i}",
                    f"actin#fixed_barbed_{i}",
                    f"actin#fixed_barbed_ATP_{i}",
                ]
            )
        return result

    @staticmethod
    def _middle_actin_types():
        """
        Get all the types for actins in the middle of a filament.
        """
        result = []
        for i in range(1, 6):
            result.append(f"actin#{i}")
            result.append(f"actin#ATP_{i}")
            result.append(f"actin#mid_{i}")
            result.append(f"actin#mid_ATP_{i}")
            result.append(f"actin#fixed_{i}")
            result.append(f"actin#fixed_ATP_{i}")
            result.append(f"actin#mid_fixed_{i}")
            result.append(f"actin#mid_fixed_ATP_{i}")
        return result

    @staticmethod
    def _barbed_actin_types():
        """
        Get all the types for actins at the barbed end of a filament.
        """
        result = []
        for i in range(1, 6):
            result.append(f"actin#barbed_{i}")
            result.append(f"actin#barbed_ATP_{i}")
            result.append(f"actin#fixed_barbed_{i}")
            result.append(f"actin#fixed_barbed_ATP_{i}")
        return result

    @staticmethod
    def _branch_actin_types():
        """
        Get all the types for actins at the pointed end of a branch.
        """
        return [
            "actin#branch_1",
            "actin#branch_ATP_1",
            "actin#branch_barbed_1",
            "actin#branch_barbed_ATP_1",
        ]

    @staticmethod
    def _filamentous_actin_types():
        """
        Get all the types for actins in a filament.
        """
        return (
            ActinAnalyzer._pointed_actin_types()
            + ActinAnalyzer._middle_actin_types()
            + ActinAnalyzer._barbed_actin_types()
            + ActinAnalyzer._branch_actin_types()
        )

    @staticmethod
    def _filamentous_ATP_actin_types():
        """
        Get all the types for actins with ATP bound.
        """
        result = []
        for i in range(1, 6):
            result.append(f"actin#ATP_{i}")
            result.append(f"actin#mid_ATP_{i}")
            result.append(f"actin#pointed_ATP_{i}")
            result.append(f"actin#barbed_ATP_{i}")
            result.append(f"actin#fixed_ATP_{i}")
            result.append(f"actin#mid_fixed_ATP_{i}")
            result.append(f"actin#pointed_fixed_ATP_{i}")
            result.append(f"actin#fixed_barbed_ATP_{i}")
        result.append("actin#branch_ATP_1")
        result.append("actin#branch_barbed_ATP_1")
        return result

    @staticmethod
    def _filamentous_ADP_actin_types():
        """
        Get all the types for actins with ADP bound.
        """
        result = []
        for i in range(1, 6):
            result.append(f"actin#{i}")
            result.append(f"actin#mid_{i}")
            result.append(f"actin#pointed_{i}")
            result.append(f"actin#barbed_{i}")
            result.append(f"actin#fixed_{i}")
            result.append(f"actin#mid_fixed_{i}")
            result.append(f"actin#pointed_fixed_{i}")
            result.append(f"actin#fixed_barbed_{i}")
        result.append("actin#branch_1")
        result.append("actin#branch_barbed_1")
        return result

    @staticmethod
    def _get_frame_filaments_from_start_actins(start_actin_ids, frame_particle_data):
        """
        Get a list of filaments in the given frame of data
        starting from each of the start_actin_ids.
        Each filament is a list of the actin ids in the filament
        in order from pointed to barbed end.
        """
        result = []
        chain_types = ActinAnalyzer._actin_chain_types()
        for start_actin_id in start_actin_ids:
            pointed_number = int(
                frame_particle_data["particles"][start_actin_id]["type_name"][-1]
            )
            result.append(
                ReaddyUtil.analyze_frame_get_chain_of_types(
                    start_actin_id,
                    chain_types,
                    frame_particle_data,
                    result=[start_actin_id],
                    next_neighbor_index=pointed_number,
                )
            )
        return result

    @staticmethod
    def _frame_mother_filaments(frame_particle_data):
        """
        Get a list of mother filaments in the given frame of data,
        each filament is a list of the actin ids in the filament
        in order from pointed to barbed end.
        """
        return ActinAnalyzer._get_frame_filaments_from_start_actins(
            ReaddyUtil.analyze_frame_get_ids_for_types(
                ActinAnalyzer._pointed_actin_types(), frame_particle_data
            ),
            frame_particle_data,
        )

    @staticmethod
    def _frame_daughter_filaments(frame_particle_data):
        """
        Get a list of daughter filaments in the given frame of data,
        each filament is a list of the actin ids in the filament
        in order from pointed to barbed end.
        """
        return ActinAnalyzer._get_frame_filaments_from_start_actins(
            ReaddyUtil.analyze_frame_get_ids_for_types(
                ActinAnalyzer._branch_actin_types(), frame_particle_data
            ),
            frame_particle_data,
        )

    @staticmethod
    def _frame_all_filaments(frame_particle_data):
        """
        Get a list of mother and daughter filaments
        in the given frame of data,
        each filament is a list of the actin ids in the filament
        in order from pointed to barbed end.
        """
        return ActinAnalyzer._frame_mother_filaments(
            frame_particle_data
        ) + ActinAnalyzer._frame_daughter_filaments(frame_particle_data)

    @staticmethod
    def analyze_ratio_of_filamentous_to_total_actin(monomer_data):
        """
        Get a list of the ratio of actin in filaments to total actin over time.
        """
        result = []
        for t in range(len(monomer_data)):
            free_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._free_actin_types(), monomer_data[t]
                )
            )
            filamentous_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._filamentous_actin_types(),
                    monomer_data[t],
                )
            )
            if free_actin + filamentous_actin > 0:
                result.append(filamentous_actin / float(free_actin + filamentous_actin))
            else:
                result.append(0)
        return np.array(result)

    @staticmethod
    def analyze_ratio_of_bound_ATP_actin_to_total_actin(monomer_data):
        """
        Get a list of the ratio of bound ATP-actin to total actin over time.
        """
        result = []
        for t in range(len(monomer_data)):
            ATP_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._filamentous_ATP_actin_types(),
                    monomer_data[t],
                )
            )
            free_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._free_actin_types(), monomer_data[t]
                )
            )
            filamentous_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._filamentous_actin_types(),
                    monomer_data[t],
                )
            )
            if free_actin + filamentous_actin > 0:
                result.append(ATP_actin / float(free_actin + filamentous_actin))
            else:
                result.append(1.0)
        return np.array(result)

    @staticmethod
    def analyze_ratio_of_daughter_to_total_actin(monomer_data):
        """
        Get a list of the ratio
        [daughter filament actin] / [total actin] over time.
        """
        result = []
        for t in range(len(monomer_data)):
            daughter_actin = 0
            daughter_filaments = ActinAnalyzer._frame_daughter_filaments(
                monomer_data[t]
            )
            for daughter_filament in daughter_filaments:
                daughter_actin += len(daughter_filament)
            free_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._free_actin_types(), monomer_data[t]
                )
            )
            filamentous_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._filamentous_actin_types(),
                    monomer_data[t],
                )
            )
            if free_actin + filamentous_actin > 0:
                result.append(daughter_actin / float(free_actin + filamentous_actin))
            else:
                result.append(0)
        return np.array(result)

    @staticmethod
    def analyze_mother_filament_lengths(monomer_data):
        """
        Get a list of the number of monomers in each mother filament
        in each frame of the trajectory.
        """
        result = []
        for t in range(len(monomer_data)):
            mother_filaments = ActinAnalyzer._frame_mother_filaments(monomer_data[t])
            result.append([])
            for filament in mother_filaments:
                result[t].append(len(filament))
        return result

    @staticmethod
    def analyze_daughter_filament_lengths(monomer_data):
        """
        Get a list of the number of monomers in each daughter filament
        in each frame of the trajectory.
        """
        result = []
        for t in range(len(monomer_data)):
            daughter_filaments = ActinAnalyzer._frame_daughter_filaments(
                monomer_data[t]
            )
            result.append([])
            for filament in daughter_filaments:
                result[t].append(len(filament))
        return result

    @staticmethod
    def analyze_ratio_of_bound_to_total_arp23(monomer_data):
        """
        Get a list of the ratio of bound to total arp2/3 complexes over time.
        """
        result = []
        for t in range(len(monomer_data)):
            bound_arp23 = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ["arp2", "arp2#branched"], monomer_data[t]
                )
            )
            free_arp23 = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ["arp2#free"], monomer_data[t]
                )
            )
            if free_arp23 + bound_arp23 > 0:
                result.append(bound_arp23 / float(free_arp23 + bound_arp23))
            else:
                result.append(0)
        return np.array(result)

    @staticmethod
    def analyze_ratio_of_capped_ends_to_total_ends(monomer_data):
        """
        Get a list of the ratio of barbed ends capped
        with capping protein to all barbed ends over time.
        """
        capped_end_types = ["cap#bound"]
        growing_end_types = ActinAnalyzer._barbed_actin_types() + [
            "actin#branch_barbed_1",
            "actin#branch_barbed_ATP_1",
        ]
        result = []
        for t in range(len(monomer_data)):
            capped_ends = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    capped_end_types, monomer_data[t]
                )
            )
            growing_ends = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    growing_end_types, monomer_data[t]
                )
            )
            if growing_ends + capped_ends > 0:
                result.append(capped_ends / float(growing_ends + capped_ends))
            else:
                result.append(0)
        return np.array(result)

    @staticmethod
    def _get_axis_position_for_actin(
        frame_particle_data, actin_ids, box_size, periodic_boundary=True
    ):
        """
        get the position on the filament axis closest to an actin
        actin_ids = [previous actin id, this actin id, next actin id].
        """
        positions = []
        for i in range(3):
            positions.append(frame_particle_data["particles"][actin_ids[i]]["position"])
        return ActinUtil.get_actin_axis_position(positions, box_size, periodic_boundary)

    @staticmethod
    def neighbor_types_to_string(particle_id, frame_particle_data):
        """ """
        result = ""
        for neighbor_id in (frame_particle_data["particles"][particle_id])[
            "neighbor_ids"
        ]:
            result += (frame_particle_data["particles"][neighbor_id])[
                "type_name"
            ] + ", "
        return result[:-2]

    @staticmethod
    def positions_to_string(particle_ids, box_size, frame_particle_data):
        """ """
        positions = []
        for i in range(3):
            positions.append(
                frame_particle_data["particles"][particle_ids[i]]["position"]
            )
        for i in range(len(positions)):
            if i == 1:
                continue
            positions[i] = ReaddyUtil.get_non_periodic_boundary_position(
                positions[1], positions[i], box_size
            )
        return str(positions)

    @staticmethod
    def _get_frame_branch_ids(frame_particle_data):
        """
        for each branch point at a time frame, get list of ids for (in order):
        - [0,1,2,3] 4 actins after branch on main filament
          (toward barbed end, ordered from pointed end toward barbed end)
        - [4,5,6,7] first 4 actins on branch
          (ordered from pointed end toward barbed end).
        """
        arp2_ids = ReaddyUtil.analyze_frame_get_ids_for_types(
            ["arp2#branched"], frame_particle_data
        )
        actin_types = (
            ActinAnalyzer._pointed_actin_types()
            + ActinAnalyzer._middle_actin_types()
            + ActinAnalyzer._barbed_actin_types()
        )
        chain_types = ActinAnalyzer._actin_chain_types()
        result = []
        for arp2_id in arp2_ids:
            actin1_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
                arp2_id, ActinAnalyzer._branch_actin_types(), frame_particle_data
            )
            if actin1_id is None:
                print(
                    "Failed to parse branch point: couldn't find branch actin\n"
                    + frame_particle_data["particles"][arp2_id]["type_name"]
                    + " neighbor types are: ["
                    + ActinAnalyzer.neighbor_types_to_string(
                        arp2_id, frame_particle_data
                    )
                    + "]"
                )
                continue
            pointed_number = int(
                frame_particle_data["particles"][actin1_id]["type_name"][-1]
            )
            branch_actins = ReaddyUtil.analyze_frame_get_chain_of_types(
                actin1_id,
                chain_types,
                frame_particle_data,
                3,
                arp2_id,
                [actin1_id],
                next_neighbor_index=pointed_number,
            )
            if len(branch_actins) < 4:
                # not enough daughter actins to measure branch
                continue
            actin_arp2_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
                arp2_id, actin_types, frame_particle_data, [actin1_id]
            )
            if actin_arp2_id is None:
                print(
                    "Failed to parse branch point: failed to find actin_arp2\n"
                    + frame_particle_data["particles"][arp2_id]["type_name"]
                    + " neighbor types are: ["
                    + ActinAnalyzer.neighbor_types_to_string(
                        arp2_id, frame_particle_data
                    )
                    + "]"
                )
                continue
            n = ReaddyUtil.calculate_polymer_number(
                int(frame_particle_data["particles"][actin_arp2_id]["type_name"][-1:]),
                1,
                ActinUtil.n_polymer_numbers(),
            )
            actin_arp3_types = [
                f"actin#{n}",
                f"actin#ATP_{n}",
                f"actin#mid_{n}",
                f"actin#mid_ATP_{n}",
                f"actin#pointed_{n}",
                f"actin#pointed_ATP_{n}",
                f"actin#branch_{n}",
                f"actin#branch_ATP_{n}",
            ]
            actin_arp3_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
                actin_arp2_id, actin_arp3_types, frame_particle_data
            )
            if actin_arp3_id is None:
                raise Exception(
                    "Failed to parse branch point: failed to find actin_arp3\n"
                    + frame_particle_data["particles"][actin_arp2_id]["type_name"]
                    + " neighbor types are: ["
                    + ActinAnalyzer.neighbor_types_to_string(
                        actin_arp2_id, frame_particle_data
                    )
                    + "]\n"
                    + frame_particle_data["particles"][arp2_id]["type_name"]
                    + " neighbor types are: ["
                    + ActinAnalyzer.neighbor_types_to_string(
                        arp2_id, frame_particle_data
                    )
                    + "]"
                )
            pointed_number = int(
                frame_particle_data["particles"][actin_arp3_id]["type_name"][-1]
            )
            main_actins = ReaddyUtil.analyze_frame_get_chain_of_types(
                actin_arp3_id,
                chain_types,
                frame_particle_data,
                2,
                actin_arp2_id,
                [actin_arp2_id, actin_arp3_id],
                next_neighbor_index=pointed_number,
            )
            if len(main_actins) < 4:
                # not enough mother actins to measure branch
                continue
            result.append(main_actins + branch_actins)
        return result

    @staticmethod
    def _get_frame_branch_angles(frame_particle_data, box_size, periodic_boundary=True):
        """
        get the angle between mother and daughter filament
        at each branch point in the given frame of the trajectory.
        """
        branch_ids = ActinAnalyzer._get_frame_branch_ids(frame_particle_data)
        result = []
        for branch in branch_ids:
            actin_ids = [branch[0], branch[1], branch[2]]
            main_pos1 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, actin_ids, box_size, periodic_boundary
            )
            if main_pos1 is None or ReaddyUtil.vector_is_invalid(main_pos1):
                pos_to_string = "None" if main_pos1 is None else main_pos1
                raise Exception(
                    "Failed to get axis position for mother actin 1, "
                    f"pos = {pos_to_string}\ntried to use positions: "
                    + ActinAnalyzer.positions_to_string(
                        actin_ids, box_size, frame_particle_data
                    )
                )
            actin_ids = [branch[1], branch[2], branch[3]]
            main_pos2 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, actin_ids, box_size, periodic_boundary
            )
            if main_pos2 is None or ReaddyUtil.vector_is_invalid(main_pos2):
                pos_to_string = "None" if main_pos2 is None else main_pos2
                raise Exception(
                    "Failed to get axis position for mother actin 2"
                    f"pos = {pos_to_string}\ntried to use positions: "
                    + ActinAnalyzer.positions_to_string(
                        actin_ids, box_size, frame_particle_data
                    )
                )
            actin_ids = [branch[4], branch[5], branch[6]]
            v_main = ReaddyUtil.normalize(main_pos2 - main_pos1)
            branch_pos1 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, actin_ids, box_size, periodic_boundary
            )
            if branch_pos1 is None or ReaddyUtil.vector_is_invalid(branch_pos1):
                pos_to_string = "None" if branch_pos1 is None else branch_pos1
                raise Exception(
                    "Failed to get axis position for daughter actin 1"
                    f"pos = {pos_to_string}\ntried to use positions: "
                    + ActinAnalyzer.positions_to_string(
                        actin_ids, box_size, frame_particle_data
                    )
                )
            actin_ids = [branch[5], branch[6], branch[7]]
            branch_pos2 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, actin_ids, box_size, periodic_boundary
            )
            if branch_pos2 is None or ReaddyUtil.vector_is_invalid(branch_pos2):
                pos_to_string = "None" if branch_pos2 is None else branch_pos2
                raise Exception(
                    "Failed to get axis position for daughter actin 2"
                    f"pos = {pos_to_string}\ntried to use positions: "
                    + ActinAnalyzer.positions_to_string(
                        actin_ids, box_size, frame_particle_data
                    )
                )
            v_branch = ReaddyUtil.normalize(branch_pos2 - branch_pos1)
            result.append(ReaddyUtil.get_angle_between_vectors(v_main, v_branch, True))
        return result

    @staticmethod
    def analyze_branch_angles(monomer_data, box_size, periodic_boundary):
        """
        Get a list of the angles between mother and daughter filaments
        at each branch point in each frame of the trajectory.
        """
        result = []
        for t in range(len(monomer_data)):
            branch_angles = ActinAnalyzer._get_frame_branch_angles(
                monomer_data[t], box_size, periodic_boundary
            )
            result.append(branch_angles)
        return result

    @staticmethod
    def _calculate_pitch(
        frame_particle_data, actin1_ids, actin2_ids, box_size, periodic_boundary=True
    ):
        """
        Calculate the pitch of the helix between two actins
        actin_ids = [previous actin id, this actin id, next actin id]
        for each of the two actins.
        """
        actin1_pos = frame_particle_data["particles"][actin1_ids[1]]["position"]
        actin1_axis_pos = ActinAnalyzer._get_axis_position_for_actin(
            frame_particle_data, actin1_ids, box_size, periodic_boundary
        )
        if actin1_axis_pos is None or ReaddyUtil.vector_is_invalid(actin1_axis_pos):
            raise Exception(
                "Failed to get axis position for actin 1\n"
                "tried to use positions: "
                + ActinAnalyzer.positions_to_string(
                    actin1_ids, box_size, frame_particle_data
                )
            )
        v1 = ReaddyUtil.normalize(actin1_axis_pos - actin1_pos)
        actin2_pos = frame_particle_data["particles"][actin2_ids[1]]["position"]
        actin2_axis_pos = ActinAnalyzer._get_axis_position_for_actin(
            frame_particle_data, actin2_ids, box_size, periodic_boundary
        )
        if actin2_axis_pos is None or ReaddyUtil.vector_is_invalid(actin2_axis_pos):
            raise Exception(
                "Failed to get axis position for actin 2\n"
                "tried to use positions: "
                + ActinAnalyzer.positions_to_string(
                    actin2_ids, box_size, frame_particle_data
                )
            )
        v2 = ReaddyUtil.normalize(actin2_axis_pos - actin2_pos)
        actin2_axis_pos = ReaddyUtil.get_non_periodic_boundary_position(
            actin1_axis_pos, actin2_axis_pos, box_size
        )
        length = np.linalg.norm(actin2_axis_pos - actin1_axis_pos)
        angle = ReaddyUtil.get_angle_between_vectors(v1, v2, True)
        return (360.0 / angle) * length

    @staticmethod
    def _get_frame_short_helix_pitches(
        frame_particle_data, box_size, periodic_boundary=True
    ):
        """
        Get the pitch of the short helix between all actins on each filament
        for a given frame of data.
        """
        result = []
        filaments = ActinAnalyzer._frame_all_filaments(frame_particle_data)
        for filament in filaments:
            for i in range(1, len(filament) - 3):
                short_pitch = ActinAnalyzer._calculate_pitch(
                    frame_particle_data,
                    [filament[i - 1], filament[i], filament[i + 1]],
                    [filament[i], filament[i + 1], filament[i + 2]],
                    box_size,
                    periodic_boundary,
                )
                if short_pitch is not None:
                    result.append(short_pitch)
        return result

    @staticmethod
    def _get_frame_long_helix_pitches(
        frame_particle_data, box_size, periodic_boundary=True
    ):
        """
        Get the pitch of the long helix between all actins on each filament
        for a given frame of data.
        """
        result = []
        filaments = ActinAnalyzer._frame_all_filaments(frame_particle_data)
        for filament in filaments:
            for i in range(1, len(filament) - 3):
                long_pitch = ActinAnalyzer._calculate_pitch(
                    frame_particle_data,
                    [filament[i - 1], filament[i], filament[i + 1]],
                    [filament[i + 1], filament[i + 2], filament[i + 3]],
                    box_size,
                    periodic_boundary,
                )
                if long_pitch is not None:
                    result.append(long_pitch)
        return result

    @staticmethod
    def analyze_short_helix_pitches(monomer_data, box_size, periodic_boundary):
        """
        Get a list of the pitch of short helices between all actins
        on each filament in each frame of the trajectory.
        """
        result = []
        for t in range(len(monomer_data)):
            helix_pitches = ActinAnalyzer._get_frame_short_helix_pitches(
                monomer_data[t], box_size, periodic_boundary
            )
            result.append(helix_pitches)
        return result

    @staticmethod
    def analyze_long_helix_pitches(monomer_data, box_size, periodic_boundary):
        """
        Get a list of the pitch of long helices between all actins
        on each filament in each frame of the trajectory.
        """
        result = []
        for t in range(len(monomer_data)):
            helix_pitches = ActinAnalyzer._get_frame_long_helix_pitches(
                monomer_data[t], box_size, periodic_boundary
            )
            result.append(helix_pitches)
        return result

    @staticmethod
    def _calculate_line(points, length):
        """
        Use singular value decomposition (first PCA component)
        to calculate a best fit vector along the 3D points.
        """
        center = np.mean(points, axis=0)
        uu, dd, vv = np.linalg.svd(points - center)
        return np.array(
            [center - (length / 2.0) * vv[0], center + (length / 2.0) * vv[0]]
        )

    @staticmethod
    def _get_closest_point_on_line(line, point):
        """
        Get the point on the line closest to the given point.
        """
        lineDir = ReaddyUtil.normalize(line[1] - line[0])
        v = point - line[0]
        d = np.dot(v, lineDir)
        return line[0] + d * lineDir

    @staticmethod
    def _get_frame_distance_from_straight(
        frame_particle_data, box_size, periodic_boundary=True
    ):
        """
        Get the distance from each actin axis position to the ideal axis position
        if the filament axis was a straight line.
        """
        result = []
        filaments = ActinAnalyzer._frame_all_filaments(frame_particle_data)
        for filament in filaments:
            positions = []
            last_pos = frame_particle_data["particles"][filament[0]]["position"]
            for i in range(1, len(filament) - 1):
                actin_ids = [filament[i - 1], filament[i], filament[i + 1]]
                axis_pos = ActinAnalyzer._get_axis_position_for_actin(
                    frame_particle_data, actin_ids, box_size, periodic_boundary
                )
                if axis_pos is None or ReaddyUtil.vector_is_invalid(axis_pos):
                    raise Exception(
                        "Failed to get axis position for actin in filament\n"
                        "tried to use positions: "
                        + ActinAnalyzer.positions_to_string(
                            actin_ids, box_size, frame_particle_data
                        )
                    )
                axis_pos = ReaddyUtil.get_non_periodic_boundary_position(
                    last_pos, axis_pos, box_size
                )
                positions.append(axis_pos)
                last_pos = axis_pos
            if len(positions) > 2:
                axis = ActinAnalyzer._calculate_line(
                    np.squeeze(np.array(positions)), box_size
                )
                for pos in positions:
                    line_pos = ActinAnalyzer._get_closest_point_on_line(axis, pos)
                    result.append(np.linalg.norm(line_pos - pos))
        return result

    @staticmethod
    def analyze_filament_straightness(monomer_data, box_size, periodic_boundary):
        """
        Get a list of the distances from each actin axis position
        to the ideal axis position on each filament in each frame of the trajectory.
        """
        result = []
        for t in range(len(monomer_data)):
            straightness = ActinAnalyzer._get_frame_distance_from_straight(
                monomer_data[t], box_size, periodic_boundary
            )
            result.append(straightness)
        return result

    @staticmethod
    def analyze_free_actin_concentration_over_time(monomer_data, box_size):
        """
        Get an array of the concentration of free actin at each step.
        """
        result = []
        for t in range(len(monomer_data)):
            result.append(
                ReaddyUtil.calculate_concentration(
                    len(
                        ReaddyUtil.analyze_frame_get_ids_for_types(
                            ActinAnalyzer._free_actin_types(),
                            monomer_data[t],
                        )
                    ),
                    box_size,
                )
            )
        return np.array(result)

    @staticmethod
    def analyze_pointed_end_displacement(monomer_data, box_size, periodic_boundary):
        """
        Get the distance the pointed end has moved since it's initial position
        over the time course of a simulation.
        """
        result = []
        initial_pointed_pos = None
        for t in range(len(monomer_data)):
            pointed_id = ReaddyUtil.analyze_frame_get_ids_for_types(
                ActinAnalyzer._pointed_actin_types(), monomer_data[t]
            )[0]
            pointed_position = monomer_data[t]["particles"][pointed_id]["position"]
            if t == 0:
                result.append(0.0)
                initial_pointed_pos = pointed_position
                continue
            if periodic_boundary:
                pointed_position = ReaddyUtil.get_non_periodic_boundary_position(
                    initial_pointed_pos, pointed_position, box_size
                )
            result.append(np.linalg.norm(pointed_position - initial_pointed_pos))
        return np.array(result)

    @staticmethod
    def analyze_normals_and_axis_positions(monomer_data, box_size, periodic_boundary):
        """
        Get the normal vector and axis position
        for each filamentous actin monomer (except ends)
        at each timestep.
        """
        total_steps = len(monomer_data)
        normals = []
        axis_positions = []
        warned = False
        for time_index in range(total_steps):
            normals.append([])
            axis_positions.append([])
            filaments = ActinAnalyzer._frame_all_filaments(monomer_data[time_index])
            for filament in filaments:
                for index in range(1, len(filament) - 1):
                    position = monomer_data[time_index]["particles"][filament[index]][
                        "position"
                    ]
                    actin_ids = [
                        filament[index - 1],
                        filament[index],
                        filament[index + 1],
                    ]
                    axis_pos = ActinAnalyzer._get_axis_position_for_actin(
                        monomer_data[time_index], actin_ids, box_size, periodic_boundary
                    )
                    if ReaddyUtil.vector_is_invalid(axis_pos):
                        if not warned:
                            print(
                                "WARNING: Failed normal calculation: something is "
                                "wrong with actin structure starting at time index "
                                f"= {time_index}"
                            )
                            warned = True
                        axis_positions[time_index].append(None)
                        normals[time_index].append(None)
                        continue
                    if periodic_boundary:
                        axis_pos = ReaddyUtil.get_non_periodic_boundary_position(
                            position, axis_pos, box_size
                        )
                    axis_positions[time_index].append(axis_pos)
                    normals[time_index].append(
                        ReaddyUtil.normalize(position - axis_pos)
                    )
        return normals, axis_positions

    @staticmethod
    def _normal_vectors_are_none(normals, axis_positions, time_i, normal_i):
        """
        Are the normals or axis positions None?.
        """
        return (
            normals[time_i][normal_i] is None
            or axis_positions[time_i][normal_i] is None
        )

    @staticmethod
    def analyze_twist_axis(normals, axis_positions, stride=1):
        """
        Get the angles in degrees from monomer normal to monomer normal
        with normals calculated using the filament axis.
        """
        total_twist = []
        total_twist_tangent_proj = []
        filament_positions = []
        total_steps = len(normals)
        print("Analyzing twist (axis)...")
        for time_i in range(0, total_steps, stride):
            total_twist.append([])
            total_twist_tangent_proj.append([])
            filament_positions.append([])
            new_time = math.floor(time_i / stride)
            max_normals = len(normals[time_i])
            for normal_i in range(max_normals - 2):
                if ActinAnalyzer._normal_vectors_are_none(
                    normals, axis_positions, time_i, normal_i
                ) or ActinAnalyzer._normal_vectors_are_none(
                    normals, axis_positions, time_i, normal_i + 2
                ):
                    total_twist[new_time].append(0.0)
                    total_twist_tangent_proj[new_time].append(0.0)
                    filament_positions[new_time].append(normal_i)
                    continue
                tangent = (
                    axis_positions[time_i][normal_i + 2]
                    - axis_positions[time_i][normal_i]
                )
                normal1 = ReaddyUtil.get_perpendicular_components_of_vector(
                    normals[time_i][normal_i], tangent
                )
                normal2 = ReaddyUtil.get_perpendicular_components_of_vector(
                    normals[time_i][normal_i + 2], tangent
                )
                total_angle_no_bend = ReaddyUtil.get_angle_between_vectors(
                    normal1, normal2, in_degrees=True
                )
                total_angle = ReaddyUtil.get_angle_between_vectors(
                    normals[time_i][normal_i],
                    normals[time_i][normal_i + 2],
                    in_degrees=True,
                )
                if math.isnan(total_angle):
                    total_angle = 0.0
                if math.isnan(total_angle_no_bend):
                    total_angle_no_bend = 0.0
                total_twist[new_time].append(total_angle / 360.0)
                total_twist_tangent_proj[new_time].append(total_angle_no_bend / 360.0)
                filament_positions[new_time].append(normal_i)
        return (
            np.array(total_twist),
            np.array(total_twist_tangent_proj),
            np.array(filament_positions),
        )

    @staticmethod
    def analyze_twist_planes(monomer_data, box_size, periodic_boundary, stride=1):
        """
        Get the twist in degrees between
        the normal vectors to the plane defined
        by each set of 3 actin positions
        for each filamentous actin monomer (except fixed particles)
        at each timestep.
        """
        total_steps = len(monomer_data)
        plane_normals = []
        filament_positions = []
        print("Analyzing twist (planes)...")
        for time_index in range(0, total_steps, stride):
            plane_normals.append([])
            new_time = math.floor(time_index / stride)
            filaments = ActinAnalyzer._frame_all_filaments(monomer_data[time_index])
            for filament in filaments:
                filament_length = len(filament)
                for index in range(1, filament_length - 1):
                    prev_particle = monomer_data[time_index]["particles"][
                        filament[index - 1]
                    ]
                    particle = monomer_data[time_index]["particles"][filament[index]]
                    next_particle = monomer_data[time_index]["particles"][
                        filament[index + 1]
                    ]
                    prev_type_name = prev_particle["type_name"]
                    type_name = particle["type_name"]
                    next_type_name = next_particle["type_name"]
                    if (
                        "fixed" in prev_type_name
                        or "fixed" in type_name
                        or "fixed" in next_type_name
                    ):
                        continue
                    prev_position = prev_particle["position"]
                    position = particle["position"]
                    next_position = next_particle["position"]
                    if periodic_boundary:
                        prev_position = ReaddyUtil.get_non_periodic_boundary_position(
                            position, prev_position, box_size
                        )
                        next_position = ReaddyUtil.get_non_periodic_boundary_position(
                            position, next_position, box_size
                        )
                    v1 = prev_position - position
                    v2 = next_position - position
                    normal = np.cross(v1, v2)
                    normal /= np.linalg.norm(normal)
                    plane_normals[new_time].append(normal)
        twist_angles = []
        for time_index in range(len(plane_normals)):
            twist_angles.append([])
            filament_positions.append([])
            for index in range(0, len(plane_normals[time_index]) - 1, 2):
                # compare every other normal to get the long helix twist
                total_angle = ReaddyUtil.get_angle_between_vectors(
                    plane_normals[time_index][index],
                    plane_normals[time_index][index + 1],
                    in_degrees=True,
                )
                if math.isnan(total_angle):
                    total_angle = 0.0
                twist_angles[time_index].append(total_angle)
                filament_positions[time_index].append(index)
        return np.array(twist_angles), np.array(filament_positions)

    @staticmethod
    def analyze_filament_length(
        normals, axis_positions, box_size, periodic_boundary, stride=1
    ):
        """
        Get the distance in 3D space from the first to last particle
        of the first topology at each timestep.
        """
        filament_length = []
        total_steps = len(axis_positions)
        print("Analyzing filament length...")
        for time_ix in range(0, total_steps, stride):
            pointed_ix = 0
            barbed_ix = len(axis_positions[time_ix]) - 1
            if ActinAnalyzer._normal_vectors_are_none(
                normals, axis_positions, time_ix, pointed_ix
            ) or ActinAnalyzer._normal_vectors_are_none(
                normals, axis_positions, time_ix, barbed_ix
            ):
                filament_length.append(0.0)
                continue
            pointed_pos = axis_positions[time_ix][pointed_ix]
            barbed_pos = axis_positions[time_ix][barbed_ix]
            if periodic_boundary:
                barbed_pos = ReaddyUtil.get_non_periodic_boundary_position(
                    pointed_pos, barbed_pos, box_size
                )
            filament_length.append(
                np.linalg.norm(barbed_pos - pointed_pos)
                + ActinStructure.actin_to_actin_axis_distance
            )
        return np.array(filament_length)

    @staticmethod
    def analyze_bond_stretch(trajectory, box_size, periodic_boundary, stride=1):
        """
        Get the difference in bond length from ideal
        for lateral and longitudinal actin bonds.
        """
        stretch_lat = []
        stretch_long = []
        ideal_length_lat = np.linalg.norm(
            ActinStructure.mother_positions[1] - ActinStructure.mother_positions[0]
        )
        ideal_length_long = np.linalg.norm(
            ActinStructure.mother_positions[2] - ActinStructure.mother_positions[0]
        )
        print("Analyzing bond stretch...")
        for time_index in range(0, len(trajectory), stride):
            stretch_lat.append([])
            stretch_long.append([])
            new_time_index = math.floor(time_index / stride)
            filament = trajectory[time_index].topologies[0].particle_ids
            for index in range(len(filament) - 2):
                particle = trajectory[time_index].particles[filament[index]]
                particle_lat = trajectory[time_index].particles[filament[index + 1]]
                particle_long = trajectory[time_index].particles[filament[index + 2]]
                type_name = particle.type_name
                type_name_lat = particle_lat.type_name
                type_name_long = particle_long.type_name
                if (
                    "fixed" in type_name
                    or "fixed" in type_name_lat
                    or "fixed" in type_name_long
                ):
                    stretch_lat[new_time_index].append(0.0)
                    stretch_long[new_time_index].append(0.0)
                    continue
                pos = particle.position
                pos_lat = particle_lat.position
                pos_long = particle_long.position
                if periodic_boundary:
                    pos_lat = ReaddyUtil.get_non_periodic_boundary_position(
                        pos, pos_lat, box_size
                    )
                    pos_long = ReaddyUtil.get_non_periodic_boundary_position(
                        pos, pos_long, box_size
                    )
                bond_stretch_lat = np.linalg.norm(pos_lat - pos) - ideal_length_lat
                bond_stretch_long = np.linalg.norm(pos_long - pos) - ideal_length_long
                if math.isnan(bond_stretch_lat) or math.isnan(bond_stretch_long):
                    stretch_lat[new_time_index].append(0.0)
                    stretch_long[new_time_index].append(0.0)
                    continue
                stretch_lat[new_time_index].append(bond_stretch_lat)
                stretch_long[new_time_index].append(bond_stretch_long)
        return np.array(stretch_lat), np.array(stretch_long)

    @staticmethod
    def analyze_angle_stretch(trajectory, box_size, periodic_boundary, stride=1):
        """
        Get the difference in angles (degrees) from ideal
        for actin angles between:
        - lateral bonds and lateral bonds
        - lateral bonds and longitudinal bonds
        - longitudinal bonds and longitudinal bonds.
        """
        result_lat_lat = []
        result_lat_long = []
        result_long_long = []
        ideal_angle_lat_lat = ActinStructure.actin_to_actin_angle(True, True, True)
        ideal_angle_lat_long = ActinStructure.actin_to_actin_angle(True, False, True)
        ideal_angle_long_long = ActinStructure.actin_to_actin_angle(False, False, True)
        print("Analyzing angle stretch...")
        for time_index in range(0, len(trajectory), stride):
            result_lat_lat.append([])
            result_lat_long.append([])
            result_long_long.append([])
            new_time_index = math.floor(time_index / stride)
            filament = trajectory[time_index].topologies[0].particle_ids
            for index in range(len(filament) - 4):
                particles = []
                positions = []
                fixed = False
                for d in range(5):
                    particles.append(
                        trajectory[time_index].particles[filament[index + d]]
                    )
                    if "fixed" in particles[d].type_name:
                        fixed = True
                        break
                    positions.append(particles[d].position)
                    if d > 0 and periodic_boundary:
                        positions[d] = ReaddyUtil.get_non_periodic_boundary_position(
                            positions[d - 1], positions[d], box_size
                        )
                if fixed:
                    result_lat_lat[new_time_index].append(0.0)
                    result_lat_long[new_time_index].append(0.0)
                    result_long_long[new_time_index].append(0.0)
                    continue
                stretch_lat_lat = (
                    ReaddyUtil.get_angle_between_vectors(
                        positions[0] - positions[1], positions[2] - positions[1], True
                    )
                    - ideal_angle_lat_lat
                )
                stretch_lat_long = (
                    ReaddyUtil.get_angle_between_vectors(
                        positions[0] - positions[1], positions[3] - positions[1], True
                    )
                    - ideal_angle_lat_long
                )
                stretch_long_long = (
                    ReaddyUtil.get_angle_between_vectors(
                        positions[0] - positions[2], positions[4] - positions[2], True
                    )
                    - ideal_angle_long_long
                )
                if (
                    math.isnan(stretch_lat_lat)
                    or math.isnan(stretch_lat_long)
                    or math.isnan(stretch_long_long)
                ):
                    result_lat_lat[new_time_index].append(0.0)
                    result_lat_long[new_time_index].append(0.0)
                    result_long_long[new_time_index].append(0.0)
                    continue
                result_lat_lat[new_time_index].append(stretch_lat_lat)
                result_lat_long[new_time_index].append(stretch_lat_long)
                result_long_long[new_time_index].append(stretch_long_long)
        return (
            np.array(result_lat_lat),
            np.array(result_lat_long),
            np.array(result_long_long),
        )

    @staticmethod
    def analyze_dihedral_stretch(trajectory, box_size, periodic_boundary, stride=1):
        """
        Get the difference in dihedral angles (degrees) from ideal
        for actin angles between:
        - lateral bonds to lateral bonds to lateral bonds
        - lateral bonds to lateral bonds to longitudinal bonds
        - lateral bonds to longitudinal bonds to longitudinal bonds
        - longitudinal bonds to longitudinal bonds to longitudinal bonds.
        """
        result_lat_lat_lat = []
        result_long_long_long = []
        ideal_angle_lat_lat_lat = ActinStructure.actin_to_actin_dihedral_angle(
            True, True, True, True
        )
        ideal_angle_long_long_long = ActinStructure.actin_to_actin_dihedral_angle(
            False, False, False, True
        )
        print("Analyzing dihedral stretch...")
        for time_index in range(0, len(trajectory), stride):
            result_lat_lat_lat.append([])
            result_long_long_long.append([])
            new_time_index = math.floor(time_index / stride)
            filament = trajectory[time_index].topologies[0].particle_ids
            for index in range(len(filament) - 6):
                particles = []
                positions = []
                fixed = False
                for d in range(7):
                    particles.append(
                        trajectory[time_index].particles[filament[index + d]]
                    )
                    if "fixed" in particles[d].type_name:
                        fixed = True
                        break
                    positions.append(particles[d].position)
                    if d > 0 and periodic_boundary:
                        positions[d] = ReaddyUtil.get_non_periodic_boundary_position(
                            positions[d - 1], positions[d], box_size
                        )
                if fixed:
                    result_lat_lat_lat[new_time_index].append(0.0)
                    result_long_long_long[new_time_index].append(0.0)
                    continue
                stretch_lat_lat_lat = (
                    ReaddyUtil.get_angle_between_vectors(
                        positions[0] - positions[1], positions[3] - positions[2], True
                    )
                    - ideal_angle_lat_lat_lat
                )
                stretch_long_long_long = (
                    ReaddyUtil.get_angle_between_vectors(
                        positions[0] - positions[2], positions[6] - positions[4], True
                    )
                    - ideal_angle_long_long_long
                )
                if math.isnan(stretch_lat_lat_lat) or math.isnan(
                    stretch_long_long_long
                ):
                    result_lat_lat_lat[new_time_index].append(0.0)
                    result_long_long_long[new_time_index].append(0.0)
                    continue
                result_lat_lat_lat[new_time_index].append(stretch_lat_lat_lat)
                result_long_long_long[new_time_index].append(stretch_long_long_long)
        return (np.array(result_lat_lat_lat), np.array(result_long_long_long))
