#!/usr/bin/env python

import numpy as np

from ..common import ReaddyUtil
from .microtubules_util import MicrotubulesUtil


class MicrotubulesAnalyzer:
    @staticmethod
    def get_protofilaments(frame_particle_data, debug_flag=False):
        """
        get a list of the number of monomers in the closed part
        of each protofilament at the time index.
        """
        protofilaments = []

        for particle_id in frame_particle_data["particles"]:
            (
                plus_end_neighbor_id,
                minus_end_neighbor_id,
            ) = MicrotubulesAnalyzer.get_plus_and_minus_end_neighbors(
                particle_id,
                frame_particle_data,
            )

            if plus_end_neighbor_id is None or minus_end_neighbor_id is not None:
                continue
            protofilament = [particle_id]
            if debug_flag:
                import ipdb

                ipdb.set_trace()
            while plus_end_neighbor_id is not None:
                protofilament.append(plus_end_neighbor_id)
                (
                    plus_end_neighbor_id,
                    _,
                ) = MicrotubulesAnalyzer.get_plus_and_minus_end_neighbors(
                    plus_end_neighbor_id,
                    frame_particle_data,
                )
            protofilaments.append(protofilament)
        return protofilaments

    @staticmethod
    def get_plus_and_minus_end_neighbors(
        particle_id,
        frame_particle_data,
    ):
        tubulin_types = MicrotubulesUtil.get_all_tubulin_types()

        particle = frame_particle_data["particles"][particle_id]

        plus_end_neighbor_id = None
        minus_end_neighbor_id = None

        # is this a polymer tubulin?
        if "tubulin" not in particle["type_name"] or "free" in particle["type_name"]:
            return plus_end_neighbor_id, minus_end_neighbor_id

        tubulin_neighbor_ids = ReaddyUtil.analyze_frame_get_neighbor_ids_of_types(
            particle_id=particle_id,
            particle_types=tubulin_types,
            frame_particle_data=frame_particle_data,
            exact_match=False,
        )
        [x_suffix, y_suffix] = MicrotubulesUtil.get_polymer_indices(
            particle["type_name"]
        )

        plus_end_suffixes = MicrotubulesUtil.increment_polymer_indices(
            [x_suffix, y_suffix],
            [1, 0],
        )

        minus_end_suffixes = MicrotubulesUtil.increment_polymer_indices(
            [x_suffix, y_suffix],
            [-1, 0],
        )

        for neighbor_id in tubulin_neighbor_ids:
            neighbor_type = frame_particle_data["particles"][neighbor_id]["type_name"]
            [neighbor_x, neighbor_y] = MicrotubulesUtil.get_polymer_indices(
                neighbor_type
            )
            if (
                "A" in particle["type_name"]
                and "A" in neighbor_type
                or "B" in particle["type_name"]
                and "B" in neighbor_type
            ):
                continue
            if (
                neighbor_x == plus_end_suffixes[0]
                and neighbor_y == plus_end_suffixes[1]
            ):
                plus_end_neighbor_id = neighbor_id
                continue
            if (
                neighbor_x == minus_end_suffixes[0]
                and neighbor_y == minus_end_suffixes[1]
            ):
                minus_end_neighbor_id = neighbor_id
                continue

        return plus_end_neighbor_id, minus_end_neighbor_id

    @staticmethod
    def analyze_protofilament_lengths(monomer_data):
        """
        Get a list of the number of monomers in each mother filament
        in each frame of the trajectory.
        """
        result = []
        # debug_flag = False
        for t in range(len(monomer_data)):
            # print(f"processing time step {t}")
            # debug_flag = t>=3
            debug_flag = False
            if debug_flag:
                import ipdb

                ipdb.set_trace()
            protofilaments = MicrotubulesAnalyzer.get_protofilaments(
                monomer_data[t], debug_flag
            )
            result.append([])
            for filament in protofilaments:
                result[t].append(len(filament))

        return result
