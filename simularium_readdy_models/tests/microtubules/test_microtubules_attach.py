#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np


from simularium_readdy_models.tests.conftest import run_readdy, check_readdy_state
from simularium_readdy_models.tests.microtubules.microtubules_conftest import (
    create_microtubules_simulation,
    default_microtubule_parameters,
    check_for_attach_spatial_reaction,
)


@pytest.mark.parametrize(
    "ring_connections, topology_type, new_edge_tub_ix, expected_monomers, n_GTP_site_tags_after_spatial",
    [
        # # 5 rings x 4 filaments, fully crosslinked, no reaction
        # (
        #     # ring connections: + 2 rings, + 1 filament edges
        #     # (5 rings, only ring bonds shown, filament bonds always true)
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],   # 0
        #         [True, True, True, True, True],   # 1
        #         [True, True, True, True, True],   # 2
        #         [True, True, True, True, True],   # 3
        #         [True, True, True, True, True],   # 4
        #         [True, True, True, True, True], 
        #     ]),
        #     "Microtubule",
        #     None,  # no reaction to form edges
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": range(20),
        #             }
        #         },
        #         "particles": {
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 12]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 17]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
        #         },
        #     }
        # ),
        # # 5 rings x 4 filaments, crosslinked except missing one ring edge, no reaction
        # (
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],   # 0
        #         [True, True, True, True, True],   # 1
        #         [True, True, False, True, True],  # 2
        #         [True, True, True, True, True],   # 3
        #         [True, True, True, True, True],   # 4
        #         [True, True, True, True, True], 
        #     ]), 
        #     "Microtubule",
        #     None,  # no reaction to form edges
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": range(50),
        #             }
        #         },
        #         # type_name format: particle_type_{filament}_{ring}
        #         "particles": {
        #             # particle ids 6,7,8, 11,12,13 will have sites
        #             # available site types are: 
        #             #   out = normal to MT surface
        #             #   1 = towards the previous protofilament (same ring)
        #             #   2 = towards the next protofilament (same ring)
        #             #   3 = towards the minus end (same protofilament)
        #             #   4 = towards the plus end (same protofilament)
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11, 20, 21, 22, 23, 24]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 25, 26, 27, 28, 29]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13, 30, 31, 32, 33, 34]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16, 35, 36, 37, 38, 39]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 17, 40, 41, 42, 43, 44]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18, 45, 46, 47, 48, 49]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
        #             20: {"unique_id": 20, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [6, 21, 22, 23, 24, 25]},
        #             21: {"unique_id": 21, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [6, 20, 23, 24, 26]},
        #             22: {"unique_id": 22, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [6, 20, 23, 24, 27]},
        #             23: {"unique_id": 23, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [6, 20, 21, 22]},
        #             24: {"unique_id": 24, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [6, 20, 21, 22, 28]},
        #             25: {"unique_id": 25, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [7, 20, 26, 27, 28, 29, 30]},
        #             26: {"unique_id": 26, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [7, 21, 25, 28, 29, 31]},
        #             27: {"unique_id": 27, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [7, 22, 25, 28, 29, 32]},
        #             28: {"unique_id": 28, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [7, 25, 26, 27, 24]},
        #             29: {"unique_id": 29, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [7, 25, 26, 27, 33]},
        #             30: {"unique_id": 30, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [8, 25, 31, 32, 33, 34]},
        #             31: {"unique_id": 31, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [8, 26, 30, 33, 34]},
        #             32: {"unique_id": 32, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [8, 27, 30, 33, 34]},
        #             33: {"unique_id": 33, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [8, 30, 31, 32, 29]},
        #             34: {"unique_id": 34, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [8, 30, 31, 32]},
        #             35: {"unique_id": 35, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [11, 36, 37, 38, 39, 40]},
        #             36: {"unique_id": 36, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [11, 35, 38, 39, 41]},
        #             37: {"unique_id": 37, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [11, 35, 38, 39, 42]},
        #             38: {"unique_id": 38, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [11, 35, 36, 37]},
        #             39: {"unique_id": 39, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [11, 35, 36, 37, 43]},
        #             40: {"unique_id": 40, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [12, 35, 41, 42, 43, 44, 45]},
        #             41: {"unique_id": 41, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [12, 36, 40, 43, 44, 46]},
        #             42: {"unique_id": 42, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [12, 37, 40, 43, 44, 47]},
        #             43: {"unique_id": 43, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [12, 40, 41, 42, 39]},
        #             44: {"unique_id": 44, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [12, 40, 41, 42, 48]},
        #             45: {"unique_id": 45, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [13, 40, 46, 47, 48, 49]},
        #             46: {"unique_id": 46, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [13, 41, 45, 48, 49]},
        #             47: {"unique_id": 47, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [13, 42, 45, 48, 49]},
        #             48: {"unique_id": 48, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [13, 45, 46, 47, 44]},
        #             49: {"unique_id": 49, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [13, 45, 46, 47]}               
        #         },
        #     }
        # ),
        # # 5 rings x 4 filaments, crosslinked except missing one ring edge, react to attach missing edge
        # (
        #     # ring connections: + 2 rings, + 1 filament edges
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],   # 0
        #         [True, True, True, True, True],   # 1
        #         [True, True, False, True, True],  # 2
        #         [True, True, True, True, True],   # 3
        #         [True, True, True, True, True],   # 4
        #         [True, True, True, True, True], 
        #     ]), 
        #     "Microtubule#Attaching",
        #     [7, 12],  # add the one missing edge
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": range(20),
        #             }
        #         },
        #         "particles": {
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 12]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 17]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
        #         },
        #     }
        # ),
        # # 5 rings x 4 filaments, at even border of frayed end, react to attach next to border
        (
            np.array([
            # f       0     1     2     3               r
                [True, True, True, True, True], 
                [True, True, True, True, True],        # 0
                [True, True, True, True, True],        # 1
                [False, False, False, False, False],   # 2
                [False, False, False, False, False],   # 3
                [False, False, False, False, False],   # 4
                [False, False, False, False, False], 
            ]), 
            "Microtubule#Attaching",
            [7, 12],  # add edge at the border of frayed end
            {
                "topologies": {
                    0: {
                        "type_name": "Microtubule",
                        "particle_ids": range(100),
                    }
                },
                "particles": {
                    0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
                    1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6, 20, 21, 22, 23, 24]},
                    2: {"unique_id": 2, "type_name": "tubulinA#GTP_bent_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 25, 26, 27, 28, 29]},
                    3: {"unique_id": 3, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 30, 31, 32, 33, 34]},
                    4: {"unique_id": 4, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [3, 35, 36, 37, 38, 39]},
                    5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
                    6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11, 40, 41, 42, 43, 44]},
                    7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 12, 45, 46, 47, 48, 49]},
                    8: {"unique_id": 8, "type_name": "tubulinB#GTP_bent_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 50, 51, 52, 53, 54]},
                    9: {"unique_id": 9, "type_name": "tubulinA#GTP_bent_2_3", "position": np.zeros(3), "neighbor_ids": [8, 55, 56, 57, 58, 59]},
                    10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
                    11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16, 60, 61, 62, 63, 64]},
                    12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 65, 66, 67, 68, 69]},
                    13: {"unique_id": 13, "type_name": "tubulinB#GTP_bent_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 70, 71, 72, 73, 74]},
                    14: {"unique_id": 14, "type_name": "tubulinA#GTP_bent_2_1", "position": np.zeros(3), "neighbor_ids": [13, 75, 76, 77, 78, 79]},
                    15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
                    16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11, 80, 81, 82, 83, 84]},
                    17: {"unique_id": 17, "type_name": "tubulinA#GTP_bent_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 85, 86, 87, 88, 89]},
                    18: {"unique_id": 18, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 90, 91, 92, 93, 94]},
                    19: {"unique_id": 19, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [18, 95, 96, 97, 98, 99]},
                    20: {"unique_id": 20, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [1, 21, 22, 23, 24, 25]},
                    21: {"unique_id": 21, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [1, 20, 23, 24, 26]},
                    22: {"unique_id": 22, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [1, 20, 23, 24, 27]},
                    23: {"unique_id": 23, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [1, 20, 21, 22]},
                    24: {"unique_id": 24, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [1, 20, 21, 22, 28]},
                    25: {"unique_id": 25, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [2, 20, 26, 27, 28, 29, 30]},
                    26: {"unique_id": 26, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [2, 21, 25, 28, 29, 31]},
                    27: {"unique_id": 27, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [2, 22, 25, 28, 29, 32]},
                    28: {"unique_id": 28, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [2, 24, 25, 26, 27]},
                    29: {"unique_id": 29, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [2, 25, 26, 27, 33]},
                    30: {"unique_id": 30, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [3, 25, 31, 32, 33, 34, 35]},
                    31: {"unique_id": 31, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [3, 26, 30, 33, 34, 36]},
                    32: {"unique_id": 32, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [3, 27, 30, 33, 34, 37]},
                    33: {"unique_id": 33, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [3, 29, 30, 31, 32]},
                    34: {"unique_id": 34, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [3, 30, 31, 32, 38]},
                    35: {"unique_id": 35, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [4, 30, 36, 37, 38, 39]},
                    36: {"unique_id": 36, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [4, 31, 35, 38, 39]},
                    37: {"unique_id": 37, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [4, 32, 35, 38, 39]},
                    38: {"unique_id": 38, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [4, 34, 35, 36, 37]},
                    39: {"unique_id": 39, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [4, 35, 36, 37]},
                    40: {"unique_id": 40, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [6, 41, 42, 43, 44, 45]},
                    41: {"unique_id": 41, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [6, 40, 43, 44, 46]},
                    42: {"unique_id": 42, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [6, 40, 43, 44, 47]},
                    43: {"unique_id": 43, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [6, 40, 41, 42]},
                    44: {"unique_id": 44, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [6, 40, 41, 42, 48]},
                    45: {"unique_id": 45, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [7, 40, 46, 47, 48, 49, 50]},
                    46: {"unique_id": 46, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [7, 41, 45, 48, 49, 51]},
                    47: {"unique_id": 47, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [7, 42, 45, 48, 49, 52]},
                    48: {"unique_id": 48, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [7, 44, 45, 46, 47]},
                    49: {"unique_id": 49, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [7, 45, 46, 47, 53]},
                    50: {"unique_id": 50, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [8, 45, 51, 52, 53, 54, 55]},
                    51: {"unique_id": 51, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [8, 46, 50, 53, 54, 56]},
                    52: {"unique_id": 52, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [8, 47, 50, 53, 54, 57]},
                    53: {"unique_id": 53, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [8, 49, 50, 51, 52]},
                    54: {"unique_id": 54, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [8, 50, 51, 52, 58]},
                    55: {"unique_id": 55, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [9, 50, 56, 57, 58, 59]},
                    56: {"unique_id": 56, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [9, 51, 55, 58, 59]},
                    57: {"unique_id": 57, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [9, 52, 55, 58, 59]},
                    58: {"unique_id": 58, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [9, 54, 55, 56, 57]},
                    59: {"unique_id": 59, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [9, 55, 56, 57]},
                    60: {"unique_id": 60, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [11, 61, 62, 63, 64, 65]},
                    61: {"unique_id": 61, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [11, 60, 63, 64, 66]},
                    62: {"unique_id": 62, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [11, 60, 63, 64, 67]},
                    63: {"unique_id": 63, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [11, 60, 61, 62]},
                    64: {"unique_id": 64, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [11, 60, 61, 62, 68]},
                    65: {"unique_id": 65, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [12, 60, 66, 67, 68, 69, 70]},
                    66: {"unique_id": 66, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [12, 61, 65, 68, 69, 71]},
                    67: {"unique_id": 67, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [12, 62, 65, 68, 69, 72]},
                    68: {"unique_id": 68, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [12, 64, 65, 66, 67]},
                    69: {"unique_id": 69, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [12, 65, 66, 67, 73]},
                    70: {"unique_id": 70, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [13, 65, 71, 72, 73, 74, 75]},
                    71: {"unique_id": 71, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [13, 66, 70, 73, 74, 76]},
                    72: {"unique_id": 72, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [13, 67, 70, 73, 74, 77]},
                    73: {"unique_id": 73, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [13, 69, 70, 71, 72]},
                    74: {"unique_id": 74, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [13, 70, 71, 72, 78]},
                    75: {"unique_id": 75, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [14, 70, 76, 77, 78, 79]},
                    76: {"unique_id": 76, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [14, 71, 75, 78, 79]},
                    77: {"unique_id": 77, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [14, 72, 75, 78, 79]},
                    78: {"unique_id": 78, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [14, 74, 75, 76, 77]},
                    79: {"unique_id": 79, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [14, 75, 76, 77]},
                    80: {"unique_id": 80, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [16, 81, 82, 83, 84, 85]},
                    81: {"unique_id": 81, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [16, 80, 83, 84, 86]},
                    82: {"unique_id": 82, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [16, 80, 83, 84, 87]},
                    83: {"unique_id": 83, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [16, 80, 81, 82]},
                    84: {"unique_id": 84, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [16, 80, 81, 82, 88]},
                    85: {"unique_id": 85, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [17, 80, 86, 87, 88, 89, 90]},
                    86: {"unique_id": 86, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [17, 81, 85, 88, 89, 91]},
                    87: {"unique_id": 87, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [17, 82, 85, 88, 89, 92]},
                    88: {"unique_id": 88, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [17, 84, 85, 86, 87]},
                    89: {"unique_id": 89, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [17, 85, 86, 87, 93]},
                    90: {"unique_id": 90, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [18, 85, 91, 92, 93, 94, 95]},
                    91: {"unique_id": 91, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [18, 86, 90, 93, 94, 96]},
                    92: {"unique_id": 92, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [18, 87, 90, 93, 94, 97]},
                    93: {"unique_id": 93, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [18, 89, 90, 91, 92]},
                    94: {"unique_id": 94, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [18, 90, 91, 92, 98]},
                    95: {"unique_id": 95, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [19, 90, 96, 97, 98, 99]},
                    96: {"unique_id": 96, "type_name": "site#1_GTP", "position": np.zeros(3), "neighbor_ids": [19, 91, 95, 98, 99]},
                    97: {"unique_id": 97, "type_name": "site#2_GTP", "position": np.zeros(3), "neighbor_ids": [19, 92, 95, 98, 99]},
                    98: {"unique_id": 98, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [19, 94, 95, 96, 97]},
                    99: {"unique_id": 99, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [19, 95, 96, 97]},
                },
            },
            20,
        ),
        # # 5 rings x 4 filaments, at uneven border of frayed end, react to attach next to border
        # (
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],        # 0
        #         [True, True, True, True, True],        # 1
        #         [False, True, False, False, False],    # 2
        #         [False, False, False, False, False],   # 3
        #         [False, False, False, False, False],   # 4
        #         [False, False, False, False, False], 
        #     ]), 
        #     "Microtubule#Attaching",
        #     [7, 12],  # add edge at the border of frayed end
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": [i for i in range(100) if i < 40 or i >= 45],
        #             }
        #         },
        #         "particles": {
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6, 20, 21, 22, 23, 24]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7, 25, 26, 27, 28, 29]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 30, 31, 32, 33, 34]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [3, 35, 36, 37, 38, 39]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [2, 6, 8, 12, 45, 46, 47, 48, 49]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_bent_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 50, 51, 52, 53, 54]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_bent_2_3", "position": np.zeros(3), "neighbor_ids": [8, 55, 56, 57, 58, 59]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16, 60, 61, 62, 63, 64]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 65, 66, 67, 68, 69]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_bent_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 70, 71, 72, 73, 74]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_bent_2_1", "position": np.zeros(3), "neighbor_ids": [13, 75, 76, 77, 78, 79]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11, 80, 81, 82, 83, 84]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_bent_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 85, 86, 87, 88, 89]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 90, 91, 92, 93, 94]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [18, 95, 96, 97, 98, 99]},
        #             20: {"unique_id": 20, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [1, 21, 22, 23, 24, 25]},
        #             21: {"unique_id": 21, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [1, 20, 23, 24, 26]},
        #             22: {"unique_id": 22, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [1, 20, 23, 24, 27]},
        #             23: {"unique_id": 23, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [1, 20, 21, 22]},
        #             24: {"unique_id": 24, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [1, 20, 21, 22, 28]},
        #             25: {"unique_id": 25, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [2, 20, 26, 27, 28, 29, 30]},
        #             26: {"unique_id": 26, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [2, 21, 25, 28, 29, 31]},
        #             27: {"unique_id": 27, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [2, 22, 25, 28, 29, 32]},
        #             28: {"unique_id": 28, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [2, 24, 25, 26, 27]},
        #             29: {"unique_id": 29, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [2, 25, 26, 27, 33]},
        #             30: {"unique_id": 30, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [3, 25, 31, 32, 33, 34, 35]},
        #             31: {"unique_id": 31, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [3, 26, 30, 33, 34, 36]},
        #             32: {"unique_id": 32, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [3, 27, 30, 33, 34, 37]},
        #             33: {"unique_id": 33, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [3, 29, 30, 31, 32]},
        #             34: {"unique_id": 34, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [3, 30, 31, 32, 38]},
        #             35: {"unique_id": 35, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [4, 30, 36, 37, 38, 39]},
        #             36: {"unique_id": 36, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [4, 31, 35, 38, 39]},
        #             37: {"unique_id": 37, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [4, 32, 35, 38, 39]},
        #             38: {"unique_id": 38, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [4, 34, 35, 36, 37]},
        #             39: {"unique_id": 39, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [4, 35, 36, 37]},
        #             45: {"unique_id": 45, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [7, 46, 47, 48, 49, 50]},
        #             46: {"unique_id": 46, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [7, 45, 48, 49, 51]},
        #             47: {"unique_id": 47, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [7, 45, 48, 49, 52]},
        #             48: {"unique_id": 48, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [7, 45, 46, 47]},
        #             49: {"unique_id": 49, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [7, 45, 46, 47, 53]},
        #             50: {"unique_id": 50, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [8, 45, 51, 52, 53, 54, 55]},
        #             51: {"unique_id": 51, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [8, 46, 50, 53, 54, 56]},
        #             52: {"unique_id": 52, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [8, 47, 50, 53, 54, 57]},
        #             53: {"unique_id": 53, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [8, 49, 50, 51, 52]},
        #             54: {"unique_id": 54, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [8, 50, 51, 52, 58]},
        #             55: {"unique_id": 55, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [9, 50, 56, 57, 58, 59]},
        #             56: {"unique_id": 56, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [9, 51, 55, 58, 59]},
        #             57: {"unique_id": 57, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [9, 52, 55, 58, 59]},
        #             58: {"unique_id": 58, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [9, 54, 55, 56, 57]},
        #             59: {"unique_id": 59, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [9, 55, 56, 57]},
        #             60: {"unique_id": 60, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [11, 61, 62, 63, 64, 65]},
        #             61: {"unique_id": 61, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [11, 60, 63, 64, 66]},
        #             62: {"unique_id": 62, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [11, 60, 63, 64, 67]},
        #             63: {"unique_id": 63, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [11, 60, 61, 62]},
        #             64: {"unique_id": 64, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [11, 60, 61, 62, 68]},
        #             65: {"unique_id": 65, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [12, 60, 66, 67, 68, 69, 70]},
        #             66: {"unique_id": 66, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [12, 61, 65, 68, 69, 71]},
        #             67: {"unique_id": 67, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [12, 62, 65, 68, 69, 72]},
        #             68: {"unique_id": 68, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [12, 64, 65, 66, 67]},
        #             69: {"unique_id": 69, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [12, 65, 66, 67, 73]},
        #             70: {"unique_id": 70, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [13, 65, 71, 72, 73, 74, 75]},
        #             71: {"unique_id": 71, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [13, 66, 70, 73, 74, 76]},
        #             72: {"unique_id": 72, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [13, 67, 70, 73, 74, 77]},
        #             73: {"unique_id": 73, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [13, 69, 70, 71, 72]},
        #             74: {"unique_id": 74, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [13, 70, 71, 72, 78]},
        #             75: {"unique_id": 75, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [14, 70, 76, 77, 78, 79]},
        #             76: {"unique_id": 76, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [14, 71, 75, 78, 79]},
        #             77: {"unique_id": 77, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [14, 72, 75, 78, 79]},
        #             78: {"unique_id": 78, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [14, 74, 75, 76, 77]},
        #             79: {"unique_id": 79, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [14, 75, 76, 77]},
        #             80: {"unique_id": 80, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [16, 81, 82, 83, 84, 85]},
        #             81: {"unique_id": 81, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [16, 80, 83, 84, 86]},
        #             82: {"unique_id": 82, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [16, 80, 83, 84, 87]},
        #             83: {"unique_id": 83, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [16, 80, 81, 82]},
        #             84: {"unique_id": 84, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [16, 80, 81, 82, 88]},
        #             85: {"unique_id": 85, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [17, 80, 86, 87, 88, 89, 90]},
        #             86: {"unique_id": 86, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [17, 81, 85, 88, 89, 91]},
        #             87: {"unique_id": 87, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [17, 82, 85, 88, 89, 92]},
        #             88: {"unique_id": 88, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [17, 84, 85, 86, 87]},
        #             89: {"unique_id": 89, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [17, 85, 86, 87, 93]},
        #             90: {"unique_id": 90, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [18, 85, 91, 92, 93, 94, 95]},
        #             91: {"unique_id": 91, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [18, 86, 90, 93, 94, 96]},
        #             92: {"unique_id": 92, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [18, 87, 90, 93, 94, 97]},
        #             93: {"unique_id": 93, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [18, 89, 90, 91, 92]},
        #             94: {"unique_id": 94, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [18, 90, 91, 92, 98]},
        #             95: {"unique_id": 95, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [19, 90, 96, 97, 98, 99]},
        #             96: {"unique_id": 96, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [19, 91, 95, 98, 99]},
        #             97: {"unique_id": 97, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [19, 92, 95, 98, 99]},
        #             98: {"unique_id": 98, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [19, 94, 95, 96, 97]},
        #             99: {"unique_id": 99, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [19, 95, 96, 97]},
        #         },
        #     }
        # ),
        # # 5 rings x 4 filaments, at uneven border of frayed end, react to attach next to border
        # (
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],        # 0
        #         [True, True, True, True, True],        # 1
        #         [True, True, False, False, False],     # 2
        #         [False, False, False, False, False],   # 3
        #         [False, False, False, False, False],   # 4
        #         [False, False, False, False, False], 
        #     ]), 
        #     "Microtubule#Attaching",
        #     [12, 17],  # add edge at the border of frayed end
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": range(95),
        #             }
        #         },
        #         "particles": {
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7, 20, 21, 22, 23, 24]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 25, 26, 27, 28, 29]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [3, 30, 31, 32, 33, 34]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11, 35, 36, 37, 38, 39]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [2, 6, 8, 40, 41, 42, 43, 44]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_bent_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 45, 46, 47, 48, 49]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_bent_2_3", "position": np.zeros(3), "neighbor_ids": [8, 50, 51, 52, 53, 54]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16, 55, 56, 57, 58, 59]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 17, 60, 61, 62, 63, 64]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_bent_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 65, 66, 67, 68, 69]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_bent_2_1", "position": np.zeros(3), "neighbor_ids": [13, 70, 71, 72, 73, 74]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11, 75, 76, 77, 78, 79]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [12, 16, 18, 80, 81, 82, 83, 84]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 85, 86, 87, 88, 89]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [18, 90, 91, 92, 93, 94]},
        #             20: {"unique_id": 20, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [2, 21, 22, 23, 24, 25]},
        #             21: {"unique_id": 21, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [2, 20, 23, 24, 26]},
        #             22: {"unique_id": 22, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [2, 20, 23, 24, 27]},
        #             23: {"unique_id": 23, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [2, 20, 21, 22]},
        #             24: {"unique_id": 24, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [2, 20, 21, 22, 28]},
        #             25: {"unique_id": 25, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [3, 20, 26, 27, 28, 29, 30]},
        #             26: {"unique_id": 26, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [3, 21, 25, 28, 29, 31]},
        #             27: {"unique_id": 27, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [3, 22, 25, 28, 29, 32]},
        #             28: {"unique_id": 28, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [3, 24, 25, 26, 27]},
        #             29: {"unique_id": 29, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [3, 25, 26, 27, 33]},
        #             30: {"unique_id": 30, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [4, 25, 31, 32, 33, 34]},
        #             31: {"unique_id": 31, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [4, 26, 30, 33, 34]},
        #             32: {"unique_id": 32, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [4, 27, 30, 33, 34]},
        #             33: {"unique_id": 33, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [4, 29, 30, 31, 32]},
        #             34: {"unique_id": 34, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [4, 30, 31, 32]},
        #             35: {"unique_id": 35, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [6, 36, 37, 38, 39, 40]},
        #             36: {"unique_id": 36, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [6, 35, 38, 39, 41]},
        #             37: {"unique_id": 37, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [6, 35, 38, 39, 42]},
        #             38: {"unique_id": 38, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [6, 35, 36, 37]},
        #             39: {"unique_id": 39, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [6, 35, 36, 37, 43]},
        #             40: {"unique_id": 40, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [7, 35, 41, 42, 43, 44, 45]},
        #             41: {"unique_id": 41, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [7, 36, 40, 43, 44, 46]},
        #             42: {"unique_id": 42, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [7, 37, 40, 43, 44, 47]},
        #             43: {"unique_id": 43, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [7, 39, 40, 41, 42]},
        #             44: {"unique_id": 44, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [7, 40, 41, 42, 48]},
        #             45: {"unique_id": 45, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [8, 40, 46, 47, 48, 49, 50]},
        #             46: {"unique_id": 46, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [8, 41, 45, 48, 49, 51]},
        #             47: {"unique_id": 47, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [8, 42, 45, 48, 49, 52]},
        #             48: {"unique_id": 48, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [8, 44, 45, 46, 47]},
        #             49: {"unique_id": 49, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [8, 45, 46, 47, 53]},
        #             50: {"unique_id": 50, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [9, 45, 51, 52, 53, 54]},
        #             51: {"unique_id": 51, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [9, 46, 50, 53, 54]},
        #             52: {"unique_id": 52, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [9, 47, 50, 53, 54]},
        #             53: {"unique_id": 53, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [9, 49, 50, 51, 52]},
        #             54: {"unique_id": 54, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [9, 50, 51, 52]},
        #             55: {"unique_id": 55, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [11, 56, 57, 58, 59, 60]},
        #             56: {"unique_id": 56, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [11, 55, 58, 59, 61]},
        #             57: {"unique_id": 57, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [11, 55, 58, 59, 62]},
        #             58: {"unique_id": 58, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [11, 55, 56, 57]},
        #             59: {"unique_id": 59, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [11, 55, 56, 57, 63]},
        #             60: {"unique_id": 60, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [12, 55, 61, 62, 63, 64, 65]},
        #             61: {"unique_id": 61, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [12, 56, 60, 63, 64, 66]},
        #             62: {"unique_id": 62, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [12, 57, 60, 63, 64, 67]},
        #             63: {"unique_id": 63, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [12, 59, 60, 61, 62]},
        #             64: {"unique_id": 64, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [12, 60, 61, 62, 68]},
        #             65: {"unique_id": 65, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [13, 60, 66, 67, 68, 69, 70]},
        #             66: {"unique_id": 66, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [13, 61, 65, 68, 69, 71]},
        #             67: {"unique_id": 67, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [13, 62, 65, 68, 69, 72]},
        #             68: {"unique_id": 68, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [13, 64, 65, 66, 67]},
        #             69: {"unique_id": 69, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [13, 65, 66, 67, 73]},
        #             70: {"unique_id": 70, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [14, 65, 71, 72, 73, 74]},
        #             71: {"unique_id": 71, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [14, 66, 70, 73, 74]},
        #             72: {"unique_id": 72, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [14, 67, 70, 73, 74]},
        #             73: {"unique_id": 73, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [14, 69, 70, 71, 72]},
        #             74: {"unique_id": 74, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [14, 70, 71, 72]},
        #             75: {"unique_id": 75, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [16, 76, 77, 78, 79, 80]},
        #             76: {"unique_id": 76, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [16, 75, 78, 79, 81]},
        #             77: {"unique_id": 77, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [16, 75, 78, 79, 82]},
        #             78: {"unique_id": 78, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [16, 75, 76, 77]},
        #             79: {"unique_id": 79, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [16, 75, 76, 77, 83]},
        #             80: {"unique_id": 80, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [17, 75, 81, 82, 83, 84, 85]},
        #             81: {"unique_id": 81, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [17, 76, 80, 83, 84, 86]},
        #             82: {"unique_id": 82, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [17, 77, 80, 83, 84, 87]},
        #             83: {"unique_id": 83, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [17, 79, 80, 81, 82]},
        #             84: {"unique_id": 84, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [17, 80, 81, 82, 88]},
        #             85: {"unique_id": 85, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [18, 80, 86, 87, 88, 89, 90]},
        #             86: {"unique_id": 86, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [18, 81, 85, 88, 89, 91]},
        #             87: {"unique_id": 87, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [18, 82, 85, 88, 89, 92]},
        #             88: {"unique_id": 88, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [18, 84, 85, 86, 87]},
        #             89: {"unique_id": 89, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [18, 85, 86, 87, 93]},
        #             90: {"unique_id": 90, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [19, 85, 91, 92, 93, 94]},
        #             91: {"unique_id": 91, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [19, 86, 90, 93, 94]},
        #             92: {"unique_id": 92, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [19, 87, 90, 93, 94]},
        #             93: {"unique_id": 93, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [19, 89, 90, 91, 92]},
        #             94: {"unique_id": 94, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [19, 90, 91, 92]},
        #         },
        #     }
        # ),
        # # 5 rings x 4 filaments, at uneven border of frayed end, react to attach entire new ring at border
        # (
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],        # 0
        #         [True, True, True, True, True],        # 1
        #         [True, True, False, True, True],       # 2
        #         [False, False, False, False, False],   # 3
        #         [False, False, False, False, False],   # 4
        #         [False, False, False, False, False], 
        #     ]), 
        #     "Microtubule#Attaching",
        #     [7, 12],  # add edge at the border of frayed end
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": [i for i in range(90) if i < 35 or (i >= 40 and i < 55) or i >= 60],
        #             }
        #         },
        #         "particles": {
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7, 20, 21, 22, 23, 24]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 25, 26, 27, 28, 29]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [3, 30, 31, 32, 33, 34]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [2, 6, 8, 12, 40, 41, 42, 43, 44]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_bent_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 45, 46, 47, 48, 49]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_bent_2_3", "position": np.zeros(3), "neighbor_ids": [8, 50, 51, 52, 53, 54]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [7, 11, 13, 17, 60, 61, 62, 63, 64]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_bent_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 65, 66, 67, 68, 69]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_bent_2_1", "position": np.zeros(3), "neighbor_ids": [13, 70, 71, 72, 73, 74]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [12, 16, 18, 75, 76, 77, 78, 79]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_bent_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 80, 81, 82, 83, 84]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_bent_2_2", "position": np.zeros(3), "neighbor_ids": [18, 85, 86, 87, 88, 89]},
        #             20: {"unique_id": 20, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [2, 21, 22, 23, 24, 25]},
        #             21: {"unique_id": 21, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [2, 20, 23, 24, 26]},
        #             22: {"unique_id": 22, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [2, 20, 23, 24, 27]},
        #             23: {"unique_id": 23, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [2, 20, 21, 22]},
        #             24: {"unique_id": 24, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [2, 20, 21, 22, 28]},
        #             25: {"unique_id": 25, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [3, 20, 26, 27, 28, 29, 30]},
        #             26: {"unique_id": 26, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [3, 21, 25, 28, 29, 31]},
        #             27: {"unique_id": 27, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [3, 22, 25, 28, 29, 32]},
        #             28: {"unique_id": 28, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [3, 24, 25, 26, 27]},
        #             29: {"unique_id": 29, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [3, 25, 26, 27, 33]},
        #             30: {"unique_id": 30, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [4, 25, 31, 32, 33, 34]},
        #             31: {"unique_id": 31, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [4, 26, 30, 33, 34]},
        #             32: {"unique_id": 32, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [4, 27, 30, 33, 34]},
        #             33: {"unique_id": 33, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [4, 29, 30, 31, 32]},
        #             34: {"unique_id": 34, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [4, 30, 31, 32]},
        #             40: {"unique_id": 40, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [7, 41, 42, 43, 44, 45]},
        #             41: {"unique_id": 41, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [7, 40, 43, 44, 46]},
        #             42: {"unique_id": 42, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [7, 40, 43, 44, 47]},
        #             43: {"unique_id": 43, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [7, 40, 41, 42]},
        #             44: {"unique_id": 44, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [7, 40, 41, 42, 48]},
        #             45: {"unique_id": 45, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [8, 40, 46, 47, 48, 49, 50]},
        #             46: {"unique_id": 46, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [8, 41, 45, 48, 49, 51]},
        #             47: {"unique_id": 47, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [8, 42, 45, 48, 49, 52]},
        #             48: {"unique_id": 48, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [8, 44, 45, 46, 47]},
        #             49: {"unique_id": 49, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [8, 45, 46, 47, 53]},
        #             50: {"unique_id": 50, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [9, 45, 51, 52, 53, 54]},
        #             51: {"unique_id": 51, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [9, 46, 50, 53, 54]},
        #             52: {"unique_id": 52, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [9, 47, 50, 53, 54]},
        #             53: {"unique_id": 53, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [9, 49, 50, 51, 52]},
        #             54: {"unique_id": 54, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [9, 50, 51, 52]},
        #             60: {"unique_id": 60, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [12, 61, 62, 63, 64, 65]},
        #             61: {"unique_id": 61, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [12, 60, 63, 64, 66]},
        #             62: {"unique_id": 62, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [12, 60, 63, 64, 67]},
        #             63: {"unique_id": 63, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [12, 60, 61, 62]},
        #             64: {"unique_id": 64, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [12, 60, 61, 62, 68]},
        #             65: {"unique_id": 65, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [13, 60, 66, 67, 68, 69, 70]},
        #             66: {"unique_id": 66, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [13, 61, 65, 68, 69, 71]},
        #             67: {"unique_id": 67, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [13, 62, 65, 68, 69, 72]},
        #             68: {"unique_id": 68, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [13, 64, 65, 66, 67]},
        #             69: {"unique_id": 69, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [13, 65, 66, 67, 73]},
        #             70: {"unique_id": 70, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [14, 65, 71, 72, 73, 74]},
        #             71: {"unique_id": 71, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [14, 66, 70, 73, 74]},
        #             72: {"unique_id": 72, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [14, 67, 70, 73, 74]},
        #             73: {"unique_id": 73, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [14, 69, 70, 71, 72]},
        #             74: {"unique_id": 74, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [14, 70, 71, 72]},
        #             75: {"unique_id": 75, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [17, 76, 77, 78, 79, 80]},
        #             76: {"unique_id": 76, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [17, 75, 78, 79, 81]},
        #             77: {"unique_id": 77, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [17, 75, 78, 79, 82]},
        #             78: {"unique_id": 78, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [17, 75, 76, 77]},
        #             79: {"unique_id": 79, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [17, 75, 76, 77, 83]},
        #             80: {"unique_id": 80, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [18, 75, 81, 82, 83, 84, 85]},
        #             81: {"unique_id": 81, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [18, 76, 80, 83, 84, 86]},
        #             82: {"unique_id": 82, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [18, 77, 80, 83, 84, 87]},
        #             83: {"unique_id": 83, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [18, 79, 80, 81, 82]},
        #             84: {"unique_id": 84, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [18, 80, 81, 82, 88]},
        #             85: {"unique_id": 85, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [19, 80, 86, 87, 88, 89]},
        #             86: {"unique_id": 86, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [19, 81, 85, 88, 89]},
        #             87: {"unique_id": 87, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [19, 82, 85, 88, 89]},
        #             88: {"unique_id": 88, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [19, 84, 85, 86, 87]},
        #             89: {"unique_id": 89, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [19, 85, 86, 87]},
        #         },
        #     }
        # ),
        # #
    ],
)
def test_attach(ring_connections, topology_type, new_edge_tub_ix, expected_monomers, n_GTP_site_tags_after_spatial):
    parameters = default_microtubule_parameters
    parameters["ring_attach_GTP_rate"] = 1e30
    parameters["attach_reaction_distance"] = 5.
    mt_simulation = create_microtubules_simulation(parameters, ring_connections, topology_type, new_edge_tub_ix)
    run_readdy(1, mt_simulation.system, mt_simulation.simulation)
    check_readdy_state(mt_simulation, expected_monomers)
    run_readdy(1, mt_simulation.system, mt_simulation.simulation)
    check_readdy_state(mt_simulation, expected_monomers, ignore_extra_spatial_rxn=True)
    check_for_attach_spatial_reaction(n_GTP_site_tags_after_spatial, mt_simulation)
    del mt_simulation
