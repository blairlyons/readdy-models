# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import pytest
from .actin_conftest import actin_parameters_rxns_off

# from simularium_readdy_models.actin import ActinSimulation
# from simularium_readdy_models.tests.conftest import (
#     actin_parameters_rxns_off,
#     assert_monomers_equal,
#     actin_dimer,
# )


# def reverse_dimerize_parameters():
#     result = actin_parameters_rxns_off
#     result["dimerize_reverse_rate"] = 1e30
#     return result


# @pytest.mark.parametrize(
#     "parameters, start_monomers, expected_monomers",
#     [
#         (
#             reverse_dimerize_parameters(),
#             actin_dimer(),
#             actin_dimer(),  # TODO get reaction to happen
#         ),
#     ],
# )
# def test_generate_monomers(parameters, start_monomers, expected_monomers):
#     actin_simulation = ActinSimulation(parameters)
#     actin_simulation.add_monomers_from_data(start_monomers)
#     actin_simulation.simulate(0.000000001)  # 10 steps
#     monomers = actin_simulation.get_current_monomers()
#     assert_monomers_equal(monomers, expected_monomers)
