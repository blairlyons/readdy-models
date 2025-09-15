#!/usr/bin/env python

import numpy as np
import pytest

from simularium_readdy_models.actin import (
    ActinGenerator,
    FiberData,
)
from simularium_readdy_models.tests.conftest import assert_fibers_equal


@pytest.mark.parametrize(
    "fibers, min_extent, max_extent, position_offset, expected_fibers",
    [
        (
            [
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([100.0, 200.0, 200.0]),
                        np.array([316.0, 200.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=2,
                    points=[
                        np.array([50.0, 200.0, 200.0]),
                        np.array([100.0, 200.0, 200.0]),
                        np.array([316.0, 200.0, 200.0]),
                        np.array([400.0, 200.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=3,
                    points=[
                        np.array([100.0, 200.0, 200.0]),
                        np.array([220.0, 200.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=4,
                    points=[
                        np.array([170.0, 200.0, 200.0]),
                        np.array([316.0, 200.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=5,
                    points=[
                        np.array([100.0, 300.0, 200.0]),
                        np.array([200.0, 200.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=6,
                    points=[
                        np.array([200.0, 200.0, 200.0]),
                        np.array([300.0, 300.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=7,
                    points=[
                        np.array([100.0, 300.0, 200.0]),
                        np.array([300.0, 100.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=8,
                    points=[
                        np.array([100.0, 300.0, 200.0]),
                        np.array([200.0, 400.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=9,
                    points=[
                        np.array([170.0, 220.0, 200.0]),
                        np.array([270.0, 220.0, 200.0]),
                        np.array([270.0, 180.0, 200.0]),
                        np.array([200.0, 180.0, 200.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
            ],
            np.array(3 * [150.0]),
            np.array(3 * [250.0]),
            np.array(3 * [-200.0]),
            [
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-50.0, 0.0, 0.0]),
                        np.array([50.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=2,
                    points=[
                        np.array([-50.0, 0.0, 0.0]),
                        np.array([50.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=3,
                    points=[
                        np.array([-50.0, 0.0, 0.0]),
                        np.array([20.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=4,
                    points=[
                        np.array([-30.0, 0.0, 0.0]),
                        np.array([50.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=5,
                    points=[
                        np.array([-50.0, 50.0, 0.0]),
                        np.array([0.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=6,
                    points=[
                        np.array([0.0, 0.0, 0.0]),
                        np.array([50.0, 50.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=7,
                    points=[
                        np.array([-50.0, 50.0, 0.0]),
                        np.array([50.0, -50.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=9,
                    points=[
                        np.array([-30.0, 20.0, 0.0]),
                        np.array([50.0, 20.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=10,
                    points=[
                        np.array([50.0, -20.0, 0.0]),
                        np.array([0.0, -20.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
            ],
        ),
    ],
)
def test_crop_fibers(fibers, min_extent, max_extent, position_offset, expected_fibers):
    cropped_fibers = ActinGenerator.get_cropped_fibers(
        fibers, min_extent, max_extent, position_offset
    )
    assert_fibers_equal(cropped_fibers, expected_fibers)
