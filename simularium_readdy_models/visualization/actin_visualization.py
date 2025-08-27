#!/usr/bin/env python

import argparse
import os

import numpy as np
from simulariumio import (
    DISPLAY_TYPE,
    CameraData,
    DisplayData,
    InputFileData,
    MetaData,
    TrajectoryConverter,
    UnitData,
)
from simulariumio.filters import EveryNthTimestepFilter
from simulariumio.readdy import ReaddyConverter, ReaddyData


class ActinVisualization:
    """
    visualize an actin trajectory in Simularium.
    """

    @staticmethod
    def _display_data() -> dict[str, DisplayData]:
        """
        Get DisplayData for ReaDDy actin simulations.
        """
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        n_polymer_numbers = 5
        result = {
            "actin#free": DisplayData(
                name="actin#free",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#free_ATP": DisplayData(
                name="actin#free",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#new": DisplayData(
                name="actin#new",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#new_ATP": DisplayData(
                name="actin#new",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#branch_1": DisplayData(
                name="actin#branch",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#branch_ATP_1": DisplayData(
                name="actin#ATP_branch",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#branch_barbed_1": DisplayData(
                name="actin#barbed",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#branch_barbed_ATP_1": DisplayData(
                name="actin#barbed_ATP",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#ffdc73",
            ),
            "arp2": DisplayData(
                name="arp#2",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
            "arp2#branched": DisplayData(
                name="arp#2",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
            "arp2#free": DisplayData(
                name="arp#2",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
            "arp3": DisplayData(
                name="arp#3",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
            "arp3#ATP": DisplayData(
                name="arp#3_ATP",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
            "arp3#new": DisplayData(
                name="arp#3",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
            "arp3#new_ATP": DisplayData(
                name="arp#3_ATP",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=actin_radius,
                color="#7230bf",
            ),
        }
        for i in range(1, n_polymer_numbers + 1):
            result.update(
                {
                    f"actin#{i}": DisplayData(
                        name="actin",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#mid_{i}": DisplayData(
                        name="actin#mid",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#fixed_{i}": DisplayData(
                        name="actin#fixed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#mid_fixed_{i}": DisplayData(
                        name="actin#mid_fixed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#ATP_{i}": DisplayData(
                        name="actin#ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#mid_ATP_{i}": DisplayData(
                        name="actin#mid_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#fixed_ATP_{i}": DisplayData(
                        name="actin#fixed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#mid_fixed_ATP_{i}": DisplayData(
                        name="actin#mid_fixed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#barbed_{i}": DisplayData(
                        name="actin#barbed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#barbed_ATP_{i}": DisplayData(
                        name="actin#barbed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#fixed_barbed_{i}": DisplayData(
                        name="actin#fixed_barbed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#fixed_barbed_ATP_{i}": DisplayData(
                        name="actin#fixed_barbed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#pointed_{i}": DisplayData(
                        name="actin#pointed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                    f"actin#pointed_ATP_{i}": DisplayData(
                        name="actin#pointed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                    f"actin#pointed_fixed_{i}": DisplayData(
                        name="actin#pointed_fixed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                    f"actin#pointed_fixed_ATP_{i}": DisplayData(
                        name="actin#pointed_fixed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                },
            )
        return result

    @staticmethod
    def visualize_actin(
        path_to_readdy_h5: str,
        box_size: np.ndarray,
        total_steps: int,
    ):
        """
        Load from ReaDDy outputs and generate a TrajectoryConverter to visualize an
        actin trajectory in Simularium.
        """
        n_timepoints = 1000
        converter = ReaddyConverter(
            ReaddyData(
                timestep=1e-6 * (0.1 * total_steps / float(n_timepoints)),
                path_to_readdy_h5=path_to_readdy_h5,
                meta_data=MetaData(
                    box_size=box_size,
                    camera_defaults=CameraData(
                        position=np.array([0.0, 0.0, 250.0]),
                        look_at_position=np.array([0.0, 0.0, 0.0]),
                        fov_degrees=60.0,
                    ),
                    scale_factor=1.0,
                ),
                display_data=ActinVisualization._display_data(),
                time_units=UnitData("ms"),
                spatial_units=UnitData("nm"),
            )
        )
        time_inc = int(converter._data.agent_data.times.shape[0] / n_timepoints)
        if time_inc >= 2:
            converter._data = converter.filter_data(
                [EveryNthTimestepFilter(n=time_inc)]
            )
        converter.save(output_path=path_to_readdy_h5, validate_ids=False)


def main():
    parser = argparse.ArgumentParser(
        description="Parses an actin hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into Simularium format."
    )
    parser.add_argument(
        "dir_path",
        help="the file path of the directory\
         containing the trajectories to parse",
    )
    parser.add_argument("box_size", help="width of simulation cube")
    parser.add_argument("total_steps", help="total steps run in ReaDDy")
    args = parser.parse_args()
    dir_path = args.dir_path
    for file in os.listdir(dir_path):
        if file.endswith(".h5"):
            file_path = os.path.join(dir_path, file)
            print(f"visualize {file_path}")
            ActinVisualization.visualize_actin(
                file_path, np.array(3 * [float(args.box_size)]), int(args.total_steps)
            )


if __name__ == "__main__":
    main()
