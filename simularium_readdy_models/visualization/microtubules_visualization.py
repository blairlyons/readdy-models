#!/usr/bin/env python

import argparse
import os

import numpy as np
from simulariumio import DISPLAY_TYPE, DisplayData, MetaData, ScatterPlotData, UnitData
from simulariumio.readdy import ReaddyConverter, ReaddyData

from ..common import ReaddyUtil
from ..microtubules import (
    MICROTUBULES_REACTIONS,
    MicrotubulesAnalyzer,
    MicrotubulesUtil,
)


class MicrotubulesVisualization:
    """
    visualize a microtubules trajectory in Simularium.
    """

    @staticmethod
    def get_display_data_for_all_polymer_types(raw_display_data):
        """
        creates a dictionary mapping particle type for all polymer types to a value
        for a dictionary of types and values.

        returns dictionary mapping all types to values
        """
        result = {}
        for particle_type in raw_display_data:
            types = MicrotubulesUtil.get_all_polymer_tubulin_types(particle_type)
            for t in types:
                result[t] = raw_display_data[particle_type]
        return result

    @staticmethod
    def get_protofilament_length_plot(monomer_data, times):
        """
        Add a plot of protofilament lengths.
        """
        protofilament_list = MicrotubulesAnalyzer.analyze_protofilament_lengths(
            monomer_data
        )
        max_len = max([len(filament) for filament in protofilament_list])
        reshaped_protofilaments = [
            filament + [0] * (max_len - len(filament))
            for filament in protofilament_list
        ]
        reshaped_protofilaments = np.transpose(np.array(reshaped_protofilaments))
        protofilaments = {}
        for count, protofilament in enumerate(reshaped_protofilaments):
            protofilaments["Filament " + str(count + 1)] = np.array(protofilament)

        return ScatterPlotData(
            title="Length of protofilaments",
            xaxis_title="Time (µs)",
            yaxis_title="Protofilament length (monomers)",
            xtrace=times,
            ytraces=protofilaments,
            render_mode="lines",
        )

    @staticmethod
    def get_avg_microtubule_length_plot(monomer_data, times):
        """
        Add a plot of average microtubule length.
        """
        protofilament_list = MicrotubulesAnalyzer.analyze_protofilament_lengths(
            monomer_data
        )

        mean_lengths = np.array(
            [
                np.nanmean(protofilament) if len(protofilament) else 0
                for protofilament in protofilament_list
            ]
        )
        return ScatterPlotData(
            title="Average microtubule length",
            xaxis_title="Time (µs)",
            yaxis_title="Average microtubule length (monomers)",
            xtrace=times,
            ytraces={"Average microtubule length": mean_lengths},
            render_mode="lines",
        )

    @staticmethod
    def get_growth_reactions_plot(reactions, times):
        """
        Add a plot of reaction events over time
        for each total growth reaction.
        """
        reaction_events = {}
        for total_rxn_name in MICROTUBULES_REACTIONS["Grow"]:
            reaction_counts = ReaddyUtil.analyze_reaction_count_over_time(
                reactions, total_rxn_name
            )
            if reaction_counts is not None:
                reaction_events[total_rxn_name] = reaction_counts
                if "Total Grow" not in reaction_events:
                    tmp = reaction_counts
                else:
                    tmp += reaction_counts
        reaction_events["Total Grow"] = tmp

        return ScatterPlotData(
            title="Growth reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=reaction_events,
            render_mode="lines",
        )

    @staticmethod
    def get_shrink_reactions_plot(reactions, times):
        """
        Add a plot of reaction events over time
        for each total shrink reaction.
        """
        reaction_events = {}

        reaction_counts = ReaddyUtil.analyze_reaction_count_over_time(
            reactions, MICROTUBULES_REACTIONS["MT Shrink"][0]
        )
        if reaction_counts is not None:
            reaction_events[MICROTUBULES_REACTIONS["MT Shrink"][0]] = reaction_counts

        return ScatterPlotData(
            title="Shrink reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=reaction_events,
            render_mode="lines",
        )

    @staticmethod
    def get_attach_reactions_plot(reactions, times):
        """
        Add a plot of attachment events over time
        for each total attach reaction.
        """
        reaction_events = {}
        tmp = None
        for total_rxn_name in MICROTUBULES_REACTIONS["Lateral Attach"]:
            reaction_counts = ReaddyUtil.analyze_reaction_count_over_time(
                reactions, total_rxn_name
            )
            if reaction_counts is not None:
                if tmp is None:
                    tmp = reaction_counts
                else:
                    tmp += reaction_counts
        reaction_events["Total Attach"] = tmp

        return ScatterPlotData(
            title="Attach reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=reaction_events,
            render_mode="lines",
        )

    @staticmethod
    def generate_plots(
        monomer_data,
        reactions,
        times,
    ):
        """
        Use an MicrotubulesAnalyzer to generate plots of observables.
        """
        return {
            "scatter": [
                MicrotubulesVisualization.get_avg_microtubule_length_plot(
                    monomer_data, times
                ),
                MicrotubulesVisualization.get_protofilament_length_plot(
                    monomer_data, times
                ),
                MicrotubulesVisualization.get_growth_reactions_plot(reactions, times),
                MicrotubulesVisualization.get_shrink_reactions_plot(reactions, times),
                MicrotubulesVisualization.get_attach_reactions_plot(reactions, times),
            ],
        }

    @staticmethod
    def visualize_microtubules(path_to_readdy_h5, box_size, scaled_time_step_us, plots):
        """
        visualize a microtubule trajectory in Simularium.
        """
        # radii
        tubulin_radius = 2.0
        display_type = DISPLAY_TYPE.SPHERE
        polymer_display_data = {
            "tubulinA#GTP_": DisplayData(
                name="tubulinA#GTP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinA#GDP_": DisplayData(
                name="tubulinA#GDP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GTP_": DisplayData(
                name="tubulinB#GTP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GDP_": DisplayData(
                name="tubulinB#GDP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinA#GTP_bent_": DisplayData(
                name="tubulinA#GTP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinA#GDP_bent_": DisplayData(
                name="tubulinA#GDP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GTP_bent_": DisplayData(
                name="tubulinB#GTP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GDP_bent_": DisplayData(
                name="tubulinB#GDP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
        }
        display_data = MicrotubulesVisualization.get_display_data_for_all_polymer_types(
            polymer_display_data
        )
        dimer_display_data = {
            "tubulinA#free": DisplayData(
                name="tubulinA#free",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#free": DisplayData(
                name="tubulinB#free",
                radius=tubulin_radius,
                display_type=display_type,
            ),
        }
        display_data = dict(display_data, **dimer_display_data)
        # types to ignore
        ignore_types = [
            # "site#out",
            # "site#1",
            # "site#1_GTP",
            # "site#1_GDP",
            # "site#1_detach",
            # "site#2",
            # "site#2_GTP",
            # "site#2_GDP",
            # "site#2_detach",
            # "site#3",
            # "site#4",
            # "site#4_GTP",
            # "site#4_GDP",
            # "site#new",
            # "site#remove",
            "tubulinA#free",
            "tubulinB#free",
        ]
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=box_size,
            ),
            timestep=scaled_time_step_us,
            path_to_readdy_h5=path_to_readdy_h5,
            # display_data=display_data,
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
            ignore_types=ignore_types,
            plots=[],
        )
        return ReaddyConverter(data)

    @staticmethod
    def add_plots(
        converter,
        plots,
    ):
        if plots is not None:
            for plot_type in plots:
                for plot in plots[plot_type]:
                    converter.add_plot(plot, plot_type)
        return converter

    @staticmethod
    def save(
        converter,
        output_path,
    ):
        converter.save(output_path, binary=True)


def main():
    parser = argparse.ArgumentParser(
        description="Parses a microtuble hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into the Simularium\
         visualization-data-format"
    )
    parser.add_argument(
        "dir_path",
        help="the file path of the directory\
         containing the trajectories to parse",
    )
    parser.add_argument("box_size", help="width of simulation cube")
    args = parser.parse_args()
    dir_path = args.dir_path
    for file in os.listdir(dir_path):
        if file.endswith(".h5"):
            file_path = os.path.join(dir_path, file)
            print(f"visualize {file_path}")
            MicrotubulesVisualization.visualize_microtubules(
                file_path, args.box_size, []
            )


if __name__ == "__main__":
    main()
