#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time

import numpy as np
import pandas
import psutil

from simularium_readdy_models.actin import (
    FiberData,
    ActinSimulation,
    ActinGenerator,
    ActinTestData,
)
from simularium_readdy_models import ReaddyUtil
from simularium_readdy_models.visualization import ActinVisualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy branched actin simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters"
    )
    parser.add_argument(
        "data_column", help="the column index for the parameter set to use"
    )
    parser.add_argument(
        "model_name", help="prefix for output file names", nargs="?", default=""
    )
    parser.add_argument(
        "replicate", help="which replicate?", nargs="?", default=""
    )
    return parser.parse_args()


def setup_parameters(args):
    parameters = pandas.read_excel(
        args.params_path,
        sheet_name="actin",
        usecols=[0, int(args.data_column)],
        dtype=object,
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name].to_dict()
    parameters["box_size"] = ReaddyUtil.get_box_size(parameters["box_size"])
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = (
        "outputs/" + 
        args.model_name + "_" + 
        str(run_name) + 
        ("_" + args.replicate if args.replicate else "")
    )
    return parameters


def config_init_conditions(actin_simulation):
    actin_simulation.add_obstacles()
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers(use_uuids=False)
    longitudinal_bonds = bool(actin_simulation.parameters.get("longitudinal_bonds", True))
    if bool(actin_simulation.parameters.get("orthogonal_seed", False)):
        print("Starting with orthogonal seed")
        monomers = ActinGenerator.get_monomers(
            fibers_data=[
                FiberData(
                    28,
                    [
                        np.array([-250, 0, 0]),
                        np.array([250, 0, 0]),
                    ],
                    "Actin-Polymer",
                )
            ], 
            use_uuids=False, 
            start_normal=np.array([0., 1., 0.]), 
            longitudinal_bonds=longitudinal_bonds,
        )
        monomers = ActinGenerator.setup_fixed_monomers(monomers, actin_simulation.parameters)
        actin_simulation.add_monomers_from_data(monomers)
    if bool(actin_simulation.parameters.get("branched_seed", False)):
        print("Starting with branched seed")
        actin_simulation.add_monomers_from_data(
            ActinGenerator.get_monomers(
                fibers_data=ActinTestData.simple_branched_actin_fiber(),
                use_uuids=False,
                longitudinal_bonds=longitudinal_bonds,
            )
        )


def report_hardware_usage():
    avg_load = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
    print(
        f"AVG load: {avg_load[0]} last min, {avg_load[1]} last 5 min, {avg_load[2]} last 15 min\n"
        f"RAM % used: {psutil.virtual_memory()[2]}\n"
        f"CPU % used: {psutil.cpu_percent()}\n"
        f"Disk % used: {psutil.disk_usage('/').percent}\n"
    )


def main():
    args = parse_args()
    parameters = setup_parameters(args)
    start_time = time.time()
    actin_simulation = ActinSimulation(
        parameters=parameters, 
        record=True, 
        save_checkpoints=False,
    )
    config_init_conditions(actin_simulation)
    actin_simulation.simulation.run(
        n_steps=int(actin_simulation.parameters["total_steps"]), 
        timestep=actin_simulation.parameters.get("internal_timestep", 0.1),
        show_summary=False,
    )
    print("Run time: %s seconds " % (time.time() - start_time))
    report_hardware_usage()
    ActinVisualization.visualize_actin(
        parameters["name"] + ".h5", 
        parameters["box_size"], 
        parameters["total_steps"], 
    )


if __name__ == "__main__":
    main()
