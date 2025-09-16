from simularium_readdy_models.visualization import (
    MicrotubulesVisualization,
    microtubules_visualization,
)
from simularium_readdy_models.microtubules import MICROTUBULES_REACTIONS
from simularium_readdy_models import ReaddyUtil
from simulariumio import FileConverter, InputFileData
import numpy as np

h5_path = "/Users/blairl/Documents/Dev/simularium-models-util/examples/microtubules/outputs/test_run1.h5"

# box_size = np.array([150.0, 150.0, 250.0])
box_size = np.array([300.0] * 3)
stride = 1
pickle_file_path = h5_path + ".dat"
sim_steps = 10
scaled_time_step_us = 0.1 * 10

(
    monomer_data,
    reactions,
    times,
    _,
) = ReaddyUtil.monomer_data_and_reactions_from_file(
    h5_file_path=h5_path,
    stride=stride,
    timestep=0.1,
    reaction_names=MICROTUBULES_REACTIONS,
    save_pickle_file=True,
    pickle_file_path=pickle_file_path,
)

plots = MicrotubulesVisualization.generate_plots(
    monomer_data=monomer_data,
    reactions=reactions,
    times=times,
)

save_converter = True
load_from_file = False

input_path = h5_path + ".simularium"

if load_from_file:
    converter = FileConverter(InputFileData(file_path=input_path))
else:
    converter = MicrotubulesVisualization.visualize_microtubules(
        h5_path,
        box_size=box_size,
        scaled_time_step_us=scaled_time_step_us,
        plots=plots,
    )
    converter._data = ReaddyUtil._add_edge_agents(
        traj_data=converter._data,
        monomer_data=monomer_data,
        box_size=box_size,
        exclude_types=["tubulinA#free", "tubulinB#free"],
    )
converter = MicrotubulesVisualization.add_plots(
    converter,
    plots,
)
if save_converter:
    MicrotubulesVisualization.save(
        converter,
        output_path=h5_path,
    )
