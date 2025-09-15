# Parameters in actin model

## "name"
default value: "actin"

units: NA

Output file name.

## "total_steps"
default value: 1e3

units: count

Total number of steps to run in ReaDDy.

## "time_step"
default value: 0.0000001

units: s

Time step for Vivarium process.

## "internal_timestep"
default value: 0.1

units: ns

Time that passes in one ReaDDy timestep.

## "box_size"
default value: [500.0, 500.0, 500.0]

units: nm

Dimensions of the simulation box in nanometers. If one value is given (e.g. 500.0) that value will be used for all three dimensions.

## "periodic_boundary"
default value: True

units: NA

Use a periodic boundary at the edges of the simulation box? If True, particles will reflect to the other side of the box when they cross the boundary. If False, particles will be constrained to the edges of the box with a box potential.

## "reaction_distance"
default value: 1.0

units: nm

Extra distance between particles where a reaction can be triggered. Larger values increase reaction events at the expense of spatial accuracy.

## "n_cpu"
default value: 4

units: count

How many CPU cores to use?

## "actin_concentration"
default value: 200.0

units: uM

Concentration of free actin monomers to add in the initial conditions.

## "arp23_concentration"
default value: 10.0

units: uM

Concentration of free Arp23 monomers to add in the initial conditions.

## "cap_concentration"
default value: 0.0

units: uM

Concentration of free capping protein molecules to add in the initial conditions.

## "seed_n_fibers"
default value: 0

units: count

How many randomly oriented linear actin fibers to seed the simulation with in the initial conditions?

## "seed_fiber_length"
default value: 0.0

units: nm

Length of random linear actin seed fibers.

## "orthogonal_seed"
default value: False

units: NA

Add a seed actin fiber along the x-axis?

## "orthogonal_seed_length"
default value: 50.0

units: nm

Length of seed actin fiber along x-axis.

## "branched_seed"
default value: False

units: NA

Add a branched actin molecule in the initial conditions?

## "only_linear_actin_constraints"
default value: False

units: NA

If true, only add constraints for linear actin, skip adding branch constraints. Skipping these can speed up simulation initialization and calculation if branching is not used.

## "reactions"
default value: True

units: NA

Evaluate reactions? If false, no reactions will occur. This can speed up simulation calculation if reactions are not needed.

## "dimerize_rate"
default value: 2.1e-2

units: 1/ns

Rate of dimerization reaction between two free actin monomers. Turning this off (using a very small rate like 1e-30) can prevent actin monomers from being used up in dimerization reactions, which would normally be regulated by other molecules in vivo.

## "dimerize_reverse_rate"
default value: 1.4e-9

units: 1/ns

Rate of decay of a fiber of length 2 monomers. 

## "trimerize_rate"
default value: 2.1e-2

units: 1/ns

Rate of trimerization reaction between a free actin monomer and a fiber of length 2 monomers. 

## "trimerize_reverse_rate"
default value: 1.4e-9

units: 1/ns

Rate of decay of a fiber of length 3 monomers. 

## "pointed_growth_ATP_rate"
default value: 2.4e-5

units: 1/ns

Rate of growth of the pointed end of actin fibers of any ATP state reacting with an ATP-actin monomer.

## "pointed_growth_ADP_rate"
default value: 2.95e-6

units: 1/ns

Rate of growth of the pointed end of actin fibers of any ATP state reacting with an ADP-actin monomer.

## "pointed_shrink_ATP_rate"
default value: 8.0e-10

units: 1/ns

Rate of loss of a monomer from the pointed end of actin fibers where the pointed end monomer is bound with ATP.

## "pointed_shrink_ADP_rate"
default value: 3.0e-10

units: 1/ns

Rate of loss of a monomer from the pointed end of actin fibers where the pointed end monomer is bound with ADP.

## "barbed_growth_ATP_rate"
default value: 2.1e-2

units: 1/ns

Rate of growth of the barbed end of actin fibers of any ATP state reacting with an ATP-actin monomer.

## "barbed_growth_ADP_rate"
default value: 7.0e-5

units: 1/ns

Rate of growth of the barbed end of actin fibers of any ATP state reacting with an ADP-actin monomer.

## "nucleate_ATP_rate"
default value: 2.1e-2

units: 1/ns

Rate of nucleation (formation of a 4-monomer fiber) reacting with an ATP-actin monomer.

## "nucleate_ADP_rate"
default value: 7.0e-5

units: 1/ns

Rate of nucleation (formation of a 4-monomer fiber) reacting with an ADP-actin monomer.

## "barbed_shrink_ATP_rate"
default value: 1.4e-9

units: 1/ns

Rate of loss of a monomer from the barbed end of actin fibers where the barbed end monomer is bound with ATP.

## "barbed_shrink_ADP_rate"
default value: 8.0e-9

units: 1/ns

Rate of loss of a monomer from the barbed end of actin fibers where the barbed end monomer is bound with ADP.

## "arp_bind_ATP_rate"
default value: 2.1e-2

units: 1/ns

Rate of binding of an Arp23 dimer to an ATP-actin monomer in an actin fiber.

## "arp_bind_ADP_rate"
default value: 7.0e-5

units: 1/ns

Rate of binding of an Arp23 dimer to an ADP-actin monomer in an actin fiber.

## "arp_unbind_ATP_rate"
default value: 1.4e-9

units: 1/ns

Rate of dissociation of an ATP-Arp23 dimer from an actin fiber.

## "arp_unbind_ADP_rate"
default value: 8.0e-9

units: 1/ns

Rate of dissociation of an ADP-Arp23 dimer from an actin fiber.

## "barbed_growth_branch_ATP_rate"
default value: 2.1e-2

units: 1/ns

Rate of binding of an ATP-actin monomer to an Arp23 bound on an actin fiber.

## "barbed_growth_branch_ADP_rate"
default value: 7.0e-5

units: 1/ns

Rate of binding of an ADP-actin monomer to an Arp23 bound on an actin fiber.

## "debranching_ATP_rate"
default value: 1.4e-9

units: 1/ns

Rate of dissociation of an actin monomer from an ATP-Arp23 bound on an actin fiber.

## "debranching_ADP_rate"
default value: 7.0e-5

units: 1/ns

Rate of dissociation of an actin monomer from an ADP-Arp23 bound on an actin fiber.

## "cap_bind_rate"
default value: 2.1e-2

units: 1/ns

Rate of binding of capping protein to the barbed end of an actin fiber.

## "cap_unbind_rate"
default value: 1.4e-9

units: 1/ns

Rate of dissociation of capping protein from the barbed end of an actin fiber.

## "hydrolysis_actin_rate"
default value: 3.5e-5

units: 1/ns

Rate of hydrolysis of ATP bound to actin monomers in a fiber.

## "hydrolysis_arp_rate"
default value: 3.5e-5

units: 1/ns

Rate of hydrolysis of ATP bound to Arp23 in a fiber.

## "nucleotide_exchange_actin_rate"
default value: 1e-5

units: 1/ns

Rate of exchange of an ADP for an ATP in a free actin monomer.

## "nucleotide_exchange_arp_rate"
default value: 1e-5

units: 1/ns

Rate of exchange of an ADP for an ATP in a free Arp23 monomer.

## "verbose"
default value: False

units: NA

Print log statements including for each reaction.

## "use_box_actin"
default value: False

units: NA

Confine free actin monomers to a box. Also requires parameters "actin_box_center_x", "actin_box_center_y", "actin_box_center_z", "actin_box_size_x", "actin_box_size_y", "actin_box_size_z".

## "use_box_arp"
default value: False

units: NA

Confine free Arp23 dimers to a box. Also requires parameters "arp_box_center_x", "arp_box_center_y", "arp_box_center_z", "arp_box_size_x", "arp_box_size_y", "arp_box_size_z".

## "use_box_cap"
default value: False

units: NA

Confine free capping protein to a box. Also requires parameters "cap_box_center_x", "cap_box_center_y", "cap_box_center_z", "cap_box_size_x", "cap_box_size_y", "cap_box_size_z".

## "add_obstacles"
default value: False

units: NA

Add obstacle particles, which take up space but don't react?

## "obstacle_radius"
default value: 35.0

units: nm

Radius of obstacle particles, which take up space but don't react.

## "obstacle_diff_coeff"
default value: 0.0

units: nm^2/s

Diffusion coefficient of obstacle particles, which take up space but don't react.

## "use_box_obstacle"
default value: False

units: NA

Confine obstacle particles to a box. Also requires parameters "obstacle_box_center_x", "obstacle_box_center_y", "obstacle_box_center_z", "obstacle_box_size_x", "obstacle_box_size_y", "obstacle_box_size_z".

## "position_obstacle_stride"
default value: 0

units: count

Number of frames between each time the obstacle position is set to the controlled position. Also requires parameters "obstacle_controlled_position_x", "obstacle_controlled_position_y", "obstacle_controlled_position_z"

## "n_fixed_monomers_pointed"
default value: 0

units: count

Number of monomers at the pointed end of the orthogonal seed fiber to freeze in space.

## "n_fixed_monomers_barbed"
default value: 0

units: count

Number of monomers at the barbed end of the orthogonal seed fiber to freeze in space.

## "displace_pointed_end_tangent"
default value: False

units: NA

Displace the pointed end of the orthogonal seed fiber along the positive X-axis? Also requires parameter "tangent_displacement_nm", "displace_stride".

## "displace_pointed_end_radial"
default value: False

units: NA

Displace the pointed end of the orthogonal seed fiber in an arc with a given radius? Also requires parameters "radial_displacement_radius_nm", "radial_displacement_angle_deg", "displace_stride".

## "plot_polymerization"
default value: False

units: NA

Plot polymerization rate data in Simularium file?

## "plot_filament_structure"
default value: False

units: NA

Plot filament structure (angles, lengths, etc) data in Simularium file?

## "plot_bend_twist"
default value: False

units: NA

Plot actin fiber bend and twist data in Simularium file?

## "plot_actin_compression"
default value: False

units: NA

Plot actin fiber compression data in Simularium file?

## "visualize_edges"
default value: False

units: NA

Visualize lines for each bond edge in the Simularium file?

## "visualize_normals"
default value: False

units: NA

Visualize lines for each bound actin monomer's normal in the Simularium file?

## "visualize_control_pts"
default value: False

units: NA

Visualize points for each actin monomer's nearest position on the fiber backbone in the Simularium file?

## "longitudinal_bonds"
default value: True

units: NA

Create bonds between every other consecutive actin monomer in a fiber?

## "bonds_force_multiplier"
default value: 0.2

units: None

Multiplier to use for force constants enforcing bonds.

## "angles_force_constant"
default value: 1000.0

units: None

Multiplier to use for force constants enforcing angle constraints.

## "dihedrals_force_constant"
default value: 1000.0

units: None

Multiplier to use for force constants enforcing cosine dihedral constraints.

## "add_membrane"
default value: False

units: NA

Add a coarse grained particle membrane?

## "actin_constraints"
default value: True

units: NA

Enforce angle and dihedral constraintes to maintain actin fiber structure? If false, computation will be faster but structure will be inaccurate.

## "add_extra_box"
default value: False

units: NA

Add an extra box potential as an obstacle for actin, which could represent a membrane or another obstacle without explicitly representing it with particles. Also requires parameters "extra_box_center_x", "extra_box_center_y", "extra_box_center_z", "extra_box_size_x", "extra_box_size_y", "extra_box_size_z".

## "barbed_binding_site"
default value: False

units: NA

Add an explicit particle for a binding site for actin at the barbed end of actin filaments? If true, will increase accuracy of binding for actin monomers.

## "binding_site_reaction_distance"
default value: 0.1

units: nm

Distance between binding site particles and actin monomers to trigger a reaction. Larger values increase reaction events at the expense of spatial accuracy.