import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as py_datetime
from scipy.interpolate import PchipInterpolator
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

# global variables
goce_mass_interpolator = None

def load_and_interpolate_mass_data(file_path):
    """
    Loads GOCE mass data from the mass file, puts it into strings using pandas iloc.to_list()
    Puts mass and epoch values into list as correct time format, and floats.
    Creates a spline interpolator and stores it in a global variable.
    """
    #get that global goce_mass_interpolator from the beginning, we're going to give it values!
    global goce_mass_interpolator
    
    print(f"Loading mass data from {file_path} at the moment...")
    #reads in csv, use sep='\s+' to handle the space separated columns
    data = pd.read_csv(file_path, sep='\s+', skiprows=4, header=None)
    
    #adds data into arrays, iloc is integer location in pandas, list bc different data types so turn into strings
    time_strings = data.iloc[:, 0].to_list() # first column is time
    mass_values_str = data.iloc[:, 4].to_list() #fifth column is mass values
    
    #create empty lists for epochs and mass_values to be stored in
    epochs =[]
    mass_values =[]
    
    #puts the string lists into the empty lists above and handles exceptions
    for t_str, m_str in zip(time_strings, mass_values_str):
        try:
            #parse the string format into a time we can actually use
            dt = py_datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S")
            
            #convert to seconds since J2000 (our defined epoch)
            epoch = DateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second).to_epoch()
            
            #convert mass string to float
            mass = float(m_str)
            
            #append to empty arrays above 
            epochs.append(epoch)
            mass_values.append(mass)
        except (ValueError, TypeError):
            pass #skips anything strange
    
    #using cubic spline for smoooooothness, adds to global variable at beginning (MAGIC!!!!)
    # Using PCHIP for monotonicity (shape-preservation), which is physically-correct for mass.
    goce_mass_interpolator = PchipInterpolator(epochs, mass_values, extrapolate=True)    
    #just a quick check in to show it worked
    print(f"Mass interpolator has been created, it has {len(epochs)} data points.")
    #prints start and end epochs, or raises errors for any strange business
    if len(epochs)>0:
        print(f"The data covers epochs from {epochs[0]} to {epochs[-1]}")
    else:
        print("Error: No data points were loaded :(")
        raise ValueError("Failed to load any mass data from the file.")
    
    return epochs, mass_values



def setup_environment(simulation_start_epoch):
    """
    Sets up accurate rotation model and default Earth settings, gives Earth atmosphere.
    Sets up empty settings for GOCE, defines mass_function from interpolated mass at any time, and initial mass. 
    Defines rigid_body_settings and aero_coefficient settings for GOCE.
    Returns the SystemOfBodies object.
    """
    #Step 1: set up the environment
    bodies_to_create =["Earth","Moon","Sun"]
    precession_nutation_theory = environment_setup.rotation_model.IAUConventions.iau_2006
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    #create default body settings
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    #giving the earth shape
    body_settings.get("Earth").shape_settings = environment_setup.shape.spherical(6378.0E3)
    #adding rotation
    body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        precession_nutation_theory, global_frame_orientation)
    body_settings.get("Earth").gravity_field_settings.associated_reference_frame = "ITRS" 
    #add in earth atmosphere
    body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
    
    #1.2: define artificial bodies
    body_settings.add_empty_settings("GOCE")
    
    #gets that global variable that should have stuff in it from load_and_interpolate_data, handles errors if not
    global goce_mass_interpolator
    if goce_mass_interpolator is None:
        raise ValueError("Mass interpolator has not been initialized. Run load_and_interpolate_mass_data first.")
    
    #creates a fancy lambda function to define the mass at any time. It doesn't change it dynamically, it's stateless.
    mass_function = lambda time: float(goce_mass_interpolator(time))
    
    #gets the initial mass from the interpolator, CALCULATES initial_mass
    initial_mass = float(goce_mass_interpolator(simulation_start_epoch))
    print(f"Setting initial mass from interpolator at epoch {simulation_start_epoch}, intial mass is {initial_mass:.2f} kg")
    
    #you need these for the custom_time_dependent_rigid_body_properties, custom cos we change the mass now
    center_of_mass_function = lambda time: np.array([0.0, 0.0, 0.0])
    inertia_tensor_function = lambda time: np.diag([0.0, 0.0, 0.1])
    
    #defines the rigid_body_settings to be used in the aerodynamic stuff later
    rigid_body_settings = environment_setup.rigid_body.custom_time_dependent_rigid_body_properties(
        mass_function,
        center_of_mass_function,
        inertia_tensor_function
    )
    
    #I say later, it's right here- now defining aerodynamic properties for the satellite
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area=0.9,
        constant_force_coefficient=[1.5, 0, 0.3], # [Cd, 0, Cl]
        force_coefficients_frame=environment_setup.aerodynamic_coefficients.AerodynamicCoefficientFrames.negative_aerodynamic_frame_coefficients
    )
    #here we go, put these together: assigning aerodynamic settings
    body_settings.get("GOCE").aerodynamic_coefficient_settings = aero_coefficient_settings

    #1.3: create the system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    #you have to add rigid_body_settings after you've created the bodies object
    environment_setup.add_rigid_body_properties(
        bodies, "GOCE", rigid_body_settings)

    #from calculation earlier ASSIGNS initial_mass to GOCE
    bodies.get("GOCE").mass = initial_mass

    return bodies


def setup_propagation(bodies, simulation_start_epoch, simulation_end_epoch):
    """
    Defines the bodies that we are worrying about.
    Defines the acceleration settings for our satellite.
    Defines the initial_state and does funky shape stuff.
    Defines integrator (variable step RK4).
    Defines propagator (translational, mass handled in setup_propagation).
    Returns bodies_to_propagate and the propagator_settings
    """    
    #define bodies to propagate and central bodies
    bodies_to_propagate = ["GOCE"]
    central_bodies = ["Earth"]

    #2.1:define acceleration models
    
    #orders needed for spherical_harmonic_gravity, the earth is squashed
    maximum_degree = 12
    maximum_order = 6
    
    #actually defines GOCE acceleration settings
    acceleration_settings_GOCE = dict(
        Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(
                maximum_degree, maximum_order),
            propagation_setup.acceleration.aerodynamic()
            ],
        Moon = [propagation_setup.acceleration.point_mass_gravity()],
        Sun = [propagation_setup.acceleration.point_mass_gravity()]
    )
    #puts each objects acceleration_settings into a dictionary
    acceleration_settings = {"GOCE": acceleration_settings_GOCE}

    #create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # 2.2: Define the initial state
    #nice and easy, get this from Earth body (from spice I believe)
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    
    #defines initial_state
    initial_state_1d = element_conversion.keplerian_to_cartesian_elementwise(
        6634771.1,                     # semi_major_axis
        0.0009826,                     # eccentricity
        1.685494129,                   # inclination
        0.00,                          # argument_of_periapsis
        0.785398,                      # longitude_of_ascending_node
        0.00,                          # true_anomaly
        earth_gravitational_parameter  # gravitational_parameter
    )
    
    #tudat be funky, need to change from (6,) to (6,1), the 1 is 1 column, the -1 is a wildcard, 6 here because one body
    #you could stack multiple bodies too!
    #defines initial state for the propagator later
    initial_state_translation = initial_state_1d.reshape(-1, 1)

    # 2.3: Define integrator and propagator settings
    
    #variable step integrator, RK4 
    control_settings = propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
        propagation_setup.integrator.standard_cartesian_state_element_blocks, 1.0E-10, 1.0E-10 )
    validation_settings = propagation_setup.integrator.step_size_validation( 0.001, 1000.0 )
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step = 60.0,
        coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45,
        step_size_control_settings = control_settings,
        step_size_validation_settings = validation_settings )
    
    #defines termination of integrator
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    
    #saves the states into arrays for plotting later
    dependent_variables_to_save =[
        propagation_setup.dependent_variable.latitude("GOCE", "Earth"),
        propagation_setup.dependent_variable.longitude("GOCE", "Earth"),
        propagation_setup.dependent_variable.body_mass("GOCE")] #save the mass!
    
    #right, just translational propagation. we are hoping all the mass stuff is sorted earlier and saved
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state_translation,
        initial_time=simulation_start_epoch,
        integrator_settings=integrator_settings,
        termination_settings=termination_settings,
        output_variables=dependent_variables_to_save
    )
    
    return propagator_settings, bodies_to_propagate


def run_simulation(bodies, propagator_settings):
    """
    Actually does the propagation, does some print statements too. 
    """    
    # 3.1:create simulation object and propagate the dynamics
    
    #just a quick check in to show that everything is working
    print("Creating the simulator, usually takes a while don't panic...")
    
    dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
        bodies, propagator_settings)
    
    print("Dynamics simulator has been created, and propagation finished woohoo...")
    
    return dynamics_simulator


def process_and_plot_results(dynamics_simulator, simulation_start_epoch, simulation_end_epoch, bodies_to_propagate, raw_epochs, raw_mass_values):
    """
    Processes and plots the data. Hopefully.
    
    All the process stuff so arrays of states and dep variables, prints and plots
    """
    print("Thinking about maybe plotting the graphs...")
    #3.2: Extract results
    
    #gets all the results from dynamics_simulator, the function above
    states = dynamics_simulator.propagation_results.state_history
    states_array = result2array(states)
    dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history
    dependent_variables_array = result2array(dependent_variables)
    
    if states_array.size == 0 or dependent_variables_array.size == 0:
        print("Error: No simulation results to process :(")
        return
    
    #first state array
    #state array now has 7 columns- time, x, y, z, vx, vy, vz (time, positions and velocities in each direction)
    initial_position = states_array[0, 1:4] #this is first row of three columns (x,y,z)
    initial_velocity = states_array[0, 4:7] #this is first row of three columns (vx,vy,vz)
    final_epoch = states_array[-1, 0] #this is the last row, first column (time)
    final_position = states_array[-1, 1:4] #this is last row of three columns (x,y,z)
    final_velocity = states_array[-1, 4:7] #this is last row of three columns (vx,vy,vz)
    
    #now dependent variables array, 4 columns (0: time, 1: latitude, 2: longitude, 3: mass)
    initial_mass_from_deps = dependent_variables_array[0, 3] #this is first row of mass (column 3)
    final_mass_from_deps = dependent_variables_array[-1, 3] #this is last row of mass (column 3)
    #chucking in an extra little bit of info on mass rate, should be negative
    mass_rate = (final_mass_from_deps - initial_mass_from_deps) / (final_epoch - simulation_start_epoch)

    
    print(
        f"""
------------------------------------------------------
The results are in!    
The initial time was {simulation_start_epoch}
The final time was {simulation_end_epoch}
------------------------------------------------------
Mass from dependent variables array:
- Initial mass: {initial_mass_from_deps:} kg
- Final mass: {final_mass_from_deps:} kg
- Total mass change: {final_mass_from_deps - initial_mass_from_deps:} kg
- The mass rate is {mass_rate} kg/s
------------------------------------------------------
State from state array (time, positions, velocities):
- The initial position vector of GOCE is: {initial_position / 1E3} [km]
- The initial velocity vector of GOCE is: {initial_velocity / 1E3} [km/s]
------------------------------------------------------
After {final_epoch - simulation_start_epoch:.2f} seconds the position vector of GOCE is: {final_position / 1E3} [km]
And the velocity vector of GOCE is [km/s]: {final_velocity / 1E3}
------------------------------------------------------
"""
    )

    # 3.4: Visualize the trajectory
    
    #plot 1: 3D Trajectory
    print("Plotting 3D trajectories, ground track and mass...")
    fig = plt.figure(figsize=(6, 6), dpi=125)
    ax = plt.axes(projection='3d')
    ax.set_title('GOCE trajectory around Earth')
    ax.plot3D(states_array[:, 1], states_array[:, 2], states_array[:, 3], label=bodies_to_propagate[0], linestyle='-.')
    ax.scatter3D(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()

    #plot 2: ground track
    fig, ax = plt.subplots(tight_layout=True)
    latitude = dependent_variables_array[:, 1]
    longitude = dependent_variables_array[:, 2]

    #relative times tells you the array of how long it's been running in hours
    relative_time_hours = (dependent_variables_array[:, 0] - simulation_start_epoch) / 3600
    
    #boolean mask, is relative time less than three hours? keep(True), or discard (False)
    hours_to_extract = 3
    subset_mask = relative_time_hours <= hours_to_extract
    
    #long and lat for subset
    latitude_subset = np.rad2deg(latitude[subset_mask])
    longitude_subset = np.rad2deg(longitude[subset_mask])

    #plot ground track
    ax.set_title("3 hour ground track of GOCE")
    ax.scatter(longitude_subset, latitude_subset, s=1)
    if latitude_subset.size > 0:
        ax.scatter(longitude_subset[0], latitude_subset[0], label="Start", color="green", marker='o')
        ax.scatter(longitude_subset[-1], latitude_subset[-1], label="End", color="red", marker="x")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xticks(np.arange(-180, 181, step=45))
    ax.set_yticks(np.arange(-90, 91, step=45))
    ax.grid(True)
    ax.legend()
    plt.show()
    
    #plot 3:mass over time, shows our interpolation working essentially, should be fairly linear and definitely negative gradient. Worry if not.
    fig, ax = plt.subplots(tight_layout=True)
    relative_time_hours_all = (dependent_variables_array[:, 0] - simulation_start_epoch) / 3600
    mass = dependent_variables_array[:, 3]
    ax.set_title("GOCE Mass During Simulation")
    ax.plot(relative_time_hours_all, mass, label="Interpolated data")

    # --- MODIFIED LINES FOR FILTERING ---
    # Convert raw data lists to numpy arrays
    raw_epochs_array = np.array(raw_epochs)
    raw_mass_array = np.array(raw_mass_values)
    
    # Create a boolean mask to filter epochs within the simulation time range
    time_mask = (raw_epochs_array >= simulation_start_epoch) & (raw_epochs_array <= simulation_end_epoch)
    
    # Apply the mask to get only the points within the simulation
    filtered_epochs = raw_epochs_array[time_mask]
    filtered_masses = raw_mass_array[time_mask]
    
    # Convert the *filtered* epochs to relative time in hours
    filtered_relative_time_hours = (filtered_epochs - simulation_start_epoch) / 3600
    
    # Plot only the filtered data points
    ax.scatter(filtered_relative_time_hours, filtered_masses, 
               color='red', marker='x', label="Actual data points")
    
    ax.legend()  # Add legend to show the labels
    # --- END OF MODIFIED LINES ---

    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Mass [kg]")
    ax.grid(True)
    plt.show()
    
    print("Graphs are plotted, you did it :)")

def main():
    """
    Getting all the steps in:
        SUBSTEPS: sort kernels, define start and end epoch, *
        1. Load and interpolate mass data from file
        2. Set up the enviroment
        3. Set up the propagation
        4. Perform the propagation using dynamics_simulator
        5. Process and plot the results
        (*hope it works)
    """
    #load SPICE kernels
    spice.clear_kernels()
    spice.load_standard_kernels()
    
    #1. load mass data, process it and turn it into something readable, interpolation time
    raw_epochs, raw_mass_values = load_and_interpolate_mass_data("GOCE-Mass-Properties.txt") 

    #define the start and end epochs, format (Y, m, d, H, M, S)
    simulation_start_epoch = DateTime(2009, 5, 1, 0, 0, 0).to_epoch()
    simulation_end_epoch = DateTime(2009, 5, 3).to_epoch() 

    #2. Set up the environment
    bodies = setup_environment(simulation_start_epoch)

    #3. Set up the propagation
    propagator_settings, bodies_to_propagate = setup_propagation(bodies, simulation_start_epoch, simulation_end_epoch)

    #4. Perform the propagation
    dynamics_simulator = run_simulation(bodies, propagator_settings) 

    #5. Process and plot results
    process_and_plot_results(dynamics_simulator, simulation_start_epoch, simulation_end_epoch, bodies_to_propagate, raw_epochs, raw_mass_values)

if __name__ == "__main__":
    main()
