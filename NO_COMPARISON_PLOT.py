"""
IMPORTANT NOTE: This file does not contain the actual data plots for comparison. It JUST contains the mass plot. 
This is essentially an archived, legacy version here for completeness. You should run comparison_plots.py instead for the most up to date version!
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as py_datetime
from scipy.interpolate import PchipInterpolator
from tudatpy.interface import spice
from tudatpy import dynamics, constants
from tudatpy.dynamics import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
import configparser
import os

# global variables
goce_mass_interpolator = None

def load_config(file_name):
    """
    Loads the .ini configuration file which contains a lot of the parameters used here.
    Have an experiment!
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Error: Configuration file '{file_name}' not found.")
    
    config = configparser.ConfigParser()
    config.read(file_name)
    print(f"Configuration file '{file_name}' loaded successfully.")
    return config

def load_and_interpolate_mass_data(config):
    """
    Loads GOCE mass data from the file specified in the config(mass file), puts it into strings using pandas iloc.to_list()
    Puts mass and epoch values into list as correct time format, and floats.
    Creates a PChip interpolator and stores interpolated masses in a global variable.  
    Returns the raw epochs and mass values for plotting.
    """
    #get that global goce_mass_interpolator from the beginning, we're going to give it values!
    global goce_mass_interpolator
    
    #just configuring the mass file using the parameters
    cfg_files = config['FILES']
    file_path = cfg_files.get('mass_file')
    skiprows = cfg_files.getint('mass_file_skiprows')
    time_col = cfg_files.getint('mass_file_time_column')
    mass_col = cfg_files.getint('mass_file_mass_column')
    
    #adding in some error handling
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Mass data file '{file_path}' not found.")
    
    #a reassuring 'your file is being read' statement to calm the nerves
    print(f"Loading mass data from {file_path} at the moment...")
    
    #reads in csv, use sep='\s+' to handle the space separated columns
    data = pd.read_csv(file_path, sep='\s+', skiprows=skiprows, header=None)
    
    #adds data into arrays, iloc is integer location in pandas, list bc different data types so turn into strings
    time_strings = data.iloc[:, time_col].to_list() # first column is time
    mass_values_str = data.iloc[:, mass_col].to_list() #fifth column is mass values
    
    #create empty lists for epochs and mass_values to be stored in
    epochs = []
    mass_values = []
    
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
    
    #using PCHIP for monotonicity (shape-preservation), which avoids mass going up over time (cubic spline caused this)
    goce_mass_interpolator = PchipInterpolator(epochs, mass_values, extrapolate=True)
    
    #just a quick check in to show it worked
    print(f"Mass interpolator has been created, it has {len(epochs)} data points.")
    #prints start and end epochs, or raises errors for any strange business.
    if len(epochs) > 0:
        print(f"The data covers epochs from {epochs[0]} to {epochs[-1]}")
    else:
        print("Error: No data points were loaded :(")
        raise ValueError("Failed to load any mass data from the file.")
    
    return epochs, mass_values

def setup_environment(simulation_start_epoch, config):
    """
    Sets up accurate rotation model and default Earth settings, gives Earth atmosphere.
    Sets up empty settings for GOCE, defines mass_function from interpolated mass at any time, and initial mass. 
    Defines solar radiation pressure (SRP) settings for Sun (radiation source).
    Defines SRP settings for GOCE (radiation target).
    Defines rigid_body_settings and aero_coefficient settings for GOCE.
    Returns the SystemOfBodies object.
    """
    print("Setting up simulation environment...")
    
    #Step 1: set up the environment
    bodies_to_create = ["Earth", "Moon", "Sun"]
    precession_nutation_theory = environment_setup.rotation_model.IAUConventions.iau_2006
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    
    #create default body settings
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    #giving the earth shape
    cfg_earth = config['ENV_EARTH']
    #adding rotation
    body_settings.get("Earth").shape_settings = environment_setup.shape.spherical(
        cfg_earth.getfloat('radius'))
    body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        precession_nutation_theory, global_frame_orientation)
    body_settings.get("Earth").gravity_field_settings.associated_reference_frame = "ITRS"
    #add in earth atmosphere
    body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
    
    #1.2: define artificial bodies
    body_settings.add_empty_settings("GOCE")
    
    #gets that global variable that should have stuff in it from load_and_interpolate_mass_data, handles errors if not
    global goce_mass_interpolator
    if goce_mass_interpolator is None:
        raise ValueError("Mass interpolator has not been initialized.")
    
    #creates a fancy lambda function to define the mass at any time. It doesn't change it dynamically, it's stateless.
    mass_function = lambda time: float(goce_mass_interpolator(time))
    
    #gets the initial mass from the interpolator, CALCULATES initial_mass
    initial_mass = float(goce_mass_interpolator(simulation_start_epoch))
    print(f"Setting initial mass from interpolator at epoch {simulation_start_epoch}, initial mass is {initial_mass:.2f} kg")
    
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
    cfg_goce = config['ENV_GOCE']
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area=cfg_goce.getfloat('aero_ref_area'),
        constant_force_coefficient=[cfg_goce.getfloat('aero_cd'), 0, cfg_goce.getfloat('aero_cl')],
        force_coefficients_frame=environment_setup.aerodynamic_coefficients.AerodynamicCoefficientFrames.negative_aerodynamic_frame_coefficients
    )
    
    #here we go, put these together: assigning aerodynamic settings
    body_settings.get("GOCE").aerodynamic_coefficient_settings = aero_coefficient_settings

    #define the radiation source, sun, for solar radiation pressure (SRP)
    # (Note: Using 1367.0 W/m^2 as the solar constant at 1 AU)
    cfg_sun = config['ENV_SUN']
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    luminosity_model = environment_setup.radiation_pressure.irradiance_based_constant_luminosity(cfg_sun.getfloat('solar_constant'), constants.ASTRONOMICAL_UNIT)
    body_settings.get("Sun").radiation_source_settings = environment_setup.radiation_pressure.isotropic_radiation_source(luminosity_model)
    
    #define the radiation target for SRP, GOCE
    body_settings.get("GOCE").radiation_pressure_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        cfg_goce.getfloat('srp_ref_area'), cfg_goce.getfloat('srp_coeff'), occulting_bodies_dict )
        
    #1.3: create the system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    #you have to add rigid_body_settings after you've created the bodies object
    environment_setup.add_rigid_body_properties(
        bodies, "GOCE", rigid_body_settings)

    #from calculation earlier ASSIGNS initial_mass to GOCE
    bodies.get("GOCE").mass = initial_mass

    return bodies

def setup_propagation(bodies, simulation_start_epoch, simulation_end_epoch, config):
    """
    Defines the bodies that we are worrying about.
    Defines the acceleration settings for our satellite includinf: 
        - aerodynamic drag
        - relativistic correction
        - spherical harmonic gravity
        - third bodies.
    Defines the initial_state and does funky shape stuff.
    Defines integrator (variable step RK4).
    Defines propagator (translational, mass handled in setup_propagation).
    Returns bodies_to_propagate and the propagator_settings
    """     
    #define bodies to propagate and central bodies
    bodies_to_propagate = ["GOCE"]
    central_bodies = ["Earth"]

    #2.1:define acceleration models
    #setting up the parameters for Earth: spherical harmonic gravity, Earth squashed
    cfg_earth = config['ENV_EARTH']
    earth_gravity_settings = propagation_setup.acceleration.spherical_harmonic_gravity(
        cfg_earth.getint('gravity_max_degree'), 
        cfg_earth.getint('gravity_max_order')
    )
    #setting up the parameters for relatavistic correction to acceleration (Earth)
    cfg_physics = config['RELATIVISTIC_CORRECTION']
    relativistic_settings = propagation_setup.acceleration.relativistic_correction(
        cfg_physics.getboolean('use_schwarzschild'),
        cfg_physics.getboolean('lense_thirring'),
        cfg_physics.getboolean('de_sitter'),
        cfg_physics.get('de_sitter_central_body')
    )
    #actually defines GOCE acceleration settings    
    acceleration_settings_GOCE = dict(
        Earth=[
            earth_gravity_settings,
            propagation_setup.acceleration.aerodynamic(),
            relativistic_settings
        ],
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Sun=[
            propagation_setup.acceleration.point_mass_gravity(),
            # This is the modern API call. It finds the source/target models
            # we defined in setup_environment.
            propagation_setup.acceleration.radiation_pressure()
        ]
    )
    
    #puts each objects acceleration_settings into a dictionary
    acceleration_settings = {"GOCE": acceleration_settings_GOCE}

    #create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # 2.2: Define the initial state
    #nice and easy, get this from Earth body (from spice I believe)
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    cfg_state = config['PROPAGATION_INITIAL_STATE']
    
    #defines initial_state
    initial_state_1d = element_conversion.keplerian_to_cartesian_elementwise(
        cfg_state.getfloat('semi_major_axis'),
        cfg_state.getfloat('eccentricity'),
        cfg_state.getfloat('inclination'),
        cfg_state.getfloat('argument_of_periapsis'),
        cfg_state.getfloat('longitude_of_ascending_node'),
        cfg_state.getfloat('true_anomaly'),
        earth_gravitational_parameter
    )
    #tudat be funky, need to change from (6,) to (6,1), the 1 is 1 column, the -1 is a wildcard, 6 here because one body
    #you could stack multiple bodies too!
    #defines initial state for the propagator later
    initial_state_translation = initial_state_1d.reshape(-1, 1)

    # 2.3: Define integrator and propagator settings
    #variable step integrator, RK4
    cfg_int = config['PROPAGATION_INTEGRATOR']
    
    control_settings = propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
        propagation_setup.integrator.standard_cartesian_state_element_blocks, 
        cfg_int.getfloat('position_tolerance'), 
        cfg_int.getfloat('velocity_tolerance')
    )
    validation_settings = propagation_setup.integrator.step_size_validation(
        cfg_int.getfloat('minimum_time_step'), 
        cfg_int.getfloat('maximum_time_step')
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=cfg_int.getfloat('initial_time_step'),
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        step_size_control_settings=control_settings,
        step_size_validation_settings=validation_settings
    )
    #defines termination of integrator
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    
    #saves the states into arrays for plotting later
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.latitude("GOCE", "Earth"),
        propagation_setup.dependent_variable.longitude("GOCE", "Earth"),
        propagation_setup.dependent_variable.body_mass("GOCE")
    ]
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
    print("Creating the simulator, usually takes a while don't panic...")
    # 3.1:create simulation object and propagate the dynamics
    
    #just a quick check in to show that everything is working
    dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
        bodies, propagator_settings)  
    print("Dynamics simulator has been created, and propagation finished woohoo...")
    
    return dynamics_simulator

def process_and_plot_results(dynamics_simulator, simulation_start_epoch, simulation_end_epoch, bodies_to_propagate, raw_epochs, raw_mass_values, config):
    """
    Processes and plots the data. Hopefully.
    
    All the process stuff so arrays of states and dep variables, prints and plots.
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
Simulation Results:
Initial time: {simulation_start_epoch}
Final time:   {simulation_end_epoch}
------------------------------------------------------
Mass from dependent variables:
- Initial mass: {initial_mass_from_deps:.3f} kg
- Final mass:   {final_mass_from_deps:.3f} kg
- Mass change:  {final_mass_from_deps - initial_mass_from_deps:.3f} kg
- Mass rate:    {mass_rate:.2e} kg/s
------------------------------------------------------
State from state array:
- Initial position [km]: {initial_position / 1E3}
- Initial velocity [km/s]: {initial_velocity / 1E3}
------------------------------------------------------
After {final_epoch - simulation_start_epoch:.2f} seconds:
- Final position [km]: {final_position / 1E3}
- Final velocity [km/s]: {final_velocity / 1E3}
------------------------------------------------------
"""
    )

    # 3.4: Visualize the trajectory
    #plot 1: 3D trajectory
    cfg_plot = config['PLOTTING']
    plot_dpi = cfg_plot.getint('dpi')
    
    print("Plotting 3D trajectory, ground track, and mass...")
    fig = plt.figure(figsize=(6, 6), dpi=plot_dpi)
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
    fig, ax = plt.subplots(tight_layout=True, dpi=plot_dpi)
    latitude = dependent_variables_array[:, 1]
    longitude = dependent_variables_array[:, 2]
    
    #relative times tells you the array of how long it's been running in hours
    relative_time_hours = (dependent_variables_array[:, 0] - simulation_start_epoch) / 3600
    
    #boolean mask, is relative time less than three hours? keep(True), or discard (False)
    hours_to_extract = cfg_plot.getfloat('ground_track_hours')
    subset_mask = relative_time_hours <= hours_to_extract
    
    #long and lat for subset
    latitude_subset = np.rad2deg(latitude[subset_mask])
    longitude_subset = np.rad2deg(longitude[subset_mask])

    #plot ground track
    ax.set_title(f"{hours_to_extract} hour ground track of GOCE")
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
    fig, ax = plt.subplots(tight_layout=True, dpi=plot_dpi)
    relative_time_hours_all = (dependent_variables_array[:, 0] - simulation_start_epoch) / 3600
    mass = dependent_variables_array[:, 3]
    ax.set_title("GOCE Mass During Simulation")
    ax.plot(relative_time_hours_all, mass, label="Simulated (Interpolated)")

    #convert raw data lists to numpy arrays
    raw_epochs_array = np.array(raw_epochs)
    raw_mass_array = np.array(raw_mass_values)
    
    #create a boolean mask to filter epochs within the simulation time range
    time_mask = (raw_epochs_array >= simulation_start_epoch) & (raw_epochs_array <= simulation_end_epoch)
    
    #apply the mask to get only the points within the simulation
    filtered_epochs = raw_epochs_array[time_mask]
    filtered_masses = raw_mass_array[time_mask]
    
    #convert the filtered epochs to relative time in hours
    filtered_relative_time_hours = (filtered_epochs - simulation_start_epoch) / 3600
    
    #plot only the filtered data points
    ax.scatter(filtered_relative_time_hours, filtered_masses, 
               color='red', marker='x', label="Raw Data Points")
    ax.legend()
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Mass [kg]")
    ax.grid(True)
    plt.show()
    
    print("All plots complete.")

def main():
    """
    Main function to run the GOCE simulation:
        1. Load configuration file
        2. Load SPICE kernels 
        3. Load and interpolate mass data (from config)
        4. Set up simulation start/end times (from config)
        5. Set up environment (from config)
        6. Set up propagation (from config)
        7. Run simulation
        8. Process and plot results (from config)*
        (*hope it works)
    """
    #handles exceptions in file loading before running the script, sensible!
    try:
        #1. load parameters configuration
        config = load_config('parameters.ini')

        #2. load SPICE kernels
        spice.clear_kernels()
        spice.load_standard_kernels()
        
        #3. load mass data, process it and turn it into something readable, interpolation time, assign raw mass values
        cfg_files = config['FILES']
        raw_epochs, raw_mass_values = load_and_interpolate_mass_data(config)
    
        #4. define the start and end epochs, format (Y, m, d, H, M, S)
        cfg_time = config['SIM_TIME']
        simulation_start_epoch = DateTime(
            cfg_time.getint('start_year'),
            cfg_time.getint('start_month'),
            cfg_time.getint('start_day'),
            cfg_time.getint('start_hour'),
            cfg_time.getint('start_minute'),
            cfg_time.getint('start_second')
        ).to_epoch()
        
        simulation_end_epoch = DateTime(
            cfg_time.getint('end_year'),
            cfg_time.getint('end_month'),
            cfg_time.getint('end_day'),
            cfg_time.getint('end_hour'),
            cfg_time.getint('end_minute'),
            cfg_time.getint('end_second')
        ).to_epoch()

        #5. set up the environment
        bodies = setup_environment(simulation_start_epoch, config)

        #6. set up the propagation
        propagator_settings, bodies_to_propagate = setup_propagation(
            bodies, simulation_start_epoch, simulation_end_epoch, config)

        #7. perform the propagation
        dynamics_simulator = run_simulation(bodies, propagator_settings)

        #8. process and plot results
        process_and_plot_results(
            dynamics_simulator, 
            simulation_start_epoch, 
            simulation_end_epoch, 
            bodies_to_propagate, 
            raw_epochs, 
            raw_mass_values, 
            config
        )

    except Exception as e:
        print("--- AN ERROR OCCURRED ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    main()

