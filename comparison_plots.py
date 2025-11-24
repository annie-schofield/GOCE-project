import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as py_datetime
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize
from tudatpy.interface import spice
from tudatpy import dynamics, constants
from tudatpy.dynamics import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
import configparser
import os

#global variables
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

def load_real_orbit_data(file_path):
    """
    Loads in goce_orbit_data.csv created in create_actual_data_file.py, around 900 points.
    Puts the time into the correct format.
    Converts from km to m.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Real orbit data file '{file_path}' not found.")
        return None

    print(f"Loading real orbit data from {file_path}...")
    df = pd.read_csv(file_path)

    epochs = []
    #assumes format: 2009-10-02T00:01:39.313867
    for ts in df['timestamp']:
        try:
            dt = py_datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            dt = py_datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            
        seconds_float = dt.second + (dt.microsecond / 1e6)
        epoch = DateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds_float).to_epoch()
        epochs.append(epoch)

    x_vals = df['X'].values * 1000.0
    y_vals = df['Y'].values * 1000.0
    z_vals = df['Z'].values * 1000.0
    
    return {'epochs': np.array(epochs), 'x': x_vals, 'y': y_vals, 'z': z_vals}

def get_initial_state_from_csv(real_data, bodies):
    """
    Estimates initial state from CSV but enforces circular velocity 
    to prevent the satellite from crashing due to linear approximation errors.
    """
    print("Estimating initial state from CSV data...")
    
    # 1. Get first two points in Earth-Fixed Frame (ITRS)
    t0 = real_data['epochs'][0]
    t1 = real_data['epochs'][1]
    
    pos0_itrs = np.array([real_data['x'][0], real_data['y'][0], real_data['z'][0]])
    pos1_itrs = np.array([real_data['x'][1], real_data['y'][1], real_data['z'][1]])
    
    # 2. Convert to Inertial Frame (GCRS) using SPICE
    rot_mat_0 = spice.compute_rotation_matrix_between_frames("IAU_EARTH", "J2000", t0)
    rot_mat_1 = spice.compute_rotation_matrix_between_frames("IAU_EARTH", "J2000", t1)
    
    pos0_gcrs = np.dot(rot_mat_0, pos0_itrs)
    pos1_gcrs = np.dot(rot_mat_1, pos1_itrs)
    
    # 3. Estimate Velocity Direction (Linear Approximation)
    dt = t1 - t0
    v_linear = (pos1_gcrs - pos0_gcrs) / dt
    
    # --- FIX STARTS HERE ---
    # Normalize to get just the direction
    v_direction = v_linear / np.linalg.norm(v_linear)
    
    # Calculate the exact speed required for a circular orbit at this altitude
    # v = sqrt(mu / r)
    r_mag = np.linalg.norm(pos0_gcrs)
    mu_earth = bodies.get("Earth").gravitational_parameter
    v_circular_mag = np.sqrt(mu_earth / r_mag)
    
    # Create a new velocity vector with the correct speed but the derived direction
    velocity_corrected = v_direction * v_circular_mag
    print('------------------------------------------------------')
    print(f"Linear Velocity Estimate: {np.linalg.norm(v_linear):.2f} m/s")
    print(f"Corrected Circular Velocity: {v_circular_mag:.2f} m/s")
    
    initial_state = np.concatenate([pos0_gcrs, velocity_corrected])
    # --- FIX ENDS HERE ---
    
    # Convert to keplerian elements and output them
    keplerian_elements = element_conversion.cartesian_to_keplerian(initial_state, mu_earth)
    
    print('------------------------------------------------------')
    print("Calculated initial orbital elements (from CSV):")
    print(f"Semi-major Axis:     {keplerian_elements[0]:.2f} m")
    print(f"Eccentricity:        {keplerian_elements[1]:.6f}")
    print(f"Inclination:         {np.rad2deg(keplerian_elements[2]):.4f} deg")
    print(f"Arg of Periapsis:    {np.rad2deg(keplerian_elements[3]):.4f} deg")
    print(f"RAAN:                {np.rad2deg(keplerian_elements[4]):.4f} deg")
    print(f"True Anomaly:        {np.rad2deg(keplerian_elements[5]):.4f} deg")
    print('------------------------------------------------------')

    return initial_state, t0
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
        goce_mass_interpolator = PchipInterpolator([0, 1e9], [1000, 1000], extrapolate=True)
        return [], []
    
    #a reassuring 'your file is being read' statement to calm the nerves
    print(f"Loading mass data from {file_path} at the moment...")
    
    #reads in csv, use sep='\s+' to handle the space separated columns
    data = pd.read_csv(file_path, sep='\s+', skiprows=skiprows, header=None)
    
    #adds data into arrays, iloc is integer location in pandas, list bc different data types so turn into strings
    time_strings = data.iloc[:, time_col].to_list() 
    mass_values_str = data.iloc[:, mass_col].to_list()
    
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
    
    #prints start and end epochs, or raises errors for any strange business.
    if len(epochs) > 0:
        sorted_indices = np.argsort(epochs)
        epochs = np.array(epochs)[sorted_indices]
        mass_values = np.array(mass_values)[sorted_indices]
        
        #using PCHIP for monotonicity (shape-preservation), which avoids mass going up over time (cubic spline caused this)
        goce_mass_interpolator = PchipInterpolator(epochs, mass_values, extrapolate=True)
    else:
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
    #step 1: set up the environment
    bodies_to_create = ["Earth", "Moon", "Sun"]
    
    precession_nutation_theory = environment_setup.rotation_model.IAUConventions.iau_2006
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    
    #create default body settings
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)
    
    #giving the earth shape
    cfg_earth = config['ENV_EARTH']
    body_settings.get("Earth").shape_settings = environment_setup.shape.spherical(
        cfg_earth.getfloat('radius'))
    #adding rotation
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
    print(f"Initial mass set to {initial_mass:.2f} kg")
    
    #you need these for the custom_time_dependent_rigid_body_properties, custom cos we change the mass now
    center_of_mass_function = lambda time: np.array([0.0, 0.0, 0.0])
    inertia_tensor_function = lambda time: np.diag([0.0, 0.0, 0.1])
    
    #defines the rigid_body_settings to be used in the aerodynamic stuff later
    rigid_body_settings = environment_setup.rigid_body.custom_time_dependent_rigid_body_properties(
        mass_function, center_of_mass_function, inertia_tensor_function)
    
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
    environment_setup.add_rigid_body_properties(bodies, "GOCE", rigid_body_settings)
    
    #from calculation earlier ASSIGNS initial_mass to GOCE
    bodies.get("GOCE").mass = initial_mass
    return bodies

def setup_propagation(bodies, simulation_start_epoch, simulation_end_epoch, config, initial_state_override=None):
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
        cfg_earth.getint('gravity_max_degree'), cfg_earth.getint('gravity_max_order'))
    
    #setting up the parameters for relatavistic correction to acceleration (Earth)
    cfg_physics = config['RELATIVISTIC_CORRECTION']
    relativistic_settings = propagation_setup.acceleration.relativistic_correction(
        cfg_physics.getboolean('use_schwarzschild'), cfg_physics.getboolean('lense_thirring'),
        cfg_physics.getboolean('de_sitter'), cfg_physics.get('de_sitter_central_body')
    )
    #actually defines GOCE acceleration settings    
    acceleration_settings_GOCE = dict(
        Earth=[earth_gravity_settings, relativistic_settings], #propagation_setup.acceleration.aerodynamic()
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Sun=[propagation_setup.acceleration.point_mass_gravity(), propagation_setup.acceleration.radiation_pressure()]
    )
    #puts each objects acceleration_settings into a dictionary
    acceleration_settings = {"GOCE": acceleration_settings_GOCE}
    
    #create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    #initial state selection, will use calculated, if not then parameters
    if initial_state_override is not None:
        print("Using initial state derived from CSV data.")
        initial_state_translation = initial_state_override.reshape(-1, 1)
    else:
        print("Using INITIAL STATE from configuration file.")
        earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
        cfg_state = config['PROPAGATION_INITIAL_STATE']
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
        cfg_int.getfloat('position_tolerance'), cfg_int.getfloat('velocity_tolerance')
    )
    validation_settings = propagation_setup.integrator.step_size_validation(
        cfg_int.getfloat('minimum_time_step'), cfg_int.getfloat('maximum_time_step')
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=cfg_int.getfloat('initial_time_step'),
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        step_size_control_settings=control_settings, step_size_validation_settings=validation_settings
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
        central_bodies=central_bodies, acceleration_models=acceleration_models, bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state_translation, initial_time=simulation_start_epoch, integrator_settings=integrator_settings,
        termination_settings=termination_settings, output_variables=dependent_variables_to_save
    )
    return propagator_settings, bodies_to_propagate

def run_simulation(bodies, propagator_settings):
    """
    Actually does the propagation, does some print statements too.
    """
    # 3.1:create simulation object and propagate the dynamics
    
    #just a quick check in to show that everything is working
    print("Creating the simulator...")
    dynamics_simulator = dynamics.simulator.create_dynamics_simulator(bodies, propagator_settings)  
    print("Dynamics simulator created and propagation finished.")
    return dynamics_simulator

def transform_real_data_to_inertial(real_data):
    """
    Converts the data into the inertial frame, from ITRS to GCRS to forcibly make it function.
    """
    print("Transforming actual data from ITRS to GCRS...")
    transformed_x, transformed_y, transformed_z = [], [], []
    epochs = real_data['epochs']
    
    for i, t in enumerate(epochs):
        pos_body_fixed = np.array([real_data['x'][i], real_data['y'][i], real_data['z'][i]])
        rot_matrix = spice.compute_rotation_matrix_between_frames("IAU_EARTH", "J2000", t)
        pos_inertial = np.dot(rot_matrix, pos_body_fixed)
        transformed_x.append(pos_inertial[0])
        transformed_y.append(pos_inertial[1])
        transformed_z.append(pos_inertial[2])
    
    return {'epochs': epochs, 'x': np.array(transformed_x), 'y': np.array(transformed_y), 'z': np.array(transformed_z)}

def process_and_plot_results(dynamics_simulator, simulation_start_epoch, simulation_end_epoch, bodies_to_propagate, raw_epochs, raw_mass_values, config, real_orbit_data=None):
    """
    Processes and plots the data. Hopefully.
    
    All the process stuff so arrays of states and dep variables, prints and plots.
    """
    print("Processing results...")
    states_array = result2array(dynamics_simulator.propagation_results.state_history)
    dependent_variables_array = result2array(dynamics_simulator.propagation_results.dependent_variable_history)
    
    if states_array.size == 0:
        print("Error: No simulation results.")
        return

    #transform real data for 3D plot
    transformed_real_data = None
    if real_orbit_data is not None:
        try:
            transformed_real_data = transform_real_data_to_inertial(real_orbit_data)
        except Exception as e:
            print(f"Warning: Could not transform real data: {e}")

    cfg_plot = config['PLOTTING']
    plot_dpi = cfg_plot.getint('dpi')
    
    #plot 1: 3D trajectory
    print("Plotting 3D trajectory...")
    fig = plt.figure(figsize=(8, 8), dpi=plot_dpi)
    ax = plt.axes(projection='3d')
    ax.set_title('GOCE Trajectory: Inertial Frame (GCRS)')
    ax.plot3D(states_array[:, 1], states_array[:, 2], states_array[:, 3], label="Simulated", linestyle='-', color='blue', alpha=0.8)
    
    if transformed_real_data is not None:
        t_mask = (transformed_real_data['epochs'] >= simulation_start_epoch - 7200) & (transformed_real_data['epochs'] <= simulation_end_epoch + 7200)
        if np.any(t_mask):
            ax.plot3D(transformed_real_data['x'][t_mask], transformed_real_data['y'][t_mask], transformed_real_data['z'][t_mask], label="Real Data", color='red', alpha=0.6, linestyle='--')

    ax.scatter3D(0.0, 0.0, 0.0, label="Earth", marker='o', color='green')
    ax.legend()
    plt.show()

    #plot 2: ground track
    print("Plotting Ground Track...")
    fig, ax = plt.subplots(tight_layout=True, dpi=plot_dpi)
    latitude_sim = np.rad2deg(dependent_variables_array[:, 1])
    longitude_sim = np.rad2deg(dependent_variables_array[:, 2])
    
    relative_time = (dependent_variables_array[:, 0] - simulation_start_epoch) / 3600
    subset_mask = relative_time <= cfg_plot.getfloat('ground_track_hours')
    
    ax.set_title("Ground Track")
    ax.scatter(longitude_sim[subset_mask], latitude_sim[subset_mask], s=1, label="Simulated", color='blue')
    
    if real_orbit_data is not None:
        x_real, y_real, z_real = real_orbit_data['x'], real_orbit_data['y'], real_orbit_data['z']
        r_real = np.sqrt(x_real**2 + y_real**2 + z_real**2)
        lat_real_deg = np.rad2deg(np.arcsin(z_real / r_real))
        lon_real_deg = np.rad2deg(np.arctan2(y_real, x_real))
        
        t_mask = (real_orbit_data['epochs'] >= simulation_start_epoch - 7200) & (real_orbit_data['epochs'] <= simulation_end_epoch + 7200)
        if np.any(t_mask):
            ax.scatter(lon_real_deg[t_mask], lat_real_deg[t_mask], s=1, label="Real Data", color='red')

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.grid(True)
    ax.legend()
    plt.show()
    print("All plots complete.")

def calculate_and_plot_residuals(sim_epochs, sim_states, real_data_inertial, plot_dpi):
    """
    Calculates and plots the position error magnitude between the simulation
    and the real data (converted to inertial frame).
    """
    print("Calculating residuals (simulation vs actual data)...")
    
    real_epochs = real_data_inertial['epochs']
    real_pos = np.vstack((real_data_inertial['x'], real_data_inertial['y'], real_data_inertial['z'])).T
    
    #filter real data to be within simulation time range
    valid_mask = (real_epochs >= sim_epochs[0]) & (real_epochs <= sim_epochs[-1])
    real_epochs = real_epochs[valid_mask]
    real_pos = real_pos[valid_mask]
    
    if len(real_epochs) == 0:
        print("Error: No overlap between simulation and real data timestamps.")
        return

    #create interpolators for simulation X, Y, Z to match real data timestamps
    interp_x = interp1d(sim_epochs, sim_states[:, 0], kind='cubic')
    interp_y = interp1d(sim_epochs, sim_states[:, 1], kind='cubic')
    interp_z = interp1d(sim_epochs, sim_states[:, 2], kind='cubic')
    
    sim_pos_at_real_times = np.vstack((
        interp_x(real_epochs),
        interp_y(real_epochs),
        interp_z(real_epochs)
    )).T
    
    #calculate residuals (Vector Difference)
    pos_diff = sim_pos_at_real_times - real_pos
    
    #calculate magnitude of error (RSS Position Error)
    rss_error = np.linalg.norm(pos_diff, axis=1)
    print('------------------------------------------------------')
    print(f"Mean Position Error: {np.mean(rss_error):.2f} m")
    print(f"Max Position Error:  {np.max(rss_error):.2f} m")
    print('------------------------------------------------------')

    #plot residuals
    fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
    relative_time_hours = (real_epochs - real_epochs[0]) / 3600
    
    ax.plot(relative_time_hours, rss_error, color='purple', label='Position Error Magnitude')
    
    ax.set_title("Orbit Prediction Residuals (Simulated vs Real)")
    ax.set_xlabel("Time since start [hours]")
    ax.set_ylabel("Position Error [m]")
    ax.grid(True)
    ax.legend()
    plt.show()
    
def setup_models_for_optimization(bodies, config, simulation_end_epoch):
    """
    Pre-builds the acceleration models and integrator settings ONCE 
    so we don't have to recreate them 50 times during the loop.
    """
    bodies_to_propagate = ["GOCE"]
    central_bodies = ["Earth"]
    
    cfg_earth = config['ENV_EARTH']
    earth_gravity_settings = propagation_setup.acceleration.spherical_harmonic_gravity(
        cfg_earth.getint('gravity_max_degree'), cfg_earth.getint('gravity_max_order'))
    
    cfg_physics = config['RELATIVISTIC_CORRECTION']
    relativistic_settings = propagation_setup.acceleration.relativistic_correction(
        cfg_physics.getboolean('use_schwarzschild'), False,
        cfg_physics.getboolean('use_de_sitter'), cfg_physics.get('de_sitter_central_body')
    )
    
    #no Aerodynamic drag during optimization (makes things quicker, and GOCE not significant drag)
    acceleration_settings_GOCE = dict(
        Earth=[earth_gravity_settings, relativistic_settings],
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Sun=[propagation_setup.acceleration.point_mass_gravity(), propagation_setup.acceleration.radiation_pressure()]
    )
    
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, {"GOCE": acceleration_settings_GOCE}, bodies_to_propagate, central_bodies)

    #use looser tolerance (1.0E-6) for the optimizer to make it run fast
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=60.0,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        step_size_control_settings=propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
            propagation_setup.integrator.standard_cartesian_state_element_blocks, 1.0E-6, 1.0E-6),
        step_size_validation_settings=propagation_setup.integrator.step_size_validation(0.001, 1000.0)
    )
    
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    
    return acceleration_models, integrator_settings, termination_settings

def simulate_and_calculate_residuals(velocity_guess, position_initial, simulation_start_epoch, 
                                     bodies, acceleration_models, integrator_settings, termination_settings, 
                                     real_data_inertial):
    """
    Fast cost function. Receives pre-built models and only updates the initial state.
    Calculates the residuals.
    """
    initial_state = np.concatenate([position_initial, velocity_guess])
    initial_state_translation = initial_state.reshape(-1, 1)

    #create propagator settings 
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=["Earth"], acceleration_models=acceleration_models, 
        bodies_to_integrate=["GOCE"], initial_states=initial_state_translation, 
        initial_time=simulation_start_epoch, integrator_settings=integrator_settings, 
        termination_settings=termination_settings
    )

    #run simulation
    dynamics_simulator = dynamics.simulator.create_dynamics_simulator(bodies, propagator_settings)
    
    #calculate residuals
    states_array = result2array(dynamics_simulator.propagation_results.state_history)
    if states_array.size == 0: return 1e9

    sim_epochs = states_array[:, 0]
    sim_pos = states_array[:, 1:4]
    
    real_epochs = real_data_inertial['epochs']
    mask = (real_epochs >= sim_epochs[0]) & (real_epochs <= sim_epochs[-1])
    if not np.any(mask): return 1e9
    
    valid_real_epochs = real_epochs[mask]
    valid_real_pos = np.vstack((real_data_inertial['x'][mask], real_data_inertial['y'][mask], real_data_inertial['z'][mask])).T
    
    #interpolate
    interp_func = interp1d(sim_epochs, sim_pos, axis=0, kind='linear')
    sim_pos_interp = interp_func(valid_real_epochs)
    
    #RMS Error
    diff = sim_pos_interp - valid_real_pos
    rms_error = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rms_error

def optimize_initial_velocity(initial_state_guess, simulation_start_epoch, simulation_end_epoch, bodies, config, real_data_inertial):
    """
    Uses minimize from scipy.optimize to reduce the RMS error, uses the initial guess give by our two point approx earlier.
    """
    
    print("STARTING ORBIT DETERMINATION (scipy minimize)")
    print("Pre-building physics models...")
    
    #build models ONCE
    accel_models, int_settings, term_settings = setup_models_for_optimization(bodies, config, simulation_end_epoch)
    
    pos_fixed = initial_state_guess[0:3]
    vel_guess = initial_state_guess[3:6]
    
    #define the loop arguments
    args = (pos_fixed, simulation_start_epoch, bodies, accel_models, int_settings, term_settings, real_data_inertial)

    def objective_wrapper(v):
        error = simulate_and_calculate_residuals(v, *args)
        return error

    #run optimiser
    print("Optimizing velocity vector...")
    result = minimize(objective_wrapper, vel_guess, method='Nelder-Mead', options={'maxiter': 100, 'disp': True})
    
    print('------------------------------------------------------')
    print("OPTIMIZATION COMPLETE")
    print(f"Best RMS Error: {result.fun:.2f} m")
    print(f"Optimized Velocity: {result.x}")
    print('------------------------------------------------------')

    return np.concatenate([pos_fixed, result.x])

def main():
    try:
        #load Configuration & Kernels
        config = load_config('parameters.ini')
        spice.clear_kernels()
        spice.load_standard_kernels()
        
        #load Data
        raw_epochs, raw_mass_values = load_and_interpolate_mass_data(config)
        real_orbit_data = load_real_orbit_data("goce_orbit_data.csv")
    
        #define Simulation Time (align with CSV)
        if real_orbit_data is not None:
             # Start exactly when the CSV starts
             simulation_start_epoch = real_orbit_data['epochs'][0]
             
             #end time from config (+1 day logic)
             cfg_time = config['SIM_TIME']
             simulation_end_epoch = DateTime(
                 cfg_time.getint('end_year'), cfg_time.getint('end_month'), 
                 cfg_time.getint('end_day') + 1, 0, 0, 0).to_epoch()
             
             print(f"Simulation Start Epoch aligned to CSV: {simulation_start_epoch}")
        else:
             #fallback if no CSV
             cfg_time = config['SIM_TIME']
             simulation_start_epoch = DateTime(
                 cfg_time.getint('start_year'), cfg_time.getint('start_month'), 
                 cfg_time.getint('start_day'), cfg_time.getint('start_hour'), 
                 cfg_time.getint('start_minute'), cfg_time.getint('start_second')).to_epoch()
             simulation_end_epoch = DateTime(
                 cfg_time.getint('end_year'), cfg_time.getint('end_month'), 
                 cfg_time.getint('end_day'), cfg_time.getint('end_hour'), 
                 cfg_time.getint('end_minute'), cfg_time.getint('end_second')).to_epoch()

        #setup Environment
        bodies = setup_environment(simulation_start_epoch, config)
        
        #ORBIT DETERMINATION (Optimization)
        if real_orbit_data is not None:
            #get Rough Guess from CSV
            rough_initial_state, _ = get_initial_state_from_csv(real_orbit_data, bodies)
            
            #transform Real Data to Inertial (Needed for Optimizer)
            real_data_inertial = transform_real_data_to_inertial(real_orbit_data)
            
            #run Optimization (Refine Velocity on a 2-hour short arc)
            opt_end_epoch = simulation_start_epoch + 2 * 3600 
            optimized_state = optimize_initial_velocity(
                rough_initial_state, simulation_start_epoch, opt_end_epoch, 
                bodies, config, real_data_inertial
            )
            
            #use the Optimized State for the Final Simulation
            initial_state_final = optimized_state
        else:
            initial_state_final = None
            
        mu_earth = bodies.get("Earth").gravitational_parameter
        kep_final = element_conversion.cartesian_to_keplerian(initial_state_final, mu_earth)
        
        print("Final optimised orbital elements:")
        print(f"Semi-major Axis:     {kep_final[0]:.2f} m")
        print(f"Eccentricity:        {kep_final[1]:.6f}")
        print(f"Inclination:         {np.rad2deg(kep_final[2]):.4f} deg")
        print(f"Arg of Periapsis:    {np.rad2deg(kep_final[3]):.4f} deg")
        print(f"RAAN:                {np.rad2deg(kep_final[4]):.4f} deg")
        print(f"True Anomaly:        {np.rad2deg(kep_final[5]):.4f} deg")
        print('------------------------------------------------------')

        
        #setup final propagation (full 24 hrs)
        print("Setting up final high-precision simulation...")
        propagator_settings, bodies_to_propagate = setup_propagation(
            bodies, simulation_start_epoch, simulation_end_epoch, config, initial_state_final)
        
        #run simulation
        dynamics_simulator = run_simulation(bodies, propagator_settings)

        #plot standard results (3D Orbit, Ground Track)
        process_and_plot_results(
            dynamics_simulator, simulation_start_epoch, simulation_end_epoch, 
            bodies_to_propagate, raw_epochs, raw_mass_values, config,
            real_orbit_data=real_orbit_data
        )
        
        #plot residuals
        if real_orbit_data is not None:
            #extract results for comparison
            states = dynamics_simulator.propagation_results.state_history
            states_array = result2array(states)
            
            if states_array.size > 0:
                sim_epochs = states_array[:, 0]
                sim_pos = states_array[:, 1:4]
                
                #use the inertial data we already calculated
                cfg_plot = config['PLOTTING']
                calculate_and_plot_residuals(sim_epochs, sim_pos, real_data_inertial, cfg_plot.getint('dpi'))

    except Exception as e:
        print("--- AN ERROR OCCURRED ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
