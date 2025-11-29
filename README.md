# GOCE Satellite Orbit Propagation Simulation

## Overview

This project performs an orbit propagation of the **GOCE** (*Gravity Field and Steady-State Ocean Circulation Explorer*) satellite using the `tudatpy` orbital mechanics library and compares it to data available from the ESA website here [ESA GOCE Mass Properties](https://earth.esa.int/eogateway/missions/data/goce-mass-properties).

A key feature of this simulation is the realistic modelling of the satellite's mass. Instead of a constant mass, this script loads real mission data from the `GOCE-Mass-Properties.txt` file, interpolates it, and uses the resulting function to model the satellite's decreasing mass over time (due to propellant consumption).
It also takes into account orbital pertubations, modelling spherical harmonic gravity using the gcrs_to_itrs rotation model for the Earth and taking into account the atmosphere and aerodynamic drag- important since GOCE is an LEO satellite. I have just added in Solar Radiation Pressure (SRP) too, currently modelling GOCE as a cannonball- I will likely change this to a panelled target in future versions. Relativistic correction to acceleration has been added for completeness, it's effect is minimal- uses Schwarzchild and de Sitter contributions. 

**IMPORTANT:**  to run the `varying_mass_GOCE_plot.py` script you need to download the `parameters.ini` file (containing the parameters) and `GOCE-Mass-Properties` (containting the mass data). `parameters.ini` acts as the configuration file. 


**EVEN MORE IMPORTANT:** I have done quite a bit of work behind the scenes and haven't updated this README... oops! Anyway, I downloaded actual GOCE data from [ESA GOCE Level 2](https://earth.esa.int/eogateway/catalog/goce-level-2), specifically the SST_PSO_2_ data. Since I only have 4GB RAM I wrote some code to process this file (`create_actual_data_file.py`), and generate a csv file (`goce_orbit_data.csv`) that will contain around 1% of the positions from the GOCE data. I have uploaded an example for ease of use. I then plotted this actual GOCE data onto my existing plots. I realised at this point that using the parameters that I initially found were not going to work- so I used the first two data points to estimate the initial velocity and hence the initial orbital elements as a best guess, then used this best guess to calculate an impoved initial velocity using minimize from scipy.optimize. Then I performed a nominal and perturbed simulation and used a linear Least Squares fit to get an improved drag coefficient. I then did the simulation using this optimised initial velocity and drag coefficient. I then calculated and plotted the residuals. This updated code is in `comparison_plots.py`, and this is the one that you should run!

## Key Features

* **Orbit Propagation**: Uses tudatpy to simulate the satellite's trajectory.
* **Variable Mass Modelling**: Loads and interpolates real GOCE mass data to provide a time-varying mass to the propagator. This is cool since thrust and drag have a greater effect as mass decreases, and we are modelling this decrease in mass.
* **Monotonic Interpolation**: Uses `scipy.interpolate.PchipInterpolator` to ensure the mass function is monotonically decreasing, since there are some long gaps in our mass data that can affect interpolation.
* **Comprehensive Force Model**:
    * **Earth**: Spherical harmonic gravity (J2) (up to degree 12, order 6), taking into account the fact that the Earth is not perfectly spherical.
    * **Moon & Sun**: Third bodies, considers as point-mass sources.
    * **Atmosphere**: `NRLMSISE-00` atmospheric model for Earth.
    * **Aerodynamics**: Constant drag and lift coefficients to quantify effect of atmosphere on LEO.
    * **Solar Radiation Pressure (SRP)**: Models the acceleration on the satellite due to photons from Sun transferring momentum.
    * **Relativistic Correction**: Models the Schwarzchild and de Sitter contributions to GOCE acceleration.
* **Detailed plots of data**: Generates four key plots using `matplotlib`:
    1.  **3D Trajectory**: A 3D plot of the GOCE orbit around the Earth.
    2.  **Ground Track**: A 2D plot of the satellite's ground track for the first 3 hours, easily changeable.
    3.  **Mass vs. Time**: A plot showing the satellite's mass decreasing over the simulation period, overlaid with the original raw data points from the mass file to validate the interpolation.
    4.  **Actual data**: Plots the actual GOCE trajectory data and calculates residuals for side by side comparison. Optimises velocity and drag coefficient.

## Things Used

* **Python 3**
* **`tudatpy`**: For all orbital mechanics, environment setup, and propagation.
* **`scipy`**: For mass data interpolation: PchipInterpolator for mass, cubic spline for actual GOCE positions, linear for actual times. 
* **`pandas`**: For loading and parsing the mass data file.
* **`numpy`**: For numerical operations and data handling.
* **`matplotlib`**: For all data visualization.
* **`configparser`**: To run `parameters.ini` as the configuration file.

## How to Run

### 1. Prerequisites

Ensure you have Python 3 and the following libraries installed: `tudatpy`, `pandas`, `numpy`, `matplotlib`, `scipy`. 

You are going to need to be in the tudat-space environment. For more information on tudat installation, see here: [tudat installation](https://docs.tudat.space/en/latest/getting-started/installation.html#getting-started-installation). Note that the tudat environment comes with astropy already installed- yay!

### 2. SPICE Kernels

The script uses `spice.load_standard_kernels()` to load standard SPICE kernels. `tudatpy` will typically download these automatically if they are not found.

### 3. Execute the Script

You will need to download the `comparison_plots.py` script, alongside the `parameters.ini` file, `GOCE-Mass-Properties.txt` and `GOCE_orbit_data.csv`. You can change the parameters directly within this .ini file, and run whatever orbit data you want- the one in this reposititory is given as an example. Make sure you are in the tudat-space envionment.

### 4. View Results

The script will print simulation details to the console, including initial/final mass, 'best guess' velocity and orbital elements, optimised initial velocity and orbital elements, and lots of reassuring messages to make you feel better if it takes a long time to run (or error messages to help you troubleshoot)! It will then display the plots (3D trajectory, ground track, and mass over time, and residuals if you are using the comparison plots). Congratulations, it worked!

*Top tip: I would recommend that you don't use inline plots, instead use interactive matplotlib plotting so you can see the orbits properly.*

### 5. Example Plots

The below plots are an example of this new comparison code- they use data from the GO_CONS_SST_PSO_2__20091001T235945_20091002T235944_0201.DBL, which has been processed and is in this repository as `goce_orbit_data.csv`- data from the 1st October 2009. They use the parameters and mass data from this repository. The Mean Position Error (residual) was 25572.94 m for this example. You can see visually that our period is a bit off-look at the curvy residual plot- but that the approximation is relatively good- the blue and red are combining to make purple! *NOTE: this just used velocity optimisation*
![Example plots](EXAMPLE_PLOTS.png)

**Improved plots** now using velocity AND drag coefficient optimisation! The Mean Position Error is now **6645.77 m** which is way better! You can see interesting periodicity still in the residual plot- I wanted to add in an extra step: after finding new cd, optimise velocity again using drag. This would reduce a lot of this error drift. Unfortunately, I lack the computer power- I left it overnight and tried to run 50 iterations. It sadly did not run, so we are going to have periodic and increasing residuals over time since the period is a little off. 
![Improved_Example_Plots](IMPROVED_EXAMPLE_PLOTS.png)

*Note: not all numbers are accurate to GOCE, some are placeholders/approximations for the meantime. I am doing plenty of research, some of this stuff is a little tricky to find!*
