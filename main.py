#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib 

def gen_time_func(temp):
    temp_points = [30, 33, 36, 37, 38, 39]
    gen_times = [90, 75, 110, 120, 130, 180]
    return np.interp(temp, temp_points, gen_times)

def jump_prob_func(mother_division_time):
    min_division_time = 30
    max_division_time = 200
    p_jump_min = 0.015
    p_jump_max = 0.4
    
    # Ensure the mother's division time is within the specified range
    mother_division_time = max(min_division_time, min(mother_division_time, max_division_time))
    
    # Calculate the jump probability based on the mother's division time
    p_jump = p_jump_min + (p_jump_max - p_jump_min) * (mother_division_time - min_division_time) / (max_division_time - min_division_time)
    
    return p_jump

def dec_exp(x, range_Temp, base_Temp, curve):
    if x == 0:
        y_value = base_Temp + 0.5
        y_crit = (y_value - base_Temp) / range_Temp
        x_crit = -curve * np.log(y_crit)
        print('Multiplication of F needed to reach 0.5 a degree from target is: ' + str(x_crit))
    y = np.exp(-x / curve)
    return base_Temp + y * range_Temp

class Cell:
    def __init__(self, temp, time_of_birth, mother_division_time, gfp):
        self.generation_time = gen_time_func(temp)
        self.time_of_birth = time_of_birth
        self.mother_division_time = mother_division_time
        self.time_since_division = 0
        self.gfp = gfp

    def update_generation_time(self, temp):
        self.generation_time = gen_time_func(temp)

    def divide(self, time_since_division, current_time):
        scale_factor = 0.015
        p_division = 1 - np.exp(-scale_factor * time_since_division / self.generation_time)
        if np.random.random() < p_division:
            division_time = self.time_since_division
            self.time_since_division = 0
            return True, division_time, self.generation_time, current_time
        else:
            return False, None, None, None

def determine_gfp(mother_gfp, mother_division_time):
    p_jump = jump_prob_func(mother_division_time)
    if mother_gfp < 2:  # Assume X is around 1 and 3X is around 3
        if np.random.random() < p_jump:
            return 3 + np.random.normal(0, 0.1)  # Jump to 3X
        else:
            return 1 + np.random.normal(0, 0.1)  # Stay at X
    else:
        if np.random.random() < p_jump:
            return 1 + np.random.normal(0, 0.1)  # Jump to X
        else:
            return 3 + np.random.normal(0, 0.1)  # Stay at 3X

# Set the simulation time parameters
total_time = 1000  # Total simulation time in minutes
time_step = 1  # Time step for the simulation (1 minute)

# Initialize variables to store the division times, temperatures, and generation times
division_times = [0]  # Start with the initial division at time 0
temperature_history = []
temperature_times = []
generation_time_history = []
gfp_history = []
time_history = []
division_data = []

# Create the initial cell with the starting temperature and GFP level
initial_gfp = 1
initial_cell = Cell(30, 0, 0, initial_gfp)
population = [initial_cell]

# Set the reward curve parameters
range_Temp = 22
base_Temp = 30
curve = 1.1

# Run the simulation
current_time = 0
while current_time < total_time:
    # Update the temperature every 30 time points based on the mean GFP of the population
    if current_time % 30 == 0:
        mean_gfp = np.mean([cell.gfp for cell in population])
        temperature = dec_exp(mean_gfp, range_Temp, base_Temp, curve)
        temperature_history.append(temperature)
        temperature_times.append(current_time)
        
    if len(population) > 1000:
        # Randomly select a cell to remove
        index = np.random.randint(len(population))
        population.pop(index)
    
    # Store the average GFP level and generation time at each time step
    gfp_history.append(np.mean([cell.gfp for cell in population]))
    generation_time_history.append(np.mean([cell.generation_time for cell in population]))
    time_history.append(current_time)
    
    for cell in population:
        cell.update_generation_time(temperature)

        divided, division_time, gen_time, division_time_point = cell.divide(cell.time_since_division, current_time)
        if divided:
            division_times.append(current_time)
            daughter_gfp = determine_gfp(cell.gfp, cell.mother_division_time)
            new_cell = Cell(temperature, division_time_point, division_time, daughter_gfp)
            population.append(new_cell)
            division_data.append({
                'Temperature': temperature,
                'Division Time': division_time,
                'Generation Time': gen_time,
                'Time of Birth': new_cell.time_of_birth,
                'Time Since Mother Division': new_cell.mother_division_time,
                'GFP': new_cell.gfp
            })

        cell.time_since_division += time_step

    # Implement Moran process if population size exceeds 1000
    if len(population) > 1000:
        # Randomly select a cell to remove
        index = np.random.randint(len(population))
        population.pop(index)

    current_time += time_step

# Create a DataFrame with the division data
df = pd.DataFrame(division_data)

# Print the DataFrame
print(df)

# Create a figure with subplots
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 20))

# Plot the division events over time
ax1.plot(division_times, np.arange(len(division_times)), 'bo-')
ax1.set_ylabel('Cumulative Division Events')
ax1.set_title('Cell Division Events')
ax1.grid(True)

# Plot the temperatures over time
ax2.plot(temperature_times, temperature_history, 'r-')
ax2.set_ylabel('Temperature (°C)')
ax2.set_title('Temperatures over Time')
ax2.grid(True)

# Plot the generation times over time
ax3.plot(time_history, generation_time_history, 'g-')
ax3.set_ylabel('Generation Time (min)')
ax3.set_title('Generation Times over Time')
ax3.grid(True)

# Plot the GFP levels over time
ax4.plot(time_history, gfp_history, 'mo-')
ax4.set_xlabel('Time (minutes)')
ax4.set_ylabel('Mean GFP Level')
ax4.set_title('Mean GFP Levels over Time')
ax4.grid(True)

sns.histplot(df['GFP'], bins=100, ax=ax5)
ax5.grid(True)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Print the total number of divisions
total_divisions = len(division_times) - 1  # Subtract 1 to exclude the initial division at time 0
print(f"Total number of divisions: {total_divisions}")
