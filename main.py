# main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Utility Functions ===

def gen_time_func(temp):
    temp_points = [30, 33, 36, 37, 38, 39]
    gen_times = [90, 75, 110, 120, 130, 180]
    return np.interp(temp, temp_points, gen_times)

def jump_prob_func(mother_division_time):
    min_division_time = 30
    max_division_time = 200
    p_jump_min = 0.015
    p_jump_max = 0.4
    mother_division_time = np.clip(mother_division_time, min_division_time, max_division_time)
    return p_jump_min + (p_jump_max - p_jump_min) * (mother_division_time - min_division_time) / (max_division_time - min_division_time)

def determine_gfp(mother_gfp, mother_division_time, use_gaussian_inheritance=False):
    if use_gaussian_inheritance:
        # Continuous inheritance with slight noise
        return mother_gfp + np.random.normal(0, 0.1)

    # Original binary jumping logic
    p_jump = jump_prob_func(mother_division_time)
    if mother_gfp < 2:
        return 3 + np.random.normal(0, 0.1) if np.random.random() < p_jump else 1 + np.random.normal(0, 0.1)
    else:
        return 1 + np.random.normal(0, 0.1) if np.random.random() < p_jump else 3 + np.random.normal(0, 0.1)


def feedback_temperature(mean_gfp, mode="exp", base_temp=30, max_temp=39, sensitivity=1.0):
    if mode == "exp":
        return base_temp + np.exp(-mean_gfp / sensitivity) * (max_temp - base_temp)
    elif mode == "linear":
        return max_temp - (mean_gfp / 3.0) * (max_temp - base_temp)
    elif mode == "sigmoid":
        return base_temp + (max_temp - base_temp) / (1 + np.exp(-sensitivity * (mean_gfp - 2)))
    elif mode == "step":
        return base_temp if mean_gfp > 2 else max_temp
    elif mode == "inverse":
        return base_temp + (mean_gfp / 3.0) * (max_temp - base_temp)
    else:
        raise ValueError("Unknown feedback mode")

def background_switching(cell, temp, max_switch_rate=0.1):
    P_switch = max(0, min(max_switch_rate * (temp - 30) / 9, max_switch_rate))
    if np.random.random() < P_switch:
        if cell.gfp < 2:
            cell.gfp = 3 + np.random.normal(0, 0.1)
        else:
            cell.gfp = 1 + np.random.normal(0, 0.1)

def fitness(cell, temp, alpha=0.1, gfp_penalty=0.2):
    temp_bonus = 1 + alpha * (30 - temp)
    return temp_bonus * (1 - gfp_penalty) if cell.gfp > 2 else temp_bonus

# === Cell Class ===

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


# === Main Simulation ===

def run_simulation(
    total_time=1000,
    time_step=1,
    feedback_mode="exp",
    max_population=1000,
    enable_background_switching=True,
    use_fitness_weighted_reproduction=False,
    max_switch_rate=0.1,
    feedback_sensitivity=1.0,
    control_mode=False,
    use_gaussian_inheritance=False
):
    division_times = [0]
    temperature_history, temperature_times = [], []
    generation_time_history, gfp_history, time_history = [], [], []
    division_data = []

    initial_gfp = 1
    population = [Cell(30, 0, 0, initial_gfp)]

    base_temp, max_temp = 30, 39
    current_time = 0

    while current_time < total_time:
        if current_time % 30 == 0:
            mean_gfp = np.mean([cell.gfp for cell in population])
            if control_mode:
                temperature = max_temp
            else:
                temperature = feedback_temperature(
                    mean_gfp, mode=feedback_mode, base_temp=base_temp, max_temp=max_temp, sensitivity=feedback_sensitivity
                )
            temperature_history.append(temperature)
            temperature_times.append(current_time)

        gfp_history.append(np.mean([cell.gfp for cell in population]))
        generation_time_history.append(np.mean([cell.generation_time for cell in population]))
        time_history.append(current_time)

        for cell in population:
            cell.update_generation_time(temperature)

            if enable_background_switching:
                background_switching(cell, temperature, max_switch_rate)

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

        while len(population) > max_population:
            if use_fitness_weighted_reproduction:
                fitnesses = np.array([fitness(cell, temperature) for cell in population])
                probs = fitnesses / fitnesses.sum()
                idx = np.random.choice(len(population), p=probs)
            else:
                idx = np.random.randint(len(population))
            population.pop(idx)

        current_time += time_step

    df = pd.DataFrame(division_data)
    return df, division_times, temperature_history, temperature_times, gfp_history, generation_time_history, time_history


def run_simulation_with_external_temperature(
    external_temp_times,
    external_temp_values,
    total_time=1000,
    time_step=1,
    max_population=1000,
    enable_background_switching=True,
    use_fitness_weighted_reproduction=False,
    max_switch_rate=0.1,
    use_gaussian_inheritance=False
):
    division_times = [0]
    generation_time_history, gfp_history, time_history = [], [], []
    division_data = []

    initial_gfp = 1
    population = [Cell(30, 0, 0, initial_gfp)]

    current_time = 0

    while current_time < total_time:
        # Use interpolated external temperature at this time
        temperature = np.interp(current_time, external_temp_times, external_temp_values)

        gfp_history.append(np.mean([cell.gfp for cell in population]))
        generation_time_history.append(np.mean([cell.generation_time for cell in population]))
        time_history.append(current_time)

        for cell in population:
            cell.update_generation_time(temperature)

            if enable_background_switching:
                background_switching(cell, temperature, max_switch_rate)

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

        while len(population) > max_population:
            if use_fitness_weighted_reproduction:
                fitnesses = np.array([fitness(cell, temperature) for cell in population])
                probs = fitnesses / fitnesses.sum()
                idx = np.random.choice(len(population), p=probs)
            else:
                idx = np.random.randint(len(population))
            population.pop(idx)

        current_time += time_step

    df = pd.DataFrame(division_data)
    return df, division_times, gfp_history, generation_time_history, time_history


# === Main Function ===
def main():
    num_passive_wells = 5  # You can set this to 99 later
    use_gaussian_inheritance = False  # or True if testing from CLI


    # === Step 1: Run DRIVER (feedback-coupled) well ===
    df_driver, div_driver, temp_driver, temp_t_driver, gfp_driver, gen_driver, time_driver = run_simulation(
        feedback_mode="linear",
        feedback_sensitivity=1.0,
        max_switch_rate=0.1,
        control_mode=False,
        use_gaussian_inheritance=use_gaussian_inheritance
    )

    # === Step 2: Run multiple PASSIVE wells with shared temperature ===
    passive_results = []
    for i in range(num_passive_wells):
        df_passive, div_passive, gfp_passive, gen_passive, time_passive = run_simulation_with_external_temperature(
            external_temp_times=temp_t_driver,
            external_temp_values=temp_driver,
            max_switch_rate=0.1,
            use_gaussian_inheritance=use_gaussian_inheritance
        )
        passive_results.append({
            'df': df_passive,
            'div': div_passive,
            'gfp': gfp_passive,
            'gen': gen_passive,
            'time': time_passive
        })

    # === Step 3: Run CONTROL well (fixed at 39°C) ===
    df_control, div_control, temp_control, temp_t_control, gfp_control, gen_control, time_control = run_simulation(
        max_switch_rate=0.1,
        control_mode=True
    )

    # === Plotting aggregated passive results ===
    avg_gfp_passive = np.mean([res['gfp'] for res in passive_results], axis=0)
    avg_gen_passive = np.mean([res['gen'] for res in passive_results], axis=0)
    avg_div_passive = np.mean([len(res['div']) for res in passive_results])
    combined_df_passive = pd.concat([res['df'] for res in passive_results])

    fig, axs = plt.subplots(5, 1, figsize=(10, 24))

    # Cumulative divisions
    axs[0].plot(div_driver, np.arange(len(div_driver)), label="Driver (Feedback)", color='blue')
    axs[0].plot([], [], label=f"Passive x{num_passive_wells} (avg)", color='green')  # Legend only
    for res in passive_results:
        axs[0].plot(res['div'], np.arange(len(res['div'])), color='green', alpha=0.3)
    axs[0].plot(div_control, np.arange(len(div_control)), label="Control (Fixed)", color='gray', linestyle='--')
    axs[0].set_title("Cumulative Divisions")
    axs[0].legend()
    axs[0].grid(True)

    # Temperature
    axs[1].plot(temp_t_driver, temp_driver, label="Temperature (Driven)", color='red')
    axs[1].set_title("Temperature Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # Generation time
    axs[2].plot(time_driver, gen_driver, label="Driver", color='blue')
    axs[2].plot(time_driver, avg_gen_passive, label=f"Passive (avg of {num_passive_wells})", color='green')
    axs[2].plot(time_control, gen_control, label="Control", color='gray', linestyle='--')
    axs[2].set_title("Generation Time")
    axs[2].legend()
    axs[2].grid(True)

    # GFP
    axs[3].plot(time_driver, gfp_driver, label="Driver", color='blue')
    axs[3].plot(time_driver, avg_gfp_passive, label=f"Passive (avg of {num_passive_wells})", color='green')
    axs[3].plot(time_control, gfp_control, label="Control", color='gray', linestyle='--')
    axs[3].set_title("Mean GFP Expression")
    axs[3].legend()
    axs[3].grid(True)

    # Final GFP Distribution
    sns.histplot(df_driver['GFP'], bins=100, ax=axs[4], color='blue', label='Driver', stat="density")
    sns.histplot(combined_df_passive['GFP'], bins=100, ax=axs[4], color='green', label='Passive (Combined)', stat="density", alpha=0.5)
    sns.histplot(df_control['GFP'], bins=100, ax=axs[4], color='gray', label='Control', stat="density", alpha=0.4)
    axs[4].set_title("Final GFP Distribution")
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Driver divisions: {len(div_driver)-1}")
    print(f"Avg Passive divisions (n={num_passive_wells}): {int(avg_div_passive)}")
    print(f"Control divisions: {len(div_control)-1}")
