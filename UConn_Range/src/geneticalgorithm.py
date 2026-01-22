# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:35:54 2026

@author: SchollJamesAC3CARILL
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pygad
from datetime import datetime
from pathlib import Path
from Shared.PiController import PiController
from Shared.NSI2000Client import NSI2000Client

#Place to store experiment results
EXP_DIR = r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments"

SCAN_FILENAME = r"C:\NSI2000\Data\Carillon\calibration_scan_real.nsi"

#PHASE_MAP_FILE_LB = r"C:\Users\labuser\Documents\ReflecTekCalibrationScholl\phases_with_beam_steering_0theta_0phi_hex_12x8.txt" 

PI_HOST = "192.168.6.30" #IP of PI controlling DACs
USERNAME = "feix"         
PASSWORD = "password"          
KEY_FILE = None # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"
PI_PORT = 22

STOP_FILE = r"/home/feix/STOP.txt"
LOCAL_FILE_HB = r"C:\NSI2000\Data\Carillon\HB_voltages.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = r"C:\NSI2000\Data\Carillon\LB_voltages.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/feix/Desktop/dataHB.csv"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/feix/Downloads/2025-12-18 VoltageMap_HornCorrection.csv"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/feix/Gen3DAC60096EVM_SPI_RPi5_scholl.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"python3 {REMOTE_PROGRAM}"

LC_DELAY_TIME =40 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 1
SIZE = (12, 8)

#Loss function params
MAIN_LOBE_HALF_WIDTH = 2 #Number of points in scan for the main lobe half width
CENTER_INDEX = 71 #Index where the center lobe should be
GUARD_BAND_HALF_WIDTH = 4 #Number of points in the scan for a guard band not considered in loss function

def update_lb_array_file(V):
    #V = np.round(V * DAC_MIN_STEP_SIZE, 3)
    with open(LOCAL_FILE_LB, "w") as f:
        for row in V:
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")
    #This program is for LB only, so create a 0V array for the high band which is 24x8 in Sam's code
    with open(LOCAL_FILE_HB, "w") as f:
        for row in np.zeros((24,8)):
            line = ",".join(str(x) for x in row)
            f.write(line + "\n")

def loss_center_vs_sidelobes_db(
    amp_db,
    center_idx: int,
    main_half_width: int = 2,
    guard_half_width: int = 10,
    loss_equation: int =  0,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-15,
):
    """
    Scalar loss using VNA amplitude in dB (dB magnitude).

    Goal: maximize energy near known center_idx, minimize energy elsewhere (sidelobes).

    loss = E_main / (E_main + E_side) or
    

    - E_main: sum of linear power in [center-main_half_width, center+main_half_width]
    - E_side: sum of linear power outside a guard band
              [center-guard_half_width, center+guard_half_width]

    Returns
    -------
    loss : float
    """
    amp_db = np.asarray(amp_db, dtype=float)
    N = amp_db.size
    c = int(np.clip(center_idx, 0, N - 1))

    if guard_half_width < main_half_width:
        guard_half_width = main_half_width

    # dB magnitude -> linear power
    # If amp_db is dB magnitude: mag_lin = 10^(dB/20), power = mag_lin^2 = 10^(dB/10)
    pwr = 10.0 ** (amp_db / 10.0)

    # main window
    m0 = max(0, c - main_half_width)
    m1 = min(N, c + main_half_width + 1)

    # guard band (excluded from sidelobe calculation)
    g0 = max(0, c - guard_half_width)
    g1 = min(N, c + guard_half_width + 1)

    main_mask = np.zeros(N, dtype=bool)
    main_mask[m0:m1] = True

    side_mask = np.ones(N, dtype=bool)
    side_mask[g0:g1] = False  # everything outside guard is "sidelobes"
    #side_mask[g0:] = False  # everything outside guard is including all positive indices which are closest to horn (from milad- it interferes)
    
    E_main = float(np.sum(pwr[main_mask]))
    E_side = float(np.sum(pwr[side_mask]))

    fitness = E_main / (E_main + E_side)
    
    #lm = .25
    #loss = -E_main - lm * E_side
    
    return fitness

patterns = []
def make_fitness(vna_instance, rpi):
    
    def fitness_func(ga_instance, solution, solution_idx):
        global patterns
        voltages = np.tile(solution[:,None], (1,8))
        update_lb_array_file(voltages)
        #sends low and high band array files to PI and runs remote command to update DACs
        rpi.update_dacs()
        time.sleep(LC_DELAY_TIME)
        pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
        patterns.append(pattern)
        fitness = loss_center_vs_sidelobes_db(pattern, CENTER_INDEX, MAIN_LOBE_HALF_WIDTH, GUARD_BAND_HALF_WIDTH)
        return fitness
    
    return fitness_func

last_fitness = 0
best_fitness = 0
best_patterns = []
def on_gen(ga_instance):
    global last_fitness
    global best_fitness
    global patterns
    global best_patterns
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", solution_fitness)
    print(f"Change = {solution_fitness - last_fitness}")
    if solution_fitness > best_fitness:
        best_fitness = solution_fitness
        best_patterns.append(patterns[solution_idx])
        patterns = []
        plt.figure()
        for i, p in enumerate(best_patterns):
            plt.plot(p, label=f"Pattern {i}")
        plt.xlabel("Span -10 to +10in")
        plt.ylabel("Mag (dB)")
        plt.title("Mag vs Span best")
        plt.grid(True)
        plt.legend()
        plt.savefig(exp_folder/ "MagVsSpanBest.png", dpi=200)
        plt.show()
    last_fitness = solution_fitness

num_generations = 20
num_parents_mating = 10 #number of parent params chosen for breeding

fitness_func = make_fitness

sol_per_pop = 20 #population size
num_genes = 12 

gene_space = [{"low": 0.0, "high": 10.49487305}] * 12

parent_selection_type = "tournament"
K_tournament = 3

keep_parents = 2
keep_elitism = 1

crossover_type = "single_point"
crossover_probability = None

mutation_type = "random"
mutation_probability = None
mutation_percent_genes = 10

initial_population = None

experiment_dir = Path(r"C:\NSI2000\Data\Carillon\reflectarray_calibration\Experiments\genetic algorithm")
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_folder = experiment_dir / f"Calibration_{ts}"
exp_folder.mkdir(parents = True, exist_ok = False)

ga = pygad.GA(
    num_generations= num_generations,
    sol_per_pop= sol_per_pop,
    num_parents_mating= num_parents_mating,

    num_genes=num_genes,
    gene_space=gene_space,

    fitness_func=fitness_func,
    on_generation=on_gen,

    parent_selection_type= parent_selection_type,
    crossover_type= crossover_type,
    crossover_probability= crossover_probability,

    mutation_type= mutation_type,
    mutation_probability= mutation_probability,
    mutation_percent_genes= mutation_percent_genes,

    keep_parents= keep_parents,
    keep_elitism= keep_elitism,

    initial_population=initial_population,

    # Helpful: avoid repeated parents too much (can help diversity)
    allow_duplicate_genes=True,   # genes are continuous; duplicates are fine
    random_mutation_min_val=-1, # mutation step range (volts) if using random mutation
    random_mutation_max_val=+1,
)
ga.run()

ga.plot_fitness()
solution, solution_fitness, solution_idx = ga.best_solution()
voltages = np.tile(solution[:,None], (1,8))
print(f"Voltages from the best solution : {voltages}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")


np.savez(exp_folder / "results.npz",
         best_voltages=np.array(voltages))

zero_volts = np.zeros(SIZE)
update_lb_array_file(zero_volts)
rpi.update_dacs()
time.sleep(LC_DELAY_TIME)
nsi.disconnect()
rpi.stop_program()
rpi.close()