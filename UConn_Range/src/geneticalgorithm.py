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

#PI_HOST = "192.168.6.30" #IP of PI controlling DACs
PI_HOST = "RasPI5.local"
PI_HOST = "10.194.115.111"
USERNAME = "feix"         
PASSWORD = "password"          
KEY_FILE = None # if using an SSH key, set path like "C:/Users/you/.ssh/id_rsa"
PI_PORT = 22

STOP_FILE = r"/home/feix/STOP.txt"
LOCAL_FILE_HB = r"C:\NSI2000\Data\Carillon\HB_voltages.txt"   #HB voltage file to send to PI
LOCAL_FILE_LB = r"C:\NSI2000\Data\Carillon\LB_voltages.txt" #LB voltage file to send to PI
REMOTE_FILE_HB = r"/home/feix/Desktop/dataHB.csv"  # where to put it on the Pi
REMOTE_FILE_LB = r"/home/feix/Downloads/2025-12-18 VoltageMap_HornCorrection.csv"  # where to put it on the Pi
REMOTE_PROGRAM = "/home/feix/Gen3DAC60096EVM_SPI_RPi5_schollV2.py" #Location of program on PI that updates DACs
# Command to run on the Pi once file is uploaded
REMOTE_COMMAND = f"nohup python3 {REMOTE_PROGRAM} >/dev/null 2>&1 &"

LC_DELAY_TIME =  40 #in secs

DAC_MIN_STEP_SIZE = float(21/4096) #DAC60096 12-bit +/-10.5

#Set the beam (corresponds to frequency measured) number that you put in NSI software. 19.3 Ghz is ideal for low band
BEAM = 1
SIZE = (12, 8)

#Loss function params
MAIN_LOBE_HALF_WIDTH = 2 #Number of points in scan for the main lobe half width
CENTER_INDEX = 50 #Index where the center lobe should be
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
    #fitness = E_main
    
    #lm = .25
    #loss = -E_main - lm * E_side
    
    return fitness

def sol_key(solution):
    return tuple(np.round(np.asarray(solution, dtype=float), 4))

pattern_cache = {}
fitness_cache = {}

def make_fitness(vna_instance, rpi):
    
    def fitness_func(ga_instance, solution, solution_idx):
        global pattern_cache, fitness_cache
        top_half = solution.reshape(6,8)
        bottom_half = np.flipud(top_half)
        voltages = np.vstack((top_half, bottom_half))
        
        update_lb_array_file(voltages)
        #sends low and high band array files to PI and runs remote command to update DACs
        rpi.update_dacs()
        time.sleep(LC_DELAY_TIME)
        pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)[0, :]
        fitness = loss_center_vs_sidelobes_db(pattern, CENTER_INDEX, MAIN_LOBE_HALF_WIDTH, GUARD_BAND_HALF_WIDTH)
        
        k = sol_key(solution)
        pattern_cache[k] = pattern
        fitness_cache[k] = fitness
        
        return fitness
    
    return fitness_func

last_fitness = 0
best_fitness = 0
best_patterns = []
best_fitnesses = []
best_voltages = []
fit_std = []
fit_range = []
gene_std_mean = []
gene_std_max = []
_div_fig = None
_div_axes = None

def diversity_stats(pop, fit):
    global fit_std, fit_range, gene_std_mean, gene_std_max
    global _div_fig, _div_axes
    pop = np.asarray(pop, float)
    fit = np.asarray(fit, float)
    s = {
        "fit_std": float(np.std(fit)),
        "fit_range": float(np.max(fit) - np.min(fit)),
        "gene_std_mean": float(np.mean(np.std(pop, axis=0))),
        "gene_std_max": float(np.max(np.std(pop, axis=0))),
    }
    fit_std.append(s["fit_std"])
    fit_range.append(s["fit_range"])
    gene_std_mean.append(s["gene_std_mean"])
    gene_std_max.append(s["gene_std_max"])
    if _div_fig is None or _div_axes is None:
        _div_fig, _div_axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        
    ax00, ax01 = _div_axes[0]
    ax10, ax11 = _div_axes[1]

    # Clear and redraw (simple + reliable)
    ax00.cla(); ax01.cla(); ax10.cla(); ax11.cla()

    ax00.plot(fit_std, marker="o")
    ax00.set_title("Fitness Std Dev")
    ax00.set_xlabel("Generation")
    ax00.set_ylabel("Std(fitness)")
    ax00.grid(True)

    ax01.plot(fit_range, marker="o")
    ax01.set_title("Fitness Range")
    ax01.set_xlabel("Generation")
    ax01.set_ylabel("max(f)-min(f)")
    ax01.grid(True)

    ax10.plot(gene_std_mean, marker="o")
    ax10.set_title("Mean Gene Std Dev")
    ax10.set_xlabel("Generation")
    ax10.set_ylabel("mean(std(gene))")
    ax10.grid(True)

    ax11.plot(gene_std_max, marker="o")
    ax11.set_title("Max Gene Std Dev")
    ax11.set_xlabel("Generation")
    ax11.set_ylabel("max(std(gene))")
    ax11.grid(True)

    _div_fig.suptitle("Population Diversity (updated each generation)")

    # Update the window without blocking
    _div_fig.canvas.draw_idle()
    plt.pause(0.001)

    # Optional: save a continuously-updated image each generation
    
    _div_fig.savefig(exp_folder / "pop_div.png", dpi=200)
        
    return s

def on_gen(ga_instance):
    global last_fitness, best_fitness, best_patterns, best_fitnesses, best_voltages
    global pattern_cache, fitness_cache
    ga_instance.plot_fitness()
    pop = ga_instance.population
    fit = ga_instance.last_generation_fitness
    diversity_stats(pop, fit)
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", solution_fitness)
    print(f"Change = {solution_fitness - last_fitness}")
    k = sol_key(solution)
    pattern = pattern_cache.get(k, None)
    fitness = fitness_cache.get(k, None)
    if solution_fitness > best_fitness:
        top_half = solution.reshape(6,8)
        bottom_half = np.flipud(top_half)
        best_voltages = np.vstack((top_half, bottom_half))
        print(best_voltages)
        best_fitness = solution_fitness
        best_patterns.append(pattern)
        best_fitnesses.append(fitness)
        pattern_cache = {}
        fitness_cache = {}
        plt.figure()
        for i, p in enumerate(best_patterns):
            plt.plot(p, label=f"Fitness: {np.round(best_fitnesses[i], 4)}")
        plt.xlabel("Span -50 to +50 Deg")
        plt.ylabel("Mag (dB)")
        plt.title("Mag vs Span best")
        plt.grid(True)
        plt.legend()
        plt.savefig(exp_folder/ "MagVsSpanBest.png", dpi=200)
        plt.show()
    last_fitness = solution_fitness


init = np.array([
    [7.16632648, 1.40225182, 0.52255433, 9.80566219, 4.55549761, 1.17902949, 1.60657237, 8.52700428],
    [0.98461356, 1.07840313, 0.00709581, 6.34536230, 0.01505323, 0.30802929, 1.73472426, 9.47118206],
    [8.75371717, 0.01917238, 6.89494569, 5.50262202, 0.11468081, 0.52827583, 1.76535917, 7.60306206],
    [5.06025948, 1.99034717, 7.96443552, 2.49324405, 5.61960198, 1.55343217, 4.04412824, 7.03406372],
    [6.21501433, 0.68856494, 2.34103129, 6.49821401, 1.05292961, 1.40057762, 1.03044921, 3.28496453],
    [8.11281286, 1.04946571, 1.37568752, 1.74290315, 8.03541736, 0.03354295, 8.56124450, 3.63949732],
    [8.11281286, 1.04946571, 1.37568752, 1.74290315, 8.03541736, 0.03354295, 8.56124450, 3.63949732],
    [6.21501433, 0.68856494, 2.34103129, 6.49821401, 1.05292961, 1.40057762, 1.03044921, 3.28496453],
    [5.06025948, 1.99034717, 7.96443552, 2.49324405, 5.61960198, 1.55343217, 4.04412824, 7.03406372],
    [8.75371717, 0.01917238, 6.89494569, 5.50262202, 0.11468081, 0.52827583, 1.76535917, 7.60306206],
    [0.98461356, 1.07840313, 0.00709581, 6.34536230, 0.01505323, 0.30802929, 1.73472426, 9.47118206],
    [7.16632648, 1.40225182, 0.52255433, 9.80566219, 4.55549761, 1.17902949, 1.60657237, 8.52700428]
], dtype=np.float64)
top_half=init[:6,:]
init_pop = top_half.reshape(-1)
initial_population = np.random.uniform(0, 10.4, size=(64, 48))
initial_population[0] = init_pop
nsi = NSI2000Client().connect()
rpi = PiController(
        host=PI_HOST,
        username=USERNAME,
        password=PASSWORD,
        local_file_hb=LOCAL_FILE_HB,
        local_file_lb=LOCAL_FILE_LB,
        remote_file_hb=REMOTE_FILE_HB,
        remote_file_lb=REMOTE_FILE_LB,
        remote_command=REMOTE_COMMAND,
        port = PI_PORT,
        key_filename=KEY_FILE,
        stop_file = STOP_FILE,
    )
rpi.connect()

num_generations = 30
num_parents_mating = 20 #number of parent params chosen for breeding

fitness_func = make_fitness(nsi, rpi)

sol_per_pop = 64 #population size
num_genes = 48

gene_space = [{"low": 0.0, "high": 10.49487305}] * 48


parent_selection_type = "tournament"
K_tournament = 3

keep_parents = 2
keep_elitism = 1

crossover_type = "single_point"
crossover_probability = None

mutation_type = "random"
mutation_probability = None
mutation_percent_genes = 10

#initial_population = None

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
    K_tournament=K_tournament,
    
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


solution, solution_fitness, solution_idx = ga.best_solution(ga.last_generation_fitness)
top_half = solution.reshape(6,8)
bottom_half = np.flipud(top_half)
voltages = np.vstack((top_half, bottom_half))
print(f"Voltages from the last solution : {voltages}")
print(f"Fitness value of the last best solution = {solution_fitness}")
print(f"Index of the last best solution : {solution_idx}")

print(f"Best overall voltages: {best_voltages}")
print(f"Best overall fitness: {best_fitnesses[-1]}")

np.savez(exp_folder / "results.npz",
         best_voltages=np.array(best_voltages),
         best_patterns=np.array(best_patterns),
         best_fitnesses=np.array(best_fitnesses)
         )

zero_volts = np.zeros(SIZE)
update_lb_array_file(zero_volts)
rpi.update_dacs()
time.sleep(LC_DELAY_TIME)
nsi.disconnect()
rpi.stop_program()
rpi.close()