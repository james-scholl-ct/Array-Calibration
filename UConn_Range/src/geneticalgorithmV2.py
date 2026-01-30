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

LC_DELAY_TIME =  40 #in secs

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
    #fitness = E_main
    
    #lm = .25
    #loss = -E_main - lm * E_side
    
    return fitness

def sol_key(solution):
    return tuple(np.round(np.asarray(solution, dtype=float), 4))

pattern_cache = {}
fitness_cache = {}
old_volt = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0.],
    [10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305],
    [10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305],
    [10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305],
    [2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102],
    [0., 0., 0., 0., 0., 0., 0., 0.],
    [0.38452148, 0.38452148, 0.38452148, 0.38452148, 0.38452148, 0.38452148, 0.38452148, 0.38452148],
    [10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305, 10.49487305],
    [2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102, 2.40454102],
    [1.75341797, 1.75341797, 1.75341797, 1.75341797, 1.75341797, 1.75341797, 1.75341797, 1.75341797],
    [0.57421875, 0.57421875, 0.57421875, 0.57421875, 0.57421875, 0.57421875, 0.57421875, 0.57421875],
])

def make_fitness(vna_instance, rpi):
    
    def fitness_func(ga_instance, solution, solution_idx):
        global pattern_cache, fitness_cache
        solution = param_map(solution)
        #voltages = np.tile(solution[:,None], (1,8))
        voltages = old_volt
        voltages[:,2] = solution
        voltages[:, 5] = solution
        update_lb_array_file(voltages)
        #sends low and high band array files to PI and runs remote command to update DACs
        rpi.update_dacs()
        time.sleep(LC_DELAY_TIME)
        pattern = vna_instance.run_scan_get_hor_amp(SCAN_FILENAME, BEAM)
        
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
    pop = ga_instance.population
    fit = ga_instance.last_generation_fitness
    diversity_stats(pop, fit)
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    solution = param_map(solution)
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", solution_fitness)
    print(f"Change = {solution_fitness - last_fitness}")
    k = sol_key(solution)
    pattern = pattern_cache.get(k, None)
    fitness = fitness_cache.get(k, None)
    if solution_fitness > best_fitness:
        best_voltages = np.tile(solution[:,None], (1,8))
        best_fitness = solution_fitness
        best_patterns.append(pattern)
        best_fitnesses.append(fitness)
        pattern_cache = {}
        fitness_cache = {}
        plt.figure()
        for i, p in enumerate(best_patterns):
            plt.plot(p, label=f"Fitness: {np.round(best_fitnesses[i], 4)}")
        plt.xlabel("Span -10 to +10in")
        plt.ylabel("Mag (dB)")
        plt.title("Mag vs Span best")
        plt.grid(True)
        plt.legend()
        plt.savefig(exp_folder/ "MagVsSpanBest.png", dpi=200)
        plt.show()
    last_fitness = solution_fitness


def param_map(p):
    p = np.array(p)
    buff = .1
    for i, param in enumerate(p):
        if (param >=  10.49487305) and (param<=10.49487305+buff):
            p[i] = 10.49487305
        elif (param <= 0) and (param >= -buff):
            p[i] = 0
            
    voltages = np.mod(p, 10.49487305)
    
    return voltages

def blx_alpha_crossover(parents, offspring_size, ga_instance, alpha=0.2):
    n_off, n_genes = offspring_size
    offspring = np.empty((n_off, n_genes), dtype=float)

    for k in range(n_off):
        p1 = parents[k % parents.shape[0], :].astype(float)
        p2 = parents[(k + 1) % parents.shape[0], :].astype(float)

        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        d  = hi - lo

        child = np.random.uniform(lo - alpha*d, hi + alpha*d)

        offspring[k, :] = child

    return offspring

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

num_generations = 20
num_parents_mating = 12 #number of parent params chosen for breeding

fitness_func = make_fitness(nsi, rpi)

sol_per_pop = 30 #population size
num_genes = 12 

#gene_space = [{"low": 0.0, "high": 10.49487305}] * 12
gene_space = None
init_range_low = 0
init_range_high = 10.49487305

parent_selection_type = "tournament"
K_tournament = 3

keep_parents = 2
keep_elitism = 1

#crossover_type = "single_point"
crossover_type = blx_alpha_crossover
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
    init_range_high=init_range_high,
    init_range_low=init_range_low,

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

ga.plot_fitness()
solution, solution_fitness, solution_idx = ga.best_solution(ga.last_generation_fitness)
voltages = np.tile(solution[:,None], (1,8))
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