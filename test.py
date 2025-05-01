
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import math
import time
import os

def parse_instances(filename):
    """Parse job scheduling instances from file"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    instances = []
    i = 0  # Start from the first line
    
    # Get the number of instances from the first line
    num_instances = int(lines[0].strip())
    i += 1
    
    while i < len(lines):
        if lines[i].strip() == '':
            i += 1
            continue
        
        header = list(map(int, lines[i].split()))
        job_data = list(map(int, lines[i+1].split()))
        machine_data = list(map(int, lines[i+2].split()))
        
        instances.append({
            'header': header,
            'jobs': job_data,
            'machines': machine_data
        })
        i += 3
    
    print(f"Successfully parsed {len(instances)} instances out of expected {num_instances}")
    return instances
def parse_lower_bounds(filename):
    """
    Parse the lower bounds file to extract instance identifiers and lower bounds.
    Each line format: n m class_id instance_id lower_bound algo time
    
    Returns:
        Dictionary mapping (n, m, class_id, instance_id) to lower_bound
    """
    bounds = {}
    line_count = 0
    error_count = 0
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Print first few lines for debugging
            print("\nSample lines from bounds file:")
            for i, line in enumerate(lines[:3]):
                print(f"  Line {i+1}: {line.strip()}")
            
            for line_count, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split()
                
                # Some lines might have fewer parts if they're missing the algo and time fields
                # But we need at least the first 5 parts for our data
                if len(parts) < 5:
                    print(f"Warning: Line {line_count} has fewer than 5 parts: {line}")
                    error_count += 1
                    continue
                
                try:
                    # The format appears to be: n m class_id instance_id lower_bound algo time
                    # Handle potential integer parsing issues
                    try:
                        n = int(parts[0])
                        m = int(parts[1])
                    except ValueError:
                        print(f"Warning: Could not parse n or m as integers at line {line_count}")
                        error_count += 1
                        continue
                    
                    try:
                        class_id = int(parts[2])
                        instance_id = int(parts[3])
                    except ValueError:
                        print(f"Warning: Could not parse class_id or instance_id as integers at line {line_count}")
                        error_count += 1
                        continue
                    
                    try:
                        # Check if the lower bound is actually an integer - could be "2200" instead of 2200
                        lower_bound_raw = parts[4]
                        lower_bound = int(lower_bound_raw)
                        
                        # Check for specific issues mentioned (2200 lower bound problem)
                        if lower_bound == 2200 and len(parts) > 5:
                            print(f"Note: Found '2200' lower bound at line {line_count}, verifying...")
                            # Check if this might actually be part of another value or field
                    except ValueError:
                        print(f"Warning: Could not parse lower_bound as integer at line {line_count}: '{parts[4]}'")
                        error_count += 1
                        continue
                    
                    # Use all four identifying parts as the key
                    key = (n, m, class_id, instance_id)
                    bounds[key] = lower_bound
                    
                    # Print a few examples for verification
                    if len(bounds) <= 3 or lower_bound == 2200:
                        print(f"Parsed: n={n}, m={m}, Class {class_id}, Instance {instance_id}, LB: {lower_bound}")
                        
                except Exception as e:
                    print(f"Error parsing line {line_count}: {line}")
                    print(f"Error details: {e}")
                    error_count += 1
    except Exception as e:
        print(f"Error opening or reading bounds file: {e}")
    
    print(f"Successfully parsed {len(bounds)} lower bounds from {line_count} lines")
    print(f"Encountered {error_count} errors during parsing")
    
    # Additional debugging for the specific "2200" problem
    values_count = {}
    for lb in bounds.values():
        values_count[lb] = values_count.get(lb, 0) + 1
    
    print("\nMost common lower bound values:")
    for lb, count in sorted(values_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Value {lb}: {count} occurrences")
    
    # Check for duplicate entries with different lower bounds
    key_conflicts = {}
    for (n, m, class_id, instance_id), lb in bounds.items():
        # Check if we see the same problem multiple times with different values
        # First try with just class_id and instance_id
        simple_key = (class_id, instance_id)
        if simple_key not in key_conflicts:
            key_conflicts[simple_key] = {}
        key_conflicts[simple_key][(n, m)] = lb
    
    # Report on any conflicts
    print("\nChecking for conflicts (same class_id/instance_id with different n/m values):")
    conflict_count = 0
    for simple_key, variants in key_conflicts.items():
        if len(variants) > 1:
            conflict_count += 1
            class_id, instance_id = simple_key
            print(f"  Conflict for Class {class_id}, Instance {instance_id}:")
            for (n, m), lb in variants.items():
                print(f"    n={n}, m={m}: LB={lb}")
    
    print(f"Found {conflict_count} instances with conflicting n/m values")
    
    return bounds
def evaluate_schedule(schedule):
    """Calculate the makespan of a schedule"""
    if not schedule or not any(schedule):
        return 0
    return max(max(task[1] + task[2] for task in machine) if machine else 0 for machine in schedule)

def lpt_schedule_instance(instance):
    """Longest Processing Time First scheduling algorithm"""
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, _, _ = instance["header"]
    nb_machines = 2

    # Sort jobs by processing time (descending)
    jobs = sorted(
        [(job_durations[i], mold_assignments[i] - 1, i) for i in range(nb_jobs)],
        reverse=True  # Longest processing time first
    )

    machine_times = [0] * nb_machines
    machine_schedules = [[] for _ in range(nb_machines)]
    mold_availability = [0] * nb_molds

    for duration, mold, job_id in jobs:
        best_machine = None
        earliest_start = float('inf')

        for m in range(nb_machines):
            start_time = max(machine_times[m], mold_availability[mold])
            if start_time < earliest_start:
                earliest_start = start_time
                best_machine = m

        machine_schedules[best_machine].append((job_id, earliest_start, duration, mold))
        machine_times[best_machine] = earliest_start + duration
        mold_availability[mold] = earliest_start + duration

    return machine_schedules, evaluate_schedule(machine_schedules)

def spt_schedule_instance(instance):
    """Shortest Processing Time First scheduling algorithm"""
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, _, _ = instance["header"]
    nb_machines = 2

    # Sort jobs by processing time (ascending)
    jobs = sorted(
        [(job_durations[i], mold_assignments[i] - 1, i) for i in range(nb_jobs)],
        reverse=False  # Shortest processing time first
    )

    machine_times = [0] * nb_machines
    machine_schedules = [[] for _ in range(nb_machines)]
    mold_availability = [0] * nb_molds

    for duration, mold, job_id in jobs:
        best_machine = None
        earliest_start = float('inf')

        for m in range(nb_machines):
            start_time = max(machine_times[m], mold_availability[mold])
            if start_time < earliest_start:
                earliest_start = start_time
                best_machine = m

        machine_schedules[best_machine].append((job_id, earliest_start, duration, mold))
        machine_times[best_machine] = earliest_start + duration
        mold_availability[mold] = earliest_start + duration

    return machine_schedules, evaluate_schedule(machine_schedules)

def create_initial_schedule(instance):
    """Create an initial schedule based on job order"""
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, _, _ = instance["header"]
    nb_machines = 2

    jobs = [(job_durations[i], mold_assignments[i] - 1, i) for i in range(nb_jobs)]
    
    machine_times = [0] * nb_machines
    machine_schedules = [[] for _ in range(nb_machines)]
    mold_availability = [0] * nb_molds

    for duration, mold, job_id in jobs:
        best_machine = None
        earliest_start = float('inf')

        for m in range(nb_machines):
            start_time = max(machine_times[m], mold_availability[mold])
            if start_time < earliest_start:
                earliest_start = start_time
                best_machine = m

        machine_schedules[best_machine].append((job_id, earliest_start, duration, mold))
        machine_times[best_machine] = earliest_start + duration
        mold_availability[mold] = earliest_start + duration

    return machine_schedules

def rebuild_schedule(jobs_order, instance):
    """Rebuild schedule based on job order, respecting mold constraints"""
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs = len(job_durations)
    nb_molds = max(mold_assignments)
    nb_machines = 2

    # Create a list of jobs based on the given order
    jobs = [(job_durations[job_id], mold_assignments[job_id] - 1, job_id) for job_id in jobs_order]
    
    machine_times = [0] * nb_machines
    machine_schedules = [[] for _ in range(nb_machines)]
    mold_availability = [0] * nb_molds

    for duration, mold, job_id in jobs:
        best_machine = None
        earliest_start = float('inf')

        for m in range(nb_machines):
            start_time = max(machine_times[m], mold_availability[mold])
            if start_time < earliest_start:
                earliest_start = start_time
                best_machine = m

        machine_schedules[best_machine].append((job_id, earliest_start, duration, mold))
        machine_times[best_machine] = earliest_start + duration
        mold_availability[mold] = earliest_start + duration

    return machine_schedules

def calculate_makespan(job_order, instance):
    """Calculate makespan for a given job order"""
    schedule = rebuild_schedule(job_order, instance)
    return evaluate_schedule(schedule)
import random

def tabu_search(instance, max_iterations=100, tabu_size=10, neighborhood_size=100):
    """Optimized Tabu Search for the parallel machine scheduling problem with mold constraints."""

    job_durations = instance["jobs"]
    nb_jobs = len(job_durations)

    # Initial solution from SPT heuristic
    spt_solution, _ = spt_schedule_instance(instance)
    job_order = []

    # Reconstruct job order from initial SPT solution
    machine_jobs = [sorted([(start, job_id) for job_id, start, _, _ in machine]) for machine in spt_solution]
    for machine in machine_jobs:
        for _, job_id in machine:
            job_order.append(job_id)

    best_order = job_order.copy()
    best_makespan = calculate_makespan(best_order, instance)

    tabu_set = set()
    tabu_queue = []

    for _ in range(max_iterations):
        best_neighbor_order = None
        best_neighbor_makespan = float('inf')
        best_move = None

        # Randomly sample neighborhood pairs to limit evaluation
        neighbors = random.sample(
            [(i, j) for i in range(nb_jobs) for j in range(i + 1, nb_jobs)],
            min(neighborhood_size, nb_jobs * (nb_jobs - 1) // 2)
        )

        for i, j in neighbors:
            job1, job2 = best_order[i], best_order[j]

            if (job1, job2) in tabu_set or (job2, job1) in tabu_set:
                continue

            # Swap and evaluate neighbor
            neighbor_order = best_order.copy()
            neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
            neighbor_makespan = calculate_makespan(neighbor_order, instance)

            if neighbor_makespan < best_neighbor_makespan:
                best_neighbor_makespan = neighbor_makespan
                best_neighbor_order = neighbor_order
                best_move = (job1, job2)

        # Update best solution if improvement is found
        if best_neighbor_order and best_neighbor_makespan <= best_makespan:
            best_order = best_neighbor_order
            best_makespan = best_neighbor_makespan

        # Tabu list maintenance
        if best_move:
            tabu_set.add(best_move)
            tabu_queue.append(best_move)
            if len(tabu_queue) > tabu_size:
                old_move = tabu_queue.pop(0)
                tabu_set.remove(old_move)

    final_schedule = rebuild_schedule(best_order, instance)
    return final_schedule, best_makespan


def simulated_annealing(instance, max_temperature=1000, cooling_rate=0.95, max_iterations=1000):
    """Simulated Annealing implementation for scheduling problem"""
    job_durations = instance["jobs"]
    nb_jobs = len(job_durations)
    
    # Start with a random solution
    current_order = list(range(nb_jobs))
    random.shuffle(current_order)
    
    current_schedule = rebuild_schedule(current_order, instance)
    current_makespan = evaluate_schedule(current_schedule)
    
    best_order = current_order.copy()
    best_schedule = current_schedule
    best_makespan = current_makespan
    
    temperature = max_temperature
    
    # SA iterations
    for iteration in range(max_iterations):
        if temperature < 0.1:
            break
            
        # Generate a neighbor by swapping two random jobs
        i, j = random.sample(range(nb_jobs), 2)
        neighbor_order = current_order.copy()
        neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
        
        # Evaluate the neighbor
        neighbor_makespan = calculate_makespan(neighbor_order, instance)
        
        # Decide whether to accept the neighbor
        delta = neighbor_makespan - current_makespan
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_order = neighbor_order
            current_makespan = neighbor_makespan
            
            if current_makespan < best_makespan:
                best_order = current_order.copy()
                best_makespan = current_makespan
                best_schedule = rebuild_schedule(best_order, instance)
        
        # Cool down
        temperature *= cooling_rate
    
    return best_schedule, best_makespan

def genetic_algorithm(instance, population_size=50, num_generations=100, mutation_rate=0.1):
    """Genetic Algorithm implementation for scheduling problem"""
    job_durations = instance["jobs"]
    nb_jobs = len(job_durations)
    
    # Helper function to create a random individual
    def create_individual():
        individual = list(range(nb_jobs))
        random.shuffle(individual)
        return individual
    
    # Helper function for tournament selection
    def selection(population, fitness_values, tournament_size=3):
        selected = []
        for _ in range(2):  # Select two parents
            tournament = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament]
            # Select the best from tournament (minimization problem)
            selected.append(population[tournament[tournament_fitness.index(min(tournament_fitness))]])
        return selected
    
    # Helper function for order crossover
    def crossover(parent1, parent2):
        size = len(parent1)
        # Select crossover points
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring
        offspring = [-1] * size
        # Copy segment from parent1
        for i in range(start, end + 1):
            offspring[i] = parent1[i]
            
        # Fill remaining positions with values from parent2 in order
        j = 0
        for i in range(size):
            if offspring[i] == -1:  # If position is empty
                # Find next value from parent2 that's not already in offspring
                while parent2[j] in offspring:
                    j += 1
                offspring[i] = parent2[j]
                j += 1
                
        return offspring
    
    # Helper function for swap mutation
    def mutation(individual, rate):
        if random.random() < rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    # Main GA loop
    best_individual = None
    best_fitness = float('inf')
    
    for generation in range(num_generations):
        # Evaluate fitness for each individual
        fitness_values = [calculate_makespan(ind, instance) for ind in population]
        
        # Track best solution
        min_fitness = min(fitness_values)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_individual = population[fitness_values.index(min_fitness)]
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individual
        elite_idx = fitness_values.index(min(fitness_values))
        new_population.append(population[elite_idx])
        
        # Create rest of the new population
        while len(new_population) < population_size:
            # Selection
            parents = selection(population, fitness_values)
            
            # Crossover
            if random.random() < 0.7:  # crossover probability
                offspring = crossover(parents[0], parents[1])
            else:
                offspring = parents[0].copy()
                
            # Mutation
            offspring = mutation(offspring, mutation_rate)
            
            # Add to new population
            new_population.append(offspring)
            
        # Replace old population
        population = new_population
    
    # Return best solution found
    best_schedule = rebuild_schedule(best_individual, instance)
    return best_schedule, best_fitness

def plot_gantt_chart(schedule, title, instance_id=None, makespan=None):
    """
    Plot a Gantt chart where colors are based on molds rather than jobs.
    Each mold gets a single distinct color with no shade variations.
    
    Args:
        schedule: List of machine schedules, each containing task tuples (job_id, start, duration, mold)
        title: Title of the chart
        instance_id: Optional instance identifier to display in the subtitle
        makespan: Optional makespan value to display in the subtitle
        
    Returns:
        matplotlib figure object
    """
    all_tasks = [task for machine in schedule for task in machine]
    if not all_tasks:
        print("No jobs to plot.")
        return None
    
    max_time = max(task[1] + task[2] for task in all_tasks)
    
    # Get unique molds used in the schedule
    all_molds = sorted({task[3] for task in all_tasks})
    n_molds = len(all_molds)
    
    # We no longer need to track job IDs for coloring
    job_ids = sorted({task[0] for task in all_tasks})
    
    # Create a colormap for the molds - use tab20 for up to 20 distinct colors
    # For more than 20 molds, we'll use hsv which can generate any number of colors
    if n_molds <= 20:
        cmap_name = 'tab20'
    else:
        cmap_name = 'hsv'
        
    cmap = plt.get_cmap(cmap_name, max(20, n_molds))
    
    # Create a mapping from mold ID to base color
    mold_base_colors = {mold: cmap(i % cmap.N) for i, mold in enumerate(all_molds)}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Custom label positioning logic
    def can_fit_label(_, duration, text):  # Ignore unused variable start
        required_width = len(text) * 0.007 * max_time  # Empirical scaling
        return duration > required_width

    for machine_id, tasks in enumerate(schedule):
        for task in tasks:
            job_id, start, duration, mold = task
            
            # Use the same color for all tasks with the same mold
            color = mold_base_colors[mold]
            
            # Draw the main block
            rect = mpatches.Rectangle(
                (start, machine_id),
                duration,
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            # Always add mold number at the bottom
            ax.text(
                start + duration/2,
                machine_id + 0.2,
                f"M{mold+1}",
                ha='center',
                va='center',
                fontsize=7,
                color='white',
                weight='bold'
            )
            
            # Add job number at the top if space permits
            label = f"J{job_id}"
            if can_fit_label(start, duration, label):
                ax.text(
                    start + duration/2,
                    machine_id + 0.6,
                    label,
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black'
                )

    # Axis configuration
    ax.set_yticks([i + 0.4 for i in range(len(schedule))])
    ax.set_yticklabels([f"Machine {i+1}" for i in range(len(schedule))])
    ax.set_xlim(0, max_time * 1.05)
    ax.set_ylim(-0.2, len(schedule) + 0.2)
    ax.set_xlabel("Time")
    
    # Create subtitle with instance_id and makespan if provided
    subtitle = ""
    if instance_id is not None:
        subtitle += f"Instance {instance_id}"
    if makespan is not None:
        if subtitle:
            subtitle += f" - "
        subtitle += f"Makespan: {makespan}"
        
    # Combine title and subtitle
    title_full = f"{title} Gantt Chart with Mold Constraints"
    if subtitle:
        title_full += f"\n{subtitle}"
    
    ax.set_title(title_full, pad=20)
    
    # Add a legend for molds - if there are too many molds, only show a subset
    from matplotlib.patches import Patch
    max_legend_items = 10  # Maximum number of items to show in legend
    
    if n_molds <= max_legend_items:
        # Show all molds in the legend
        legend_elements = [Patch(facecolor=mold_base_colors[mold], 
                                edgecolor='black',
                                label=f'Mold {mold+1}') 
                          for mold in all_molds]
    else:
        # Show only a subset of molds in the legend
        step = max(1, n_molds // max_legend_items)
        sample_molds = all_molds[::step]
        legend_elements = [Patch(facecolor=mold_base_colors[mold], 
                                edgecolor='black',
                                label=f'Mold {mold+1}') 
                          for mold in sample_molds]
        # Add an ellipsis to indicate there are more molds
        legend_elements.append(Patch(facecolor='white', edgecolor='black', label='...'))
        
    ax.legend(handles=legend_elements, loc='upper right', ncol=min(5, (n_molds+4)//5))
    
    plt.grid(True, axis='x', alpha=0.5)
    plt.tight_layout()
    
    return fig

def calculate_metrics(results, lower_bounds):
    """
    Calculate comprehensive performance metrics including:
    - Number of optimal solutions found
    - Average relative gap to lower bounds
    - Number of instances where algorithm is better
    - Maximum gap to lower bound
    - Standard deviation of gaps
    - Normalized ratio (solution/lower_bound)
    - Runtime performance
    """
    # Group by class_id
    class_groups = {}
    
    for record in results:
        class_id = record['class_id']
        instance_id = record['instance_id']
        makespan = record['makespan']
        runtime = record.get('runtime', 0)
        
        if class_id not in class_groups:
            class_groups[class_id] = []
            
        # Calculate relative gap if lower bound exists
        # Try different possible key formats
        # First, see if we can find a matching key with any n, m values
        lower_bound = None
        for key in lower_bounds:
            if len(key) == 4 and key[2] == class_id and key[3] == instance_id:
                lower_bound = lower_bounds[key]
                break
        
        if lower_bound is not None:
            gap = (makespan - lower_bound) / lower_bound if lower_bound > 0 else float('inf')
            relative_gap = gap * 100  # as percentage
            is_optimal = makespan == lower_bound
            norm_ratio = makespan / lower_bound if lower_bound > 0 else float('inf')
        else:
            relative_gap = None
            is_optimal = False
            norm_ratio = None
            
        class_groups[class_id].append({
            'makespan': makespan,
            'relative_gap': relative_gap,
            'is_optimal': is_optimal,
            'norm_ratio': norm_ratio,
            'runtime': runtime
        })
    
    # Calculate metrics
    metrics = {}
    for class_id, records in class_groups.items():
        makespans = [r['makespan'] for r in records]
        gaps = [r['relative_gap'] for r in records if r['relative_gap'] is not None]
        ratios = [r['norm_ratio'] for r in records if r['norm_ratio'] is not None]
        runtimes = [r['runtime'] for r in records]
        optimal_count = sum(1 for r in records if r['is_optimal'])
        
        metrics[class_id] = {
            'avg_makespan': np.mean(makespans),
            'optimal_solutions': optimal_count,
            'optimal_percentage': (optimal_count / len(records)) * 100 if records else 0,
            'avg_gap_percent': np.mean(gaps) if gaps else None,
            'std_gap_percent': np.std(gaps) if gaps and len(gaps) > 1 else None,
            'max_gap_percent': max(gaps) if gaps else None,
            'min_gap_percent': min(gaps) if gaps else None,
            'avg_norm_ratio': np.mean(ratios) if ratios else None,
            'avg_runtime': np.mean(runtimes),
            'num_instances': len(records)
        }
        
    return metrics


import numpy as np
import matplotlib.pyplot as plt

def plot_comparison_chart(metrics_dict, metric_name='avg_gap_percent', title=None):
    """
    Create a bar chart comparing a specific metric across algorithms
    for each class.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get all unique class_ids across all algorithms
    class_ids = sorted(set(class_id for alg_metrics in metrics_dict.values() 
                         for class_id in alg_metrics.keys()))
    
    # Get all algorithm names
    algorithms = list(metrics_dict.keys())
    
    # Extract metric values for each algorithm and class
    algorithm_values = {}
    for alg in algorithms:
        algorithm_values[alg] = []
        for c in class_ids:
            # Get the metric value, defaulting to 0 if the class or metric doesn't exist
            class_metrics = metrics_dict[alg].get(c, {})
            value = class_metrics.get(metric_name)
            
            # Handle None values before rounding (this fixes the TypeError)
            if value is None:
                print(f"Warning: {metric_name} for algorithm {alg}, class {c} is None")
                algorithm_values[alg].append(0)
            else:
                # Only round floating point values
                try:
                    if isinstance(value, (int, float)):
                        algorithm_values[alg].append(round(value, 6))
                    else:
                        algorithm_values[alg].append(value)
                except Exception as e:
                    print(f"Warning: Error processing value for {alg}, class {c}: {e}")
                    algorithm_values[alg].append(0)
    
    x = np.arange(len(class_ids))
    width = 0.8 / len(algorithms)  # Adjust bar width based on number of algorithms
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each algorithm
    rects_list = []
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms)/2 + 0.5) * width
        rects = ax.bar(x + offset, algorithm_values[alg], width, label=alg)
        rects_list.append(rects)
    
    # Set labels and title
    y_label = {
        'avg_gap_percent': 'Average Gap to Lower Bound (%)',
        'avg_norm_ratio': 'Average Normalized Ratio',
        'avg_runtime': 'Average Runtime (s)',
        'optimal_percentage': 'Optimal Solutions (%)'
    }.get(metric_name, metric_name)
    
    ax.set_ylabel(y_label)
    ax.set_xlabel('Class ID')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Algorithm Performance Comparison: {y_label}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(class_ids)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects, values):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            if isinstance(val, float):
                label = f'{val:.1f}'
                if metric_name == 'avg_gap_percent':
                    label += '%'
            else:
                label = f'{val}'
                
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    # Add labels to each bar
    for i, alg in enumerate(algorithms):
        autolabel(rects_list[i], algorithm_values[alg])
    
    fig.tight_layout()
    return fig

def run_all_algorithms(instances, lower_bounds, plot_instance_index=None, output_dir=None):
    """
    Run all algorithms on all instances and compare results
    
    Args:
        instances: List of instance dictionaries
        lower_bounds: Dictionary mapping (class_id, instance_id) to lower bound values
        plot_instance_index: Optional index of instance to plot
        output_dir: Directory to save output images
    """
    # Prepare result containers
    algorithm_results = {
        'LPT': [],
        'SPT': [],
        'Tabu Search': [],
        'Simulated Annealing': [],
        'Genetic Algorithm': []
    }
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each instance
    for i, instance in enumerate(instances):
        _, _, class_id, instance_id = instance['header']  
        print(f"Processing instance {i+1}/{len(instances)} (Class {class_id}, ID {instance_id})")
        
        results_for_instance = {}
        
        # Run LPT
        start_time = time.time()
        lpt_schedule, lpt_makespan = lpt_schedule_instance(instance)
        lpt_time = time.time() - start_time
        algorithm_results['LPT'].append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': lpt_makespan,
            'runtime': lpt_time
        })
        results_for_instance['LPT'] = (lpt_schedule, lpt_makespan, lpt_time)
        
        # Run SPT
        start_time = time.time()
        spt_schedule, spt_makespan = spt_schedule_instance(instance)
        spt_time = time.time() - start_time
        algorithm_results['SPT'].append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': spt_makespan,
            'runtime': spt_time
        })
        results_for_instance['SPT'] = (spt_schedule, spt_makespan, spt_time)
        
        # Run Tabu Search
        start_time = time.time()
        ts_schedule, ts_makespan = tabu_search(instance)
        ts_time = time.time() - start_time
        algorithm_results['Tabu Search'].append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': ts_makespan,
            'runtime': ts_time
        })
        results_for_instance['Tabu Search'] = (ts_schedule, ts_makespan, ts_time)
        
        # Run Simulated Annealing
        start_time = time.time()
        sa_schedule, sa_makespan = simulated_annealing(instance)
        sa_time = time.time() - start_time
        algorithm_results['Simulated Annealing'].append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': sa_makespan,
            'runtime': sa_time
        })
        results_for_instance['Simulated Annealing'] = (sa_schedule, sa_makespan, sa_time)
        
        # Run Genetic Algorithm - reduce parameters for speed
        start_time = time.time()
        ga_schedule, ga_makespan = genetic_algorithm(instance, population_size=30, num_generations=50)
        ga_time = time.time() - start_time
        algorithm_results['Genetic Algorithm'].append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': ga_makespan,
            'runtime': ga_time
        })
        results_for_instance['Genetic Algorithm'] = (ga_schedule, ga_makespan, ga_time)
        
        # If this is the instance to plot or save charts, do so
        if plot_instance_index is not None and i == plot_instance_index:
            print(f"\nDetailed results for instance {instance_id} (Class {class_id}):")
            
            # Get lower bound if available
            key = (class_id, instance_id)
            lb = lower_bounds.get(key, None)
            
            if lb:
                print(f"Lower bound: {lb}")
            
            # Display results for each algorithm
            for alg_name, (schedule, makespan, runtime) in results_for_instance.items():
                print(f"{alg_name}: Makespan = {makespan}, Runtime = {runtime:.4f}s")
                
                if lb:
                    gap = (makespan - lb) / lb * 100 if lb > 0 else float('inf')
                    print(f"Gap to lower bound: {gap:.2f}%")
                
                fig = plot_gantt_chart(schedule, alg_name, instance_id, makespan)
                
                if output_dir:
                    fig.savefig(os.path.join(output_dir, f"{alg_name}_{class_id}_{instance_id}.png"))
                else:
                    plt.show()
                plt.close(fig)
    
    # Calculate metrics for each algorithm
    algorithm_metrics = {}
    for alg_name, results in algorithm_results.items():
        algorithm_metrics[alg_name] = calculate_metrics(results, lower_bounds)
    
    # Print summary table
    print("\nPerformance Summary:")
    print("-" * 100)
    print(f"{'Class ID':<10} | {'Algorithm':<20} | {'Avg Gap %':<12} | {'Std Dev %':<12} | {'Opt Sol %':<10} | {'Avg Runtime':<12}")
    print("-" * 100)
    
    class_ids = sorted(set(class_id for alg_metrics in algorithm_metrics.values() 
                         for class_id in alg_metrics.keys()))
    
    for class_id in class_ids:
        for alg_name, metrics in algorithm_metrics.items():
            class_metrics = metrics.get(class_id, {})
            avg_gap = class_metrics.get('avg_gap_percent')
            std_gap = class_metrics.get('std_gap_percent')
            opt_pct = class_metrics.get('optimal_percentage')
            avg_runtime = class_metrics.get('avg_runtime')
            
            gap_str = f"{avg_gap:.2f}%" if avg_gap is not None else "N/A"
            std_str = f"{std_gap:.2f}%" if std_gap is not None else "N/A"
            opt_str = f"{opt_pct:.1f}%" if opt_pct is not None else "N/A"
            runtime_str = f"{avg_runtime:.4f}s" if avg_runtime is not None else "N/A"
            
            print(f"{class_id:<10} | {alg_name:<20} | {gap_str:<12} | {std_str:<12} | {opt_str:<10} | {runtime_str:<12}")
    
    # Plot comparative charts
    metrics_to_plot = ['avg_gap_percent', 'avg_norm_ratio', 'avg_runtime', 'optimal_percentage']
    plot_titles = {
        'avg_gap_percent': 'Average Gap to Lower Bound (%)',
        'avg_norm_ratio': 'Average Solution / Lower Bound Ratio',
        'avg_runtime': 'Average Runtime (seconds)',
        'optimal_percentage': 'Percentage of Optimal Solutions Found (%)'
    }
    
    for metric in metrics_to_plot:
        fig = plot_comparison_chart(algorithm_metrics, metric, plot_titles[metric])
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"comparison_{metric}.png"))
        else:
            plt.show()
        plt.close(fig)
    
    # Find best algorithm for each class
    print("\nBest Algorithm by Class:")
    print("-" * 60)
    print(f"{'Class ID':<10} | {'Best Algorithm':<20} | {'Avg Gap %':<12}")
    print("-" * 60)
    
    for class_id in class_ids:
        best_alg = None
        best_gap = float('inf')
        
        for alg_name, metrics in algorithm_metrics.items():
            if class_id in metrics and metrics[class_id].get('avg_gap_percent') is not None:
                gap = metrics[class_id]['avg_gap_percent']
                if gap < best_gap:
                    best_gap = gap
                    best_alg = alg_name
        
        if best_alg:
            gap_str = f"{best_gap:.2f}%" if best_gap != float('inf') else "N/A"
            print(f"{class_id:<10} | {best_alg:<20} | {gap_str:<12}")
    
    return algorithm_results, algorithm_metrics

def main():
    """Main function to run the scheduling algorithms"""
    # Define file paths
    instances_file = r"c:\Users\oussa\Desktop\oc\2machinesNmolds\instances A.txt"
    bounds_file = r"c:\Users\oussa\Desktop\oc\2machinesNmolds\borneunf.txt"

    output_dir = "gantt_charts"
    
    try:
        # Load instances and lower bounds
        instances = parse_instances(instances_file)
        print(f"Loaded {len(instances)} instances")
        
        lower_bounds = parse_lower_bounds(bounds_file)
        print(f"Loaded {len(lower_bounds)} lower bounds")
        
        # Run all algorithms on all instances
        # Optionally plot the first instance (change index as needed, or set to None to skip plotting)
        algorithm_results, algorithm_metrics = run_all_algorithms(
            instances, lower_bounds, plot_instance_index=0, output_dir=output_dir
        )
        
        print("\nAnalysis complete! Charts saved to:", output_dir)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure the input files are in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()