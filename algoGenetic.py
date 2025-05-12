import numpy as np
import os
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_instances(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    instances = []
    i = 0
    num_instances = int(lines[0].strip())
    i += 1
    
    while i < len(lines):
        if lines[i].strip() == '':
            i += 1
            continue
        
        instances.append({
            'header': list(map(int, lines[i].split())),
            'jobs': list(map(int, lines[i+1].split())),
            'machines': list(map(int, lines[i+2].split()))
        })
        i += 3
    
    return instances

def parse_lower_bounds(filename):
    bounds = {}
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts or line.startswith('#'):
                continue
            if len(parts) >= 5:
                try:
                    n, m, class_id, instance_id = map(int, parts[:4])
                    lower_bound = int(parts[4])
                    print(f"Parsed: n={n}, m={m}, class={class_id}, instance={instance_id}, lb={lower_bound}")
                    bounds[idx] = lower_bound
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line.strip()}, Error: {e}")
    
    print(f"Parsed {len(bounds)} lower bounds from file")
    return bounds
def verify_mold_constraints(schedule):
    mold_usage = defaultdict(list)
    for machine_schedule in schedule:
        for job_id, start_time, duration, mold_id in machine_schedule:
            end_time = start_time + duration
            for prev_start, prev_end in mold_usage[mold_id]:
                if (start_time < prev_end and end_time > prev_start):
                    return False
            
            mold_usage[mold_id].append((start_time, end_time))
    
    return True

def create_schedule_from_job_order(instance, job_order):
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, _, _ = instance["header"]
    nb_machines = 2
    jobs = [(job_durations[i], mold_assignments[i] - 1, i) for i in job_order]
    machine_times = [0] * nb_machines
    machine_schedules = [[] for _ in range(nb_machines)]
    mold_availability = [0] * nb_molds
    
    for duration, mold, job_id in jobs:
        earliest_start_times = []
        for m in range(nb_machines):
            start_time = max(machine_times[m], mold_availability[mold])
            earliest_start_times.append((start_time, m))
        
        earliest_start_times.sort()  # Sort by start time
        start_time, best_machine = earliest_start_times[0]
        machine_schedules[best_machine].append((job_id, start_time, duration, mold))
        machine_times[best_machine] = start_time + duration
        mold_availability[mold] = start_time + duration  # Update mold availability
    
    makespan = max(machine_times)
    return machine_schedules, makespan

def initialize_population(instance, pop_size=50):
    """Create an initial population of random job orders"""
    nb_jobs = instance["header"][0]
    population = []
    
    for _ in range(pop_size):
        job_order = list(range(nb_jobs))
        random.shuffle(job_order)
        schedule, makespan = create_schedule_from_job_order(instance, job_order)
        population.append((job_order, schedule, makespan))
    
    return population

def crossover(parent1, parent2):
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    # Copy segment from parent1
    for i in range(start, end + 1):
        child[i] = parent1[i]
    
    # Fill remaining positions with values from parent2 (in order)
    j = (end + 1) % n
    for i in range(n):
        idx = (end + 1 + i) % n
        if child[idx] == -1:  # If position is empty
            # Find next value from parent2 that's not in child
            while parent2[j] in child:
                j = (j + 1) % n
            child[idx] = parent2[j]
            j = (j + 1) % n
    
    return child

def mutate(job_order, mutation_rate=0.05):
    """Mutate a job order by swapping positions"""
    mutated = job_order.copy()
    
    for _ in range(int(len(job_order) * mutation_rate) + 1):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(job_order)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated

def tournament_selection(population, tournament_size=3):
    """Select an individual using tournament selection"""
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x[2])  # Select the one with lowest makespan

def genetic_algorithm(instance, pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
    
    # Initialize population
    population = initialize_population(instance, pop_size)
    
    # Sort initial population by fitness (makespan - lower is better)
    population.sort(key=lambda x: x[2])
    
    best_solution = population[0]
    stagnation_counter = 0
    
    # Main loop
    for generation in range(generations):
        new_population = []
        
        # Elitism - keep the best solution
        new_population.append(best_solution)
        
        # Create new population
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            if random.random() < crossover_rate:
                child_job_order = crossover(parent1[0], parent2[0])
            else:
                child_job_order = parent1[0].copy() if parent1[2] < parent2[2] else parent2[0].copy()
            
            # Mutation
            child_job_order = mutate(child_job_order, mutation_rate)
            
            # Evaluate child
            child_schedule, child_makespan = create_schedule_from_job_order(instance, child_job_order)
            
            # Add child to new population (valid by construction)
            new_population.append((child_job_order, child_schedule, child_makespan))
        
        # Replace population
        population = new_population
        
        # Sort population by fitness
        population.sort(key=lambda x: x[2])
        
        # Update best solution
        if population[0][2] < best_solution[2]:
            best_solution = population[0]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Early stopping if no improvement for many generations
        if stagnation_counter >= 20:
            break
    
    return best_solution[1], best_solution[2]

def main():    # Define file paths
    instances_file = r".\instances A.txt"
    bounds_file = r".\borneunf.txt"
    output_dir = r".\plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load instances and lower bounds
    instances = parse_instances(instances_file)
    file_bounds = parse_lower_bounds(bounds_file)
    
    print(f"Processing {len(instances)} instances using Genetic Algorithm...")
    
    results = []
    for idx, instance in enumerate(instances):
        _, _, class_id, instance_id = instance['header']
        file_lb = file_bounds.get(idx)
        
        print(f"\nProcessing instance {idx+1}/{len(instances)}: Class {class_id}, ID {instance_id}")
        
        # Apply genetic algorithm with our calculated lower bound
        schedule, makespan = genetic_algorithm(instance)
        
        # Verify the solution
        is_valid = verify_mold_constraints(schedule)
        assert is_valid, f"Invalid solution for instance {class_id}-{instance_id}!"
        
        # Calculate gap and optimality
        gap = (makespan - file_lb) / file_lb * 100
        optimal = makespan == file_lb
        
        results.append({
            'class_id': class_id,
            'instance_id': instance_id,
            'makespan': makespan,
            'file_lb': file_lb,
            'gap': gap,
            'optimal': optimal
        })
        
        print(f"   LB: {file_lb}, Gap: {gap:.2f}%, Optimal: {optimal}")
    
    # Calculate statistics
    if not results:
        print("No instances processed!")
        return
        
    # Calculate overall statistics
    optimal_count = sum(1 for r in results if r['optimal'])
    optimal_percentage = (optimal_count / len(results)) * 100
    all_gaps = [r['gap'] for r in results]
    
    print("\n=== STATISTICS FOR ALL INSTANCES ===")
    print(f"Total instances: {len(results)}")
    print(f"Optimal solutions: {optimal_count} ({optimal_percentage:.2f}%)")
    print(f"Gap statistics: Avg={sum(all_gaps)/len(all_gaps):.2f}%, Min={min(all_gaps):.2f}%, Max={max(all_gaps):.2f}%")
    
   
    
    
    # Group statistics by class
    class_stats = defaultdict(lambda: {'count': 0, 'optimal': 0, 'gaps': []})
    for r in results:
        stats = class_stats[r['class_id']]
        stats['count'] += 1
        if r['optimal']:
            stats['optimal'] += 1
        stats['gaps'].append(r['gap'])
    
    # Print class-specific statistics
    print("\n=== STATISTICS BY CLASS ===")
    print(f"{'Class ID':<10} | {'Count':<8} | {'Optimal %':<12} | {'Avg Gap %':<12}")
    print("-" * 50)
    
    for class_id, stats in sorted(class_stats.items()):
        optimal_pct = (stats['optimal'] / stats['count']) * 100
        avg_gap = sum(stats['gaps']) / len(stats['gaps'])
        print(f"{class_id:<10} | {stats['count']:<8} | {optimal_pct:<12.2f} | {avg_gap:<12.2f}")

if __name__ == "__main__":
    main()