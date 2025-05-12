import numpy as np
import os
import time
import random
from collections import defaultdict, deque
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
                    bounds[idx] = lower_bound
                except (ValueError, IndexError):
                    continue
    
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
        
        earliest_start_times.sort()
        start_time, best_machine = earliest_start_times[0]
        machine_schedules[best_machine].append((job_id, start_time, duration, mold))
        machine_times[best_machine] = start_time + duration
        mold_availability[mold] = start_time + duration
    
    makespan = max(machine_times)
    return machine_schedules, makespan

def tabu_search(instance, max_iterations=100, tabu_tenure=10, neighborhood_size=50):
    nb_jobs = instance["header"][0]
    
    current_order = list(range(nb_jobs))
    random.shuffle(current_order)
    current_schedule, current_makespan = create_schedule_from_job_order(instance, current_order)
    
    best_order = current_order.copy()
    best_makespan = current_makespan
    tabu_set = set()
    fifo_queue = deque()
    stagnation_counter = 0
    
    for _ in range(max_iterations):
        candidates = []
        for _ in range(neighborhood_size):
            i, j = random.sample(range(nb_jobs), 2)
            move = tuple(sorted((i, j)))
            candidate_order = current_order.copy()
            candidate_order[i], candidate_order[j] = candidate_order[j], candidate_order[i]
            candidates.append((candidate_order, move))
        
        best_candidate = None
        best_candidate_makespan = float('inf')
        best_move = None
        
        for candidate_order, move in candidates:
            if move in tabu_set:
                candidate_schedule, candidate_makespan = create_schedule_from_job_order(instance, candidate_order)
                if candidate_makespan < best_makespan and candidate_makespan < best_candidate_makespan:
                    best_candidate = candidate_order
                    best_candidate_makespan = candidate_makespan
                    best_move = move
            else:
                candidate_schedule, candidate_makespan = create_schedule_from_job_order(instance, candidate_order)
                if candidate_makespan < best_candidate_makespan:
                    best_candidate = candidate_order
                    best_candidate_makespan = candidate_makespan
                    best_move = move
        
        if best_candidate is None:
            current_order = list(range(nb_jobs))
            random.shuffle(current_order)
            current_schedule, current_makespan = create_schedule_from_job_order(instance, current_order)
            tabu_set.clear()
            fifo_queue.clear()
            stagnation_counter = 0
            continue
        
        current_order = best_candidate
        current_makespan = best_candidate_makespan
        
        if current_makespan < best_makespan:
            best_order = current_order.copy()
            best_makespan = current_makespan
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if best_move is not None:
            fifo_queue.append(best_move)
            tabu_set.add(best_move)
            if len(fifo_queue) > tabu_tenure:
                expired_move = fifo_queue.popleft()
                tabu_set.remove(expired_move)
        
        if stagnation_counter >= 20:
            for _ in range(nb_jobs // 2):
                i, j = random.sample(range(nb_jobs), 2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
            current_schedule, current_makespan = create_schedule_from_job_order(instance, current_order)
            tabu_set.clear()
            fifo_queue.clear()
            stagnation_counter = 0
    
    best_schedule, best_makespan = create_schedule_from_job_order(instance, best_order)
    return best_schedule, best_makespan

def main():
    instances_file = r".\instances A.txt"
    bounds_file = r".\borneunf.txt"
    output_dir = r".\plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    instances = parse_instances(instances_file)
    file_bounds = parse_lower_bounds(bounds_file)
    
    print(f"Processing {len(instances)} instances using Tabu Search...")
    
    results = []
    for idx, instance in enumerate(instances):
        _, _, class_id, instance_id = instance['header']
        file_lb = file_bounds.get(idx)
        
        print(f"\nProcessing instance {idx+1}/{len(instances)}: Class {class_id}, ID {instance_id}")
        
        schedule, makespan = tabu_search(instance)
        
        is_valid = verify_mold_constraints(schedule)
        assert is_valid, f"Invalid solution for instance {class_id}-{instance_id}!"
        
        gap = (makespan - file_lb) / file_lb * 100 if file_lb else 0
        optimal = makespan == file_lb if file_lb else False
        
        results.append({
            'class_id': class_id,
            'instance_id': instance_id,
            'makespan': makespan,
            'file_lb': file_lb,
            'gap': gap,
            'optimal': optimal
        })
        
        print(f"   LB: {file_lb}, Makespan: {makespan}, Gap: {gap:.2f}%, Optimal: {optimal}")
    
    if not results:
        print("No instances processed!")
        return
    
    optimal_count = sum(1 for r in results if r['optimal'])
    optimal_percentage = (optimal_count / len(results)) * 100
    all_gaps = [r['gap'] for r in results if r['file_lb'] is not None]
    
    print("\n=== STATISTICS FOR ALL INSTANCES ===")
    print(f"Total instances: {len(results)}")
    print(f"Optimal solutions: {optimal_count} ({optimal_percentage:.2f}%)")
    if all_gaps:
        print(f"Gap statistics: Avg={sum(all_gaps)/len(all_gaps):.2f}%, Min={min(all_gaps):.2f}%, Max={max(all_gaps):.2f}%")
    else:
        print("No gap data available.")
    
    class_stats = defaultdict(lambda: {'count': 0, 'optimal': 0, 'gaps': []})
    for r in results:
        stats = class_stats[r['class_id']]
        stats['count'] += 1
        if r['optimal']:
            stats['optimal'] += 1
        if r['file_lb'] is not None:
            stats['gaps'].append(r['gap'])
    
    print("\n=== STATISTICS BY CLASS ===")
    print(f"{'Class ID':<10} | {'Count':<8} | {'Optimal %':<12} | {'Avg Gap %':<12}")
    print("-" * 50)
    
    for class_id, stats in sorted(class_stats.items()):
        optimal_pct = (stats['optimal'] / stats['count']) * 100 if stats['count'] > 0 else 0
        avg_gap = sum(stats['gaps']) / len(stats['gaps']) if stats['gaps'] else 0
        print(f"{class_id:<10} | {stats['count']:<8} | {optimal_pct:<12.2f} | {avg_gap:<12.2f}")

if __name__ == "__main__":
    main()