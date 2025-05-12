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

def spt_optimized(instance, max_iterations=1000):
    job_durations = instance["jobs"]
    nb_jobs = len(job_durations)
    # Changed to sort by shortest processing time first (ascending order)
    job_order = sorted(range(nb_jobs), key=lambda x: job_durations[x])
    best_schedule, best_makespan = create_schedule_from_job_order(instance, job_order)
    
    for _ in range(max_iterations):
        new_order = job_order.copy()
        i, j = random.sample(range(nb_jobs), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_schedule, new_makespan = create_schedule_from_job_order(instance, new_order)
        if new_makespan < best_makespan:
            job_order = new_order
            best_makespan = new_makespan
            best_schedule = new_schedule
    
    return best_schedule, best_makespan

def main():
    instances_file = r".\instances A.txt"
    bounds_file = r".\borneunf.txt"
    output_dir = r".\plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    instances = parse_instances(instances_file)
    file_bounds = parse_lower_bounds(bounds_file)
    
    print(f"Processing {len(instances)} instances using SPT with optimization...")
    
    results = []
    for idx, instance in enumerate(instances):
        _, _, class_id, instance_id = instance['header']
        file_lb = file_bounds.get(idx)
        
        print(f"\nProcessing instance {idx+1}/{len(instances)}: Class {class_id}, ID {instance_id}")
        
        schedule, makespan = spt_optimized(instance)
        
        is_valid = verify_mold_constraints(schedule)
        assert is_valid, f"Invalid solution for instance {class_id}-{instance_id}!"
        
        gap = (makespan - file_lb) / file_lb * 100 if file_lb is not None else 0
        optimal = makespan == file_lb if file_lb is not None else False
        
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
    all_gaps = [r['gap'] for r in results if r['gap'] is not None]
    
    print("\n=== STATISTICS FOR ALL INSTANCES ===")
    print(f"Total instances: {len(results)}")
    print(f"Optimal solutions: {optimal_count} ({optimal_percentage:.2f}%)")
    if all_gaps:
        print(f"Gap statistics: Avg={sum(all_gaps)/len(all_gaps):.2f}%, Min={min(all_gaps):.2f}%, Max={max(all_gaps):.2f}%")
    else:
        print("No gap statistics available.")
    
    class_stats = defaultdict(lambda: {'count': 0, 'optimal': 0, 'gaps': []})
    for r in results:
        if r['gap'] is not None:
            stats = class_stats[r['class_id']]
            stats['count'] += 1
            stats['optimal'] += 1 if r['optimal'] else 0
            stats['gaps'].append(r['gap'])
    
    print("\n=== STATISTICS BY CLASS ===")
    print(f"{'Class ID':<10} | {'Count':<8} | {'Optimal %':<12} | {'Avg Gap %':<12}")
    print("-" * 50)
    
    for class_id, stats in sorted(class_stats.items()):
        if stats['count'] == 0:
            continue
        optimal_pct = (stats['optimal'] / stats['count']) * 100
        avg_gap = sum(stats['gaps']) / len(stats['gaps']) if stats['gaps'] else 0
        print(f"{class_id:<10} | {stats['count']:<8} | {optimal_pct:<12.2f} | {avg_gap:<12.2f}")

if __name__ == "__main__":
    main()