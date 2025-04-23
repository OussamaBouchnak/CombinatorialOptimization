def parse_instances(filename):
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
    
    return instances

def parse_lower_bounds(filename):
    """
    Parse the lower bounds file to extract instance identifiers and lower bounds.
    Format: n m class_id instance_id lower_bound algo time
    """
    bounds = {}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure we have at least 5 values
                    n, m, class_id, instance_id = map(int, parts[:4])
                    lower_bound = int(parts[4])
                    # Create a key using class_id and instance_id
                    key = (class_id, instance_id)
                    bounds[key] = lower_bound
    except Exception as e:
        print(f"Error parsing lower bounds file: {e}")
        
    return bounds

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def lpt_schedule_instance(instance):
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, class_id, instance_id = instance["header"]
    nb_machines = 2

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

    return machine_schedules, max(machine_times)

def spt_schedule_instance(instance):
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, class_id, instance_id = instance["header"]
    nb_machines = 2

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

    return machine_schedules, max(machine_times)

def plot_gantt_chart(schedule, title):
    all_tasks = [task for machine in schedule for task in machine]
    if not all_tasks:
        print("No jobs to plot.")
        return
    
    max_time = max(task[1] + task[2] for task in all_tasks)
    job_ids = {task[0] for task in all_tasks}
    cmap = plt.get_cmap('tab20', len(job_ids))

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Custom label positioning logic
    def can_fit_label(start, duration, text):
        text_width = duration * 0.8  # Allow 80% of block width for text
        required_width = len(text) * 0.007 * max_time  # Empirical scaling
        return duration > required_width

    for machine_id, tasks in enumerate(schedule):
        for task in tasks:
            job_id, start, duration, mold = task
            color = cmap(job_id % len(job_ids))
            
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
    ax.set_title(f"{title} Gantt Chart with Mold Constraints", pad=20)
    
    plt.grid(True, axis='x', alpha=0.5)
    plt.tight_layout()
    plt.show()

def calculate_metrics(results, lower_bounds):
    """
    Calculate performance metrics including:
    - Average makespan per class
    - Average relative gap to lower bounds
    - Standard deviation
    
    Args:
        results: List of dictionaries containing class_id, instance_id and makespan
        lower_bounds: Dictionary mapping (class_id, instance_id) to lower bound values
        
    Returns:
        Dictionary with metrics by class_id
    """
    # Group by class_id
    class_groups = {}
    
    for record in results:
        class_id = record['class_id']
        instance_id = record['instance_id']
        makespan = record['makespan']
        
        if class_id not in class_groups:
            class_groups[class_id] = []
            
        # Calculate relative gap if lower bound exists
        key = (class_id, instance_id)
        if key in lower_bounds:
            lower_bound = lower_bounds[key]
            gap = (makespan - lower_bound) / lower_bound if lower_bound > 0 else float('inf')
            relative_gap = gap * 100  # as percentage
        else:
            relative_gap = None
            
        class_groups[class_id].append({
            'makespan': makespan,
            'relative_gap': relative_gap
        })
    
    # Calculate metrics
    metrics = {}
    for class_id, records in class_groups.items():
        makespans = [r['makespan'] for r in records]
        gaps = [r['relative_gap'] for r in records if r['relative_gap'] is not None]
        
        metrics[class_id] = {
            'avg_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans),
            'min_makespan': min(makespans),
            'max_makespan': max(makespans),
            'avg_gap_percent': np.mean(gaps) if gaps else None,
            'std_gap_percent': np.std(gaps) if gaps else None,
            'min_gap_percent': min(gaps) if gaps else None,
            'max_gap_percent': max(gaps) if gaps else None,
            'num_instances': len(records)
        }
        
    return metrics

def plot_comparison_chart(lpt_metrics, spt_metrics):
    """
    Create a bar chart comparing the average gap percentages of LPT and SPT
    for each class.
    """
    class_ids = sorted(set(lpt_metrics.keys()).union(spt_metrics.keys()))
    
    lpt_gaps = [lpt_metrics.get(c, {}).get('avg_gap_percent', 0) for c in class_ids]
    spt_gaps = [spt_metrics.get(c, {}).get('avg_gap_percent', 0) for c in class_ids]
    
    # Replace None values with 0 for plotting
    lpt_gaps = [0 if g is None else g for g in lpt_gaps]
    spt_gaps = [0 if g is None else g for g in spt_gaps]
    
    x = np.arange(len(class_ids))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, lpt_gaps, width, label='LPT')
    rects2 = ax.bar(x + width/2, spt_gaps, width, label='SPT')
    
    ax.set_ylabel('Average Gap to Lower Bound (%)')
    ax.set_xlabel('Class ID')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_ids)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.show()

def run_comparison_with_bounds(instances, lower_bounds, plot_instance_index=None):
    """
    Run both LPT and SPT on all instances and compare results using lower bounds
    
    Args:
        instances: List of instance dictionaries
        lower_bounds: Dictionary mapping (class_id, instance_id) to lower bound values
        plot_instance_index: Optional index of instance to plot
    """
    lpt_results = []
    spt_results = []
    
    for i, instance in enumerate(instances):
        nb_jobs, nb_molds, class_id, instance_id = instance['header']
        
        # Run LPT
        lpt_schedule, lpt_makespan = lpt_schedule_instance(instance)
        lpt_results.append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': lpt_makespan
        })
        
        # Run SPT
        spt_schedule, spt_makespan = spt_schedule_instance(instance)
        spt_results.append({
            'instance_id': instance_id,
            'class_id': class_id,
            'makespan': spt_makespan
        })
        
        # If this is the instance to plot, do so
        if plot_instance_index is not None and i == plot_instance_index:
            plot_gantt_chart(lpt_schedule, "LPT")
            print(f"LPT Makespan for instance {instance_id}: {lpt_makespan}")
            
            plot_gantt_chart(spt_schedule, "SPT")
            print(f"SPT Makespan for instance {instance_id}: {spt_makespan}")
            
            # If we have a lower bound for this instance, report the gap
            key = (class_id, instance_id)
            if key in lower_bounds:
                lb = lower_bounds[key]
                lpt_gap = (lpt_makespan - lb) / lb * 100 if lb > 0 else float('inf')
                spt_gap = (spt_makespan - lb) / lb * 100 if lb > 0 else float('inf')
                print(f"Lower bound: {lb}")
                print(f"LPT gap: {lpt_gap:.2f}%")
                print(f"SPT gap: {spt_gap:.2f}%")
    
    # Calculate comprehensive metrics
    lpt_metrics = calculate_metrics(lpt_results, lower_bounds)
    spt_metrics = calculate_metrics(spt_results, lower_bounds)
    
    # Print summary
    print("\nPerformance Metrics by Class:")
    print("Class ID | LPT Avg Gap | SPT Avg Gap | Better Algorithm")
    print("---------|-------------|-------------|----------------")
    
    for class_id in sorted(set(lpt_metrics.keys()).union(spt_metrics.keys())):
        lpt_gap = lpt_metrics.get(class_id, {}).get('avg_gap_percent', float('inf'))
        spt_gap = spt_metrics.get(class_id, {}).get('avg_gap_percent', float('inf'))
        
        # Handle None values
        lpt_gap = lpt_gap if lpt_gap is not None else float('inf')
        spt_gap = spt_gap if spt_gap is not None else float('inf')
        
        better = "LPT" if lpt_gap < spt_gap else "SPT" if spt_gap < lpt_gap else "Tie"
        
        lpt_gap_str = f"{lpt_gap:.2f}%" if lpt_gap != float('inf') else "N/A"
        spt_gap_str = f"{spt_gap:.2f}%" if spt_gap != float('inf') else "N/A"
        
        print(f"{class_id:9d} | {lpt_gap_str:11s} | {spt_gap_str:11s} | {better}")
    
    # Create a visualization of the comparison
    plot_comparison_chart(lpt_metrics, spt_metrics)
    
    return lpt_results, spt_results, lpt_metrics, spt_metrics

# Load instances and lower bounds, then run comparison
try:
    instances = parse_instances("2machinesNmolds\instances A.txt")
    print(f"Loaded {len(instances)} instances")
    
    lower_bounds = parse_lower_bounds("borneunf.txt")
    print(f"Loaded {len(lower_bounds)} lower bounds")
    
    # Run comparison on all instances and plot the first one (index 0)
    lpt_results, spt_results, lpt_metrics, spt_metrics = run_comparison_with_bounds(
        instances, lower_bounds, plot_instance_index=0
    )
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")