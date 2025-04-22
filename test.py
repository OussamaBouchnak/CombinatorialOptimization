def parse_instances(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    instances = []
    i = 1  # skip the first line (e.g., 2100)
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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def lpt_schedule_instance(instance):
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, class_id, instance_id = instance["header"]
    nb_machines = 2

    jobs = sorted(
        [(job_durations[i], mold_assignments[i] - 1, i) for i in range(nb_jobs)],
        reverse=True
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

def spt_schedule_instance(instance):  # Changed function name
    job_durations = instance["jobs"]
    mold_assignments = instance["machines"]
    nb_jobs, nb_molds, class_id, instance_id = instance["header"]
    nb_machines = 2

    # Changed sorting direction (reverse=False for SPT)
    jobs = sorted(
        [(job_durations[i], mold_assignments[i] - 1, i) for i in range(nb_jobs)],
        reverse=False  # This is the key SPT change
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

def plot_gantt_chart(schedule,typee):
    all_tasks = [task for machine in schedule for task in machine]
    if not all_tasks:
        print("No jobs to plot.")
        return
    
    max_time = max(task[1] + task[2] for task in all_tasks)
    job_ids = {task[0] for task in all_tasks}
    cmap = plt.get_cmap('tab20', len(job_ids))

    fig, ax = plt.subplots(figsize=(14, 6))  # Increased figure size
    
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
    ax.set_title(f"{typee} Gantt Chart with Mold Constraints", pad=20)
    
    plt.grid(True, axis='x', alpha=0.5)
    plt.tight_layout()
    plt.show()





instances = parse_instances("instances A.txt")
schedule, makespan = lpt_schedule_instance(instances[0])
plot_gantt_chart(schedule,"LPT" )
print("LPT Makespan:", makespan)

schedule, makespan = spt_schedule_instance(instances[0])  # Changed function name
plot_gantt_chart(schedule,"SPT")
print("SPT Makespan:", makespan)