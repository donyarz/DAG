# Example usage
from task import calculate_asap_cores, allocate_resources_to_nodes, generate_task, erdos_renyi_graph, \
    generate_accesses_and_lengths, visualize_task, get_critical_path, federated_scheduling, \
    hyperperiod, schedule_tasks, print_task_execution_log, calculate_total_processors, generate_periodic_tasks

num_tasks = 2
accesses, lengths = generate_accesses_and_lengths(num_tasks)

tasks = []

for i in range(num_tasks):
    task = generate_task(task_id=i + 1, accesses=accesses, lengths=lengths)
    tasks.append(task)

# تخصیص منابع به نودها
for task in tasks:
    print(f"Task {task['task_id']}:")
    print(f"Nodes: {task['nodes']}")
    print(f"Edges: {task['edges']}")
    print(f"Accesses: {task['accesses']}")
    print(f"Lengths: {task['lengths']}")
   # print(f"Visualizing Task {task['task_id']}")
    print(f"period {task['period']}")
    visualize_task(task)

    critical_path = task["critical_path"]
    critical_path_length = task["critical_path_length"]
    print("Critical Path:", critical_path)
    print("Critical Path Length:", critical_path_length)

    asap_schedule = task["ASAP Schedule"]
    max_cores = task["Max Parallel Tasks"]

    print("\nASAP Schedule and Core Requirements:")
    for node, start_time in asap_schedule.items():
        print(f"Node {node}: Start Time {start_time}, Execution Time {task['execution_times'].get(node, 'N/A')}")

    print(f"Maximum Cores Required: {max_cores}")
    print("\n")

    allocations, execution_times = allocate_resources_to_nodes(task, task["task_id"], accesses, lengths)

    print("\nAllocations and Execution Times:")
    for node, allocation in allocations.items():
        print(f"Node {node} (Execution Time: {execution_times[node]}): {allocation}")
    print("\n")

scheduling_result = federated_scheduling(tasks)
num_cores = max(result["num_processors"] for result in scheduling_result)
core_total = calculate_total_processors(tasks)
for result in scheduling_result:
    print(f"Task {result['task_id']}:")
    print(f"  U_i: {result['U_i']:.2f}")
    print(f"  Number of processors ASAP: {result['num_processors']}")
    print(f"  Number of processors mi:", core_total)
    print("\n")

hyperperiod = hyperperiod(tasks)

# چاپ هایپرپریود و تعداد نمونه‌های هر تسک
print(f"Hyperperiod: {hyperperiod}")
periodic_tasks = generate_periodic_tasks(tasks)

for task in periodic_tasks:
    print(f"\nTask {task['task_id']}:")  # از کلید 'task_id' استفاده کنید
    for instance in task["instances"]:
        print(f"  Instance -> Release Time: {instance['release_time']}, Absolute Deadline: {instance['absolute_deadline']}")

#scheduling_log, task_execution_log = schedule_tasks(tasks)
#print_task_execution_log(task_execution_log)


