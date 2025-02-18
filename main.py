# Example usage
from task import calculate_asap_cores, allocate_resources_to_nodes, generate_task, erdos_renyi_graph, \
    generate_accesses_and_lengths, visualize_task, get_critical_path, federated_scheduling, \
    hyperperiod, schedule_tasks, calculate_total_processors, generate_periodic_tasks, \
    get_all_task_instances

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
    print(f"utilization {task['utilization']}")
    visualize_task(task)

    critical_path = task["critical_path"]
    critical_path_length = task["critical_path_length"]
    print("Critical Path:", critical_path)
    print("Critical Path Length:", critical_path_length)

    asap_schedule = task["ASAP Schedule"]
    max_cores = task["Max Parallel Tasks"]
    print(f"Max Parallel Tasks: {max_cores}")
    print("\n")

    '''print("\nASAP Schedule and Core Requirements:")
    for node, start_time in asap_schedule.items():
        print(f"Node {node}: Start Time {start_time}, Execution Time {task['execution_times'].get(node, 'N/A')}") '''

    allocations, execution_times = allocate_resources_to_nodes(task, task["task_id"], accesses, lengths)

'''    print("\nAllocations and Execution Times:")
    for node, allocation in allocations.items():
        print(f"Node {node} (Execution Time: {execution_times[node]}): {allocation}")
    print("\n") '''

processors = federated_scheduling(tasks)
print("\n=== Core Assign Result ===")
print(f"Total Processors Used: {len(processors)}")
for processor in processors:
    print(f"Processor {processor.id}: Assigned Tasks {processor.assigned_tasks}, Utilization: {processor.utilization:.2f}")




hyperperiod = hyperperiod(tasks)
print("\n")
print(f"Hyperperiod: {hyperperiod}")

periodic_tasks = generate_periodic_tasks(tasks)
all_instances = get_all_task_instances(periodic_tasks)

'''for instance in all_instances:
    print(f"Instance ID: {instance['instance_id']}, Task ID: {instance['task_id']}")
    print(f"  Release Time: {instance['release_time']}")
    print(f"  Absolute Deadline: {instance['absolute_deadline']}")
    print(f"  Nodes: {instance['nodes']}")
    print(f"  Edges: {instance['edges']}")
    print(f"  Period: {instance['period']}")
    print(f"  Critical Path: {instance['critical_path']}")
    print(f"  Allocations: {instance['allocations']}")
   # print(f"  Processors: {instance['processors']}\n")
    print() '''

''' time_taken = schedule_tasks(tasks, max_cores)
#scheduling_log, task_execution_log = schedule_tasks(tasks)
#print_task_execution_log(task_execution_log)
print(f"Total time taken for scheduling: {time_taken}")  '''

