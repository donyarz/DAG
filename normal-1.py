from task import federated_scheduling, calculate_asap_cores, calculate_total_processors, generate_periodic_tasks, \
    get_all_task_instances, edf_scheduling

tasks = [
    {
        "task_id": 1,  # اضافه کردن task_id
        "total_execution_time": 12,
        "period": 10,
        "nodes": [1, 2, 3, 4],
        "edges": [(1, 2), (1, 3), (2, 4), (3, 4)],
        "execution_times": {1: 4, 2: 3, 3: 3, 4: 2},
        "critical_path": [],
        "critical_path_length": [],
        "allocations": []
    },
    {
        "task_id": 2,  # اضافه کردن task_id
        "total_execution_time": 5,
        "period": 8,
        "nodes": [1, 2],
        "edges": [(1, 2)],
        "execution_times": {1: 2, 2: 3},
        "critical_path": [],
        "critical_path_length": [],
        "allocations": []
    },
    {
        "task_id": 3,  # اضافه کردن task_id
        "total_execution_time": 7,
        "period": 20,
        "nodes": [1, 2, 3],
        "edges": [(1, 2), (2, 3)],
        "execution_times": {1: 3, 2: 2, 3: 2},
        "critical_path": [],
        "critical_path_length": [],
        "allocations": []
    }
]

processors = federated_scheduling(tasks)

periodic_tasks = generate_periodic_tasks(tasks)

# گام 3: دریافت تمام نمونه‌های تولیدشده
all_instances = get_all_task_instances(periodic_tasks)

# گام 4: نگاشت نمونه‌های تسک‌ها به همان پردازنده‌های اصلی
for processor in processors:
    assigned_tasks = set(processor.assigned_tasks)
    processor.instances = [instance for instance in all_instances if instance["task_id"] in assigned_tasks]

# گام 5: اعمال الگوریتم EDF برای اولویت‌بندی اجرا روی هر پردازنده
edf_scheduling(processors, periodic_tasks)

# نمایش نتیجه نهایی زمان‌بندی
for processor in processors:
    print(f"Processor {processor.id}:")
    for instance in processor.instances:
        print(f"  Task {instance['task_id']} - Instance {instance['instance_id']} - Release Time: {instance['release_time']}, Deadline: {instance['absolute_deadline']}")