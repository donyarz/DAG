from task import federated_scheduling, calculate_asap_cores, calculate_total_processors

tasks = [
    {
        "task_id": 1,  # اضافه کردن task_id
        "total_execution_time": 12,
        "period": 10,
        "nodes": [1, 2, 3, 4],
        "edges": [(1, 2), (1, 3), (2, 4), (3, 4)],
        "execution_times": {1: 4, 2: 3, 3: 3, 4: 2}
    },
    {
        "task_id": 2,  # اضافه کردن task_id
        "total_execution_time": 5,
        "period": 8,
        "nodes": [1, 2],
        "edges": [(1, 2)],
        "execution_times": {1: 2, 2: 3}
    },
    {
        "task_id": 3,  # اضافه کردن task_id
        "total_execution_time": 7,
        "period": 20,
        "nodes": [1, 2, 3],
        "edges": [(1, 2), (2, 3)],
        "execution_times": {1: 3, 2: 2, 3: 2}
    }
]

processors = federated_scheduling(tasks)
