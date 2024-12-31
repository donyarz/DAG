import random


def generate_accesses_and_lengths(num_tasks: int, num_resources: int = 6) -> tuple[dict, dict]:
    accesses = {f"R{q + 1}": [0] * num_tasks for q in range(num_resources)}
    lengths = {f"R{q + 1}": [[] for _ in range(num_tasks)] for q in range(num_resources)}

    for q in range(num_resources):
        max_accesses = random.randint(1, 16)  # حداکثر تعداد دسترسی‌ها
        max_length = random.randint(5, 100)  # حداکثر طول دسترسی‌ها

        for i in range(num_tasks):
            if max_accesses > 0:
                accesses[f"R{q + 1}"][i] = random.randint(0, max_accesses)
                max_accesses -= accesses[f"R{q + 1}"][i]

                if accesses[f"R{q + 1}"][i] > 0:
                    lengths[f"R{q + 1}"][i] = [random.randint(1, max_length)
                                               for _ in range(accesses[f"R{q + 1}"][i])]

    return accesses, lengths


import random


def generate_nodes_for_tasks(num_tasks: int, max_nodes: int = 5, max_execution_time: int = 30) -> dict:
    """
    تولید نودها برای هر تسک، شامل زمان اجرای هر نود.

    :param num_tasks: تعداد تسک‌ها
    :param max_nodes: حداکثر تعداد نودها در هر تسک
    :param max_execution_time: حداکثر زمان اجرای هر نود
    :return: دیکشنری شامل نودها و زمان اجرای آن‌ها برای هر تسک
    """
    tasks = {f"T{i + 1}": [{"node_id": f"N{j + 1}", "execution_time": random.randint(1, max_execution_time)}
                           for j in range(random.randint(1, max_nodes))]
             for i in range(num_tasks)}
    return tasks


def allocate_resources_to_nodes(tasks: dict, accesses: dict, lengths: dict) -> dict:
    """
    تخصیص منابع به نودهای هر تسک بر اساس زمان اجرای نودها و طول دسترسی‌ها.

    :param tasks: دیکشنری شامل نودها و زمان اجرای آن‌ها برای هر تسک
    :param accesses: دیکشنری شامل تعداد دسترسی‌ها به منابع برای هر تسک
    :param lengths: دیکشنری شامل طول دسترسی‌ها به منابع برای هر تسک
    :return: دیکشنری شامل منابع تخصیص داده شده به هر نود
    """
    allocations = {task: {node["node_id"]: [] for node in tasks[task]} for task in tasks}

    for task in tasks:
        for node in tasks[task]:
            execution_time = node["execution_time"]
            for resource, access_lengths in lengths[task].items():
                while access_lengths and execution_time > 0:
                    access_time = access_lengths[0]  # اولین طول دسترسی
                    if access_time >= execution_time:
                        allocations[task][node["node_id"]].append((resource, execution_time))
                        access_lengths[0] -= execution_time
                        execution_time = 0
                    else:
                        allocations[task][node["node_id"]].append((resource, access_time))
                        execution_time -= access_time
                        access_lengths.pop(0)  # حذف دسترسی مصرف‌شده

    return allocations


# استفاده از توابع
def main():
    num_tasks = 5  # تعداد وظایف

    # تولید دسترسی‌ها و طول دسترسی‌ها
    accesses , lengths = generate_accesses_and_lengths(num_tasks)


    print("Accesses:")
    for resource, tasks in accesses.items():
        print(f"{resource}: {tasks}")

    print("\nLengths:")
    for resource, tasks in lengths.items():
        print(f"{resource}: {tasks}")


if __name__ == "__main__":
    main()
