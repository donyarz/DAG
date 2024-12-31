from task import Resource, Node, Edge, Task, Job, generate_tasks, generate_resources, generate_task
import random as rand

# تولید منابع
resources = generate_resources(resource_count=6)  # تعداد منابع

# تولید یک تسک
task = generate_task(task_id=1, resources=resources)

#visualize_dag(task)
# چاپ اطلاعات تسک
print(f"Task ID: {task.id}")
print(f"Period: {task.period}")
print(f"WCET: {task.wcet}")
print(f"Deadline: {task.deadline}")
#print(f"U: {task.utilization}")
print("\nNodes and Resources:")

# نمایش گره‌ها و منابع مورد نیاز
for node in task.nodes:
    print(f"{node}")


