from typing import List, Tuple, Dict
import random as rand
import random
import math
import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce
from queue import Queue



@dataclass
class Resource:
    id: str

    def map_to_color(self) -> str:
        if self.id == "R1":
            return 'red'
        if self.id == "R2":
            return 'blue'
        if self.id == "R3":
            return 'green'
        if self.id == "R4":
            return 'yellow'
        if self.id == "R5":
            return 'purple'
        if self.id == "R6":
            return 'pink'
        return 'black'


@dataclass
class Node:
    id: str
    wcet: int
    critical_st: list[int]
    critical_en: list[int]
    resources: list[Resource] = None

    def needed_resource(self, time: int) -> int:
        for i in range(len(self.resources)):
            if self.critical_st[i] <= time and self.critical_en[i] > time:
                return self.resources[i]
        return None

    def needed_resource_ends_at(self, time: int) -> bool:
        for i in self.critical_en:
            if i == time:
                return True
        return False


    def __str__(self) -> str:
        return f"Node: {self.id}, WCET: {self.wcet}, Resource: {self.resources}"


@dataclass
class Edge:
    src: Node
    sink: Node


@dataclass
class Task:
    def __init__(self, id: int, period: int, wcet: int, nodes: list, edges: list, release_time: int, \
                 absolute_deadline: int, relative_deadline:int):
        self.id = id
        self.period = period
        self.wcet = wcet
        self.deadline = absolute_deadline
        self.relative_deadline = relative_deadline
        self.release_time = release_time
        self.nodes = nodes
        self.edges = edges
        #self.U = self.wcet / self.period  # Utilization
        self.instances = []


    def get_wcet(self) -> int:
        return sum([node.wcet for node in self.nodes])

    def utilization(self) -> float:
        return self.get_wcet() / self.period

    def do_need_resource(self, resource: Resource) -> bool:
        return any([res == resource for node in self.nodes for res in node.resources])

    def nearest_deadline(self, time: int) -> int:
        return self.period - (time % self.period)

    def __str__(self) -> str:
        return f"Task: {self.id}, Period: {self.period}, WCET: {self.wcet}"


@dataclass
class Job:
    id: int
    task: Task
    arrival: int
    deadline: int
    active: bool = False


def erdos_renyi_graph() -> tuple[list[int], list[tuple[int, int]]]:
    num_nodes = random.randint(5, 20)
    edge_probability = 0.1

    G = nx.erdos_renyi_graph(num_nodes, edge_probability, directed=True)
    G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    mapping = {node: node + 1 for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    # Add source and sink nodes
    source_node = "source"
    sink_node = "sink"
    G.add_node(source_node)
    G.add_node(sink_node)

    for node in list(G.nodes):
        if G.in_degree(node) == 0 and node != source_node:
            G.add_edge(source_node, node)
        if G.out_degree(node) == 0 and node != sink_node and node != source_node:
            G.add_edge(node, sink_node)
    nodes = list(G.nodes())
    edges = list(G.edges())

    return nodes, edges

def visualize_task(task):
    G = nx.DiGraph()
    G.add_nodes_from(task["nodes"])
    G.add_edges_from(task["edges"])

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", node_size=700, font_size=10,
            edge_color="gray")
    plt.title(f"Task {task['task_id']} DAG", fontsize=14)
    plt.show()

def get_critical_path(nodes: list, edges: list[tuple], execution_times: dict) -> tuple[list, int]:
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    source = "source"
    sink = "sink"
    all_paths = list(nx.all_simple_paths(G, source=source, target=sink))

    max_execution_time = 0
    critical_path = []

    for path in all_paths:
        path_execution_time = sum(
            execution_times.get(node, 0) for node in path if node not in ["source", "sink"]
        )

        if path_execution_time > max_execution_time:
            max_execution_time = path_execution_time
            critical_path = path

    return critical_path, max_execution_time

def __repr__(self):
        return f"Task(ID: {self.task_id}, C_i: {self.C_i}, Nodes: {self.num_nodes})"


def generate_resources(resource_count: int) -> list[Resource]:
    resources = [Resource(id=f"R{i + 1}") for i in range(resource_count)]
    return resources

def generate_accesses_and_lengths(num_tasks: int, num_resources: int = 6) -> tuple[dict, dict]:
    accesses = {f"R{q + 1}": [0] * num_tasks for q in range(num_resources)}
    lengths = {f"R{q + 1}": [[] for _ in range(num_tasks)] for q in range(num_resources)}

    for q in range(num_resources):
        max_accesses = random.randint(1, 16)
        max_length = random.randint(5, 100)

        for i in range(num_tasks):
            if max_accesses > 0:
                accesses[f"R{q + 1}"][i] = random.randint(0, max_accesses)
                max_accesses -= accesses[f"R{q + 1}"][i]

                if accesses[f"R{q + 1}"][i] > 0:
                    lengths[f"R{q + 1}"][i] = [random.randint(1, max_length)
                                               for _ in range(accesses[f"R{q + 1}"][i])]

    return accesses, lengths


def generate_task(task_id: int, accesses: dict, lengths: dict) -> dict:
    nodes, edges = erdos_renyi_graph()
    if len(nodes) <= 2:
        print(f"Skipping Task {task_id}: No nodes other than source and sink.")
        return None

    execution_times = {node: random.randint(13, 30) for node in nodes if node not in ["source", "sink"]}
    critical_path, critical_path_length = get_critical_path(nodes, edges, execution_times)
    total_execution_time = sum(execution_times.values())
    period = int(critical_path_length / rand.uniform(0.125, 0.25))
    deadline = period
    U_i = round(total_execution_time / period, 2)
    asap_schedule, max_parallel_tasks = calculate_asap_cores(nodes, edges, execution_times)

    allocations, execution_times = allocate_resources_to_nodes(
        {"nodes": nodes, "edges": edges, "execution_times": execution_times}, task_id, accesses, lengths
    )

    return {
        "task_id": task_id,
        "nodes": nodes,
        "edges": edges,
        "execution_times": execution_times,
        "total_execution_time": total_execution_time,
        "period": period,
        "deadline": deadline,
        "utilization": U_i,
        "accesses": accesses,
        "lengths": lengths,
        "allocations": allocations,
        "critical_path": critical_path,
        "critical_path_length": critical_path_length,
        "ASAP Schedule": asap_schedule,
        "Max Parallel Tasks": max_parallel_tasks,
        }
def generate_tasks(resources: list[str], task_count: int) -> list[dict]:
    tasks = []
    for i in range(task_count):
        tasks.append(generate_task(i + 1, resources))
    return tasks


def allocate_resources_to_nodes(task: dict, task_id: int, accesses: dict, lengths: dict) -> tuple[dict, dict]:
    nodes = [node for node in task["nodes"] if node != "source" and node != "sink"]

    allocations = {node: [] for node in nodes}
    execution_times = task["execution_times"]

    for node in nodes:
        execution_time = execution_times[node]
        critical_sections = []
        normal_sections = []


        for resource, task_accesses in accesses.items():
            if task_accesses[task_id - 1] > 0:
                node_access_lengths = lengths[resource][task_id - 1]

                while node_access_lengths and execution_time > 0:
                    access_time = node_access_lengths[0]
                    if execution_time >= access_time:
                        critical_sections.append((resource, access_time))
                        execution_time -= access_time
                        node_access_lengths.pop(0)
                    else:
                        break

        remaining_time = execution_time
        normal_sections = []

        if critical_sections:
            num_critical_sections = len(critical_sections)
            for _ in range(num_critical_sections):
                if remaining_time > 0:
                    normal_section_time = random.randint(0, remaining_time)
                    normal_sections.append(normal_section_time)
                    remaining_time -= normal_section_time
                else:
                    normal_sections.append(0)

            normal_sections.append(remaining_time)
        else:
            normal_sections.append(remaining_time)

        allocation = []
        for i, critical in enumerate(critical_sections):
            allocation.append(("Normal", normal_sections[i]))
            allocation.append(critical)
        if normal_sections:
            allocation.append(("Normal", normal_sections[-1]))

        allocations[node] = allocation

    return allocations, execution_times


#class algorithm:

@dataclass
class Processor:
    id: int
    assigned_tasks: List[int]
    utilization: float = 0.0
def calculate_total_processors(tasks):
    for task in tasks:
        total_execution_time = task["total_execution_time"]
        period = task["period"]
        U_i = total_execution_time / period
        U_sum = sum(U_i for task in tasks)
    U_norm = rand.uniform(0.1, 1)
    m_total = math.ceil(U_sum / U_norm)
    return m_total

def calculate_asap_cores(nodes: List[int], edges: List[Tuple[int, int]], execution_times: Dict[int, int]) -> Tuple[Dict[int, int], int]:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    asap_schedule = {}
    end_times = {}

    for node in nx.topological_sort(G):
        if node == "source":
            asap_schedule[node] = 0
            end_times[node] = 0
        else:
            start_time = max(
                [asap_schedule[pred] + execution_times.get(pred, 0) for pred in G.predecessors(node)],
                default=0
            )
            asap_schedule[node] = start_time
            end_times[node] = start_time + execution_times.get(node, 0)
    max_parallel_tasks = 0
    time_slots = {}
    for node, start_time in asap_schedule.items():
        if node not in ("source", "sink"):
            time_slots.setdefault(start_time, []).append(node)
            max_parallel_tasks = max(max_parallel_tasks, len(time_slots[start_time]))

    return asap_schedule, max_parallel_tasks


def federated_scheduling(tasks):
    total_processors = calculate_total_processors(tasks)
    print(f"Total Processors: {total_processors}")

    # ایجاد پردازنده‌ها
    processors = [Processor(id=i + 1, assigned_tasks=[]) for i in range(total_processors)]
    processors_state = processors.copy()

    scheduling_result = []
    remaining_processors = total_processors

    for task in tasks:
        total_execution_time = task["total_execution_time"]
        period = task["period"]
        U_i = total_execution_time / period
        _, max_parallel_tasks = calculate_asap_cores(task["nodes"], task["edges"], task["execution_times"])

        if U_i > 1 and remaining_processors >= max_parallel_tasks:
            print(f"Assigning processors to task {task['task_id']} (U_i > 1). Max parallel tasks: {max_parallel_tasks}")

            assigned_processors = processors[:max_parallel_tasks]
            for p in assigned_processors:
                p.assigned_tasks.append(task["task_id"])
                p.utilization += U_i / max_parallel_tasks
            remaining_processors -= max_parallel_tasks
            processors = processors[max_parallel_tasks:]
        elif U_i <= 1:
            scheduling_result.append((task, U_i))

    #  WFD
    scheduling_result.sort(key=lambda x: x[1], reverse=True)

    for task, U_i in scheduling_result:
        # پیدا کردن پردازنده‌ای که کمترین استفاده را دارد
        available_processors = sorted(processors, key=lambda p: p.utilization)
        for processor in available_processors:
            if processor.utilization + U_i <= 1:
                processor.assigned_tasks.append(task["task_id"])
                processor.utilization += U_i
                break
        else:
            print(f"Task {task['task_id']} cannot be scheduled due to lack of resources.")
    print("\n=== Scheduling Result ===")
    total_used_processors = sum(1 for p in processors_state if p.assigned_tasks)
    for p in processors_state:
        print(f"Processor {p.id}: Assigned Tasks {p.assigned_tasks}, Utilization: {p.utilization:.2f}")

    print(f"\nTotal Processors Used: {total_used_processors}")
    return processors

def lcm(numbers):
    return reduce(lambda x, y: x * y // gcd(x, y), numbers)
def hyperperiod(tasks):
    periods = [task["period"] for task in tasks]
    return lcm(periods)


def generate_periodic_tasks(tasks):
    periodic_tasks = []
    hyper_period = hyperperiod(tasks)

    for task in tasks:
        num_task_instances = hyper_period // task["period"]
        print(f"T {task['task_id']} : {num_task_instances} instances")

        instances = []
        for i in range(1, num_task_instances + 1):
            instance = task.copy()
            instance = {
                "task_id": task["task_id"],
                "release_time": task["period"] * i,
                "absolute_deadline": task["period"] * i + task["period"],
                "instance_id": f"{task['task_id']}-{i}",
                "nodes": task["nodes"],
                "edges": task["edges"],
                "period": task["period"],
                "critical_path": task["critical_path"],
                "critical_path_length": task["critical_path_length"],
                "allocations": task["allocations"],
                "execution_times": task["execution_times"],  # انتقال execution_times
                "assigned_processors": task.get("assigned_processors", {})
            }
            instances.append(instance)

        task["instances"] = instances
        periodic_tasks.append(task)

    return periodic_tasks
def get_all_task_instances(periodic_tasks):
    all_instances = []
    for task in periodic_tasks:
        all_instances.extend(task["instances"])
    return all_instances
def copy(self):
    return copy.deepcopy(self) 
def map_instances_to_cores(processors, periodic_tasks):
    # ایجاد یک نگاشت از task_id به پردازنده‌های تخصیص داده شده
    task_to_processors = {}
    for processor in processors:
        for task_id in processor.assigned_tasks:
            if task_id not in task_to_processors:
                task_to_processors[task_id] = []
            task_to_processors[task_id].append(processor.id)

    # تخصیص instance های هر تسک به همان پردازنده‌های اصلی
    for task in periodic_tasks:
        assigned_processors = task_to_processors.get(task["task_id"], [])
        for instance in task["instances"]:
            instance["assigned_processors"] = assigned_processors

    return periodic_tasks


def edf_scheduling(processors, periodic_tasks):
    # ایجاد یک ساختار داده برای نگهداری تسک‌ها روی هر پردازنده
    core_tasks = {p.id: [] for p in processors}
    for task in periodic_tasks:
        for instance in task["instances"]:
            for core in instance["assigned_processors"]:
                core_tasks[core].append(instance)

    # مرتب‌سازی هر لیست تسک بر اساس Absolute Deadline و Release Time
    for core, instances in core_tasks.items():
        instances.sort(key=lambda x: (x["absolute_deadline"], x["release_time"]))

    return core_tasks