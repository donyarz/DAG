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
    scheduling_result = []

    for task in tasks:
        total_execution_time = task["total_execution_time"]
        period = task["period"]
        U_i = total_execution_time / period

        _, max_parallel_tasks = calculate_asap_cores(task["nodes"], task["edges"], task["execution_times"])

        if U_i > 1:
            num_processors = max_parallel_tasks
        else:
            num_processors = 1

        scheduling_result.append({
            "task_id": task["task_id"],
            "U_i": U_i,
            "num_processors": num_processors
        })

    return scheduling_result

#scheduling:
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

def schedule_tasks(tasks):
    h_period = hyperperiod(tasks)
    current_time = 0
    executed_nodes = set()

    task_processor_allocations = {}
    for task in tasks:
        task_id = task["task_id"]
        task_processor_allocations[task_id] = 1

    resource_queues = {res: Queue() for res in tasks[0]["accesses"].keys()}
    resource_status = {res: None for res in tasks[0]["accesses"].keys()}
    cores_status = {}

    scheduling_log = []

    while current_time < h_period:
        print(f"At time {current_time}...")

        ready_nodes = []
        for task in tasks:
            task_id = task["task_id"]

            for node in task["nodes"]:
                if node == "source" or node == "sink":
                    continue
                all_parents_completed = all(pred in executed_nodes for pred in task["edges"] if pred[1] == node)
                if all_parents_completed and node not in executed_nodes:
                    ready_nodes.append(node)

        print(f"  Ready Nodes: {ready_nodes}")

        if not ready_nodes:
            current_time += 1
            continue

        for node in ready_nodes:
            task = next(task for task in tasks if node in task["nodes"])
            execution_time = task["execution_times"].get(node, 0)

            for resource in task["accesses"].keys():
                if resource_status[resource] is None:
                    resource_status[resource] = node
                    print(f"    Resource {resource} allocated to Node {node}")
                elif resource_status[resource] != node:
                    resource_queues[resource].put(node)
                    print(f"    Resource {resource} added to queue for Node {node}")

            if all(resource_status[res] == node for res in task["accesses"]):
                executed_nodes.add(node)
                scheduling_log.append((current_time, task_id, node, execution_time))
                print(f"    Task {task_id} Node {node} executed at time {current_time}")
            else:
                print(f"    Node {node} waiting for resources.")
                scheduling_log.append((current_time, task_id, node, "Waiting for resources"))

        current_time += 1

        for res, current_node in resource_status.items():
            if current_node in executed_nodes:
                next_node = resource_queues[res].get() if not resource_queues[res].empty() else None
                resource_status[res] = next_node

    return scheduling_log


def print_task_execution_log(task_execution_log):
    print("\nTask Execution Log:")
    for task_log in task_execution_log:
        print(f"Task {task_log['task_id']}, Node {task_log['node_id']}:")
        print(f"  Start Time: {task_log['start_time']}")
        print(f"  End Time: {task_log['end_time']}")
        print(f"  Resources Used: {', '.join(task_log['resource_id'])}")
        print("-" * 40)

def visualize(self, show: bool = False, save: bool = False, title: str = None, filename: str = None):
    self.ax.set_xlabel('Time')
    self.ax.set_ylabel('Tasks')
    self.ax.set_title(title)
    y_max = sum([len(task.nodes) for task in self.tasks])
    self.ax.set_xticks([i for i in range(0, self.hyperperiod, 4)] + [self.hyperperiod])
    self.ax.set_yticks([i for i in range(y_max)])
    self.ax.set_yticklabels([node.id for task in self.tasks for node in task.nodes])
    self.ax.grid(True)
    def map_to_color(resource_id):
        colors = {
            "R1": "red",
            "R2": "blue",
            "R3": "green",
            "R4": "yellow",
            "R5": "purple",
            "R6": "orange",
        }
        return colors.get(resource_id, "black")

    y_offset = 0
    for task in self.tasks:
        for node in task.nodes:
            start_time = node.start_time
            end_time = node.end_time
            resource_id = node.resource_id
            color = map_to_color(resource_id)
            self.ax.barh(y=y_offset, width=end_time - start_time, left=start_time, color=color, edgecolor="black")
            y_offset += 1
    if show:
        plt.show()
    if save and filename:
        self.fig.savefig(f"{filename}.png")
