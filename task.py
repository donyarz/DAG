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
import heapq


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


def erdos_renyi_graph():
    """
    Generates a random directed acyclic graph (DAG) using Erdos-Renyi model.
    
    Returns:
        tuple: (nodes, edges) where:
            - nodes: list of node IDs
            - edges: list of tuples (source, target)
    """
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
    
    # Convert to lists
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    return G

def predecessors(g):
    # g= erdos_renyi_graph()
    for node in g.nodes:
        print(f"N{node} : {g.nodes[node]["wcet"]}\n -----------------")
        predecessors= g.predecessors(node)
        print(f"predecessors")
        for pred in predecessors:
            print(f"N{pred}")
        print(f"successors")
        successors = g.successors(node)
        # print(f"pred:{predecessors}")
        # print(f"suc: {successors}")
        for succe in successors:
            print(f"N{succe}")
        print("---------------------")

''' def assignwcet(g):
    for node in g.nodes:
        g.nodes[node]["wcet"]= 2 '''

def visualize_task(G):
    # G = nx.DiGraph()
    # G.add_nodes_from(task["nodes"])
    # G.add_edges_from(task["edges"])


    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", node_size=700, font_size=10,
            edge_color="gray")
    # plt.title(f"Task {task['task_id']} DAG", fontsize=14)
    plt.title(f"DAG", fontsize=14)
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
    """
    Generates a task with random DAG structure and execution times.
    """
    # Get the graph from erdos_renyi_graph
    G = erdos_renyi_graph()
    
    # Get nodes and edges from the graph
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    if len(nodes) <= 2:
        print(f"Skipping Task {task_id}: No nodes other than source and sink.")
        return None

    # Assign execution times to nodes (excluding source and sink)
    execution_times = {}
    for node in nodes:
        if node not in ["source", "sink"]:
            execution_times[node] = random.randint(13, 30)
    
    # Calculate critical path and other metrics
    critical_path, critical_path_length = get_critical_path(nodes, edges, execution_times)
    total_execution_time = sum(execution_times.values())
    period = int(critical_path_length / rand.uniform(0.125, 0.25))
    deadline = period
    U_i = round(total_execution_time / period, 2)
    
    # Calculate ASAP schedule and max parallel tasks
    asap_schedule, max_parallel_tasks = calculate_asap_cores(nodes, edges, execution_times)

    # Allocate resources to nodes
    allocations, execution_times = allocate_resources_to_nodes(
        {"nodes": nodes, "edges": edges, "execution_times": execution_times}, 
        task_id, 
        accesses, 
        lengths
    )

    # Print resource access information
    print(f"\n=== Resource Access Information for Task {task_id} ===")
    for node_id, node_allocations in allocations.items():
        print(f"\nNode {node_id} Resource Access Pattern:")
        for section in node_allocations:
            if isinstance(section, tuple):
                resource_id, duration = section
                print(f"  - {resource_id}: {duration} time units")
            else:
                print(f"  - Normal Section: {section} time units")

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
        print(f"T {task['task_id']} : {num_task_instances + 1} instances")  # +1 چون از 0 شروع می‌کنیم

        instances = []
        for i in range(num_task_instances + 1):
            instance = {
                "task_id": task["task_id"],
                "release_time": task["period"] * i,
                "absolute_deadline": task["period"] * i + task["period"],
                "instance_id": f"{task['task_id']}-{i + 1}",
                "nodes": task["nodes"],
                "edges": task["edges"],
                "period": task["period"],
                "critical_path": task.get("critical_path", []),
                "critical_path_length": task.get("critical_path_length", 0),
                "allocations": task.get("allocations", {}),
                "execution_times": task["execution_times"],
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
    task_to_processors = {}
    for processor in processors:
        for task_id in processor.assigned_tasks:
            if task_id not in task_to_processors:
                task_to_processors[task_id] = []
            task_to_processors[task_id].append(processor.id)

    for task in periodic_tasks:
        assigned_processors = task_to_processors.get(task["task_id"], [])
        for instance in task["instances"]:
            instance["assigned_processors"] = assigned_processors

    return periodic_tasks

def edf_scheduling(processors, periodic_tasks):
    core_tasks = {p.id: [] for p in processors}

    for task in periodic_tasks:
        for instance in task["instances"]:
            for core in instance["assigned_processors"]:
                core_tasks[core].append(instance)

    for core in core_tasks:
        core_tasks[core] = sorted(core_tasks[core], key=lambda x: x["absolute_deadline"])

    return core_tasks


def find_ready_nodes(nodes: list, edges: list[tuple], completed_nodes: set) -> list:

    if not isinstance(completed_nodes, set):
        completed_nodes = set(completed_nodes)
        
    ready_nodes = []

    for node in nodes:
        # Skip sink and already completed nodes
        if node == "sink" or node in completed_nodes:
            continue
            
        # Get all predecessors of current node
        predecessors = [src for src, dest in edges if dest == node]
        
        # If node has no predecessors or all predecessors are completed
        if not predecessors or all(pred in completed_nodes for pred in predecessors):
            ready_nodes.append(node)

    return ready_nodes


@dataclass
class NodeExecution:
    node_id: str
    wcet: int
    remaining_time: int
    start_time: int = 0

def print_execution_status(time: int, executing_nodes: dict, completed_nodes: set, ready_nodes: list):
    """
    Prints the current execution status of all nodes.
    
    Args:
        time (int): Current simulation time
        executing_nodes (dict): Dictionary of currently executing nodes
        completed_nodes (set): Set of completed nodes
        ready_nodes (list): List of ready nodes
    """
    print(f"\n=== Time {time} ===")
    print("Executing Nodes:")
    for node_id, node_exec in executing_nodes.items():
        print(f"  Node {node_id}: Remaining Time = {node_exec.remaining_time}/{node_exec.wcet}")
    
    print("\nReady Nodes (can start execution):")
    for node in ready_nodes:
        print(f"  Node {node}")
    
    print("\nCompleted Nodes:")
    for node in sorted(completed_nodes):
        print(f"  Node {node}")

def schedule_tasks_with_visualization(nodes: list[Node], edges: list[tuple], execution_times: dict) -> dict:
    """
    Schedules tasks and shows execution status at each time unit.
    
    Args:
        nodes (list[Node]): List of nodes with their WCET
        edges (list[tuple]): List of edges representing dependencies
        execution_times (dict): Dictionary mapping node IDs to their WCET
        
    Returns:
        dict: Dictionary containing scheduling information
    """
    current_time = 0
    completed_nodes = set()
    executing_nodes = {}  # {node_id: NodeExecution}
    scheduling_log = []
    
    print("=== Starting Task Execution ===")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Execution Times: {execution_times}")
    print(f"Dependencies: {edges}")
    
    while len(completed_nodes) < len(nodes):
        # Find ready nodes
        ready_nodes = find_ready_nodes(nodes, edges, completed_nodes)
        
        # Start execution of ready nodes
        for node_id in ready_nodes:
            if node_id not in executing_nodes:
                wcet = execution_times[node_id]
                executing_nodes[node_id] = NodeExecution(
                    node_id=node_id,
                    wcet=wcet,
                    remaining_time=wcet,
                    start_time=current_time
                )
                print(f"\nNode {node_id} started execution at time {current_time}")
        
        # Show current status
        print_execution_status(current_time, executing_nodes, completed_nodes, ready_nodes)
        
        # Update remaining time for executing nodes
        for node_id, node_exec in list(executing_nodes.items()):
            node_exec.remaining_time -= 1
            
            if node_exec.remaining_time <= 0:
                completed_nodes.add(node_id)
                del executing_nodes[node_id]
                print(f"\nNode {node_id} completed at time {current_time}")
                scheduling_log.append({
                    'time': current_time,
                    'node': node_id,
                    'action': 'completed',
                    'start_time': node_exec.start_time,
                    'end_time': current_time
                })
        
        current_time += 1
        
        # Break if no progress is being made
        if not executing_nodes and not ready_nodes and len(completed_nodes) < len(nodes):
            print(f"\nWarning: Deadlock detected at time {current_time}")
            break

    print("\n=== Execution Summary ===")
    print(f"Total Execution Time: {current_time}")
    print(f"Completed Nodes: {sorted(completed_nodes)}")
    
    return {
        'completed_nodes': completed_nodes,
        'scheduling_log': scheduling_log,
        'total_time': current_time
    }

@dataclass
class ResourceLock:
    is_locked: bool = False
    locked_by: str = None  # node_id that currently holds the lock
    waiting_queue: Queue = None  # FIFO queue for waiting nodes
    
    def __init__(self):
        self.is_locked = False
        self.locked_by = None
        self.waiting_queue = Queue()

class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, ResourceLock] = {}
        self.node_resources: Dict[str, List[str]] = {}  # node_id -> list of resources it needs
        
    def initialize_resources(self, resource_ids: List[str]):
        for resource_id in resource_ids:
            self.resources[resource_id] = ResourceLock()
    
    def request_resource(self, node_id: str, resource_id: str) -> bool:
        """
        Request a resource using spin lock protocol with FIFO queue.
        Returns True if resource is acquired, False if added to waiting queue.
        """
        if resource_id not in self.resources:
            return True  # Resource doesn't exist, no need to lock
            
        lock = self.resources[resource_id]
        
        if not lock.is_locked:
            lock.is_locked = True
            lock.locked_by = node_id
            return True
        else:
            # Add to waiting queue if not already in it
            if node_id not in [item for item in list(lock.waiting_queue.queue)]:
                lock.waiting_queue.put(node_id)
            return False
    
    def release_resource(self, node_id: str, resource_id: str):
        """Release a resource and give it to the next node in the queue."""
        if resource_id not in self.resources:
            return
            
        lock = self.resources[resource_id]
        if lock.locked_by == node_id:
            lock.is_locked = False
            lock.locked_by = None
            
            # Give resource to next node in queue
            if not lock.waiting_queue.empty():
                next_node = lock.waiting_queue.get()
                lock.is_locked = True
                lock.locked_by = next_node
    
    def get_waiting_nodes(self, resource_id: str) -> List[str]:
        """Get list of nodes waiting for a resource."""
        if resource_id not in self.resources:
            return []
        return list(self.resources[resource_id].waiting_queue.queue)

def schedule_with_processors(task: dict, processors: list[Processor]) -> dict:
    """
    Schedules task nodes on allocated processors with parallel execution support using Critical Path scheduling.
    Implements resource locking with FIFO queue and spin lock protocol.
    """
    nodes = task["nodes"]
    edges = task["edges"]
    execution_times = task["execution_times"]
    task_id = task["task_id"]
    deadline = task["deadline"]
    allocations = task["allocations"]
    
    # Initialize resource manager
    resource_manager = ResourceManager()
    resource_manager.initialize_resources([f"R{i+1}" for i in range(6)])  # Assuming 6 resources
    
    # Visualize the task graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    visualize_task(G)
    
    # Find critical path
    critical_path = get_critical_path(nodes, edges, execution_times)[0]
    print(f"\nCritical Path: {critical_path}")
    
    current_time = 0
    completed_nodes = set()
    executing_nodes = {}  # {processor_id: (node_id, remaining_time, current_section_index)}
    scheduling_log = []
    
    # Get number of processors allocated to this task
    num_processors = len(processors)
    print(f"\n=== Starting Task {task_id} Execution ===")
    print(f"Number of Processors: {num_processors}")
    print(f"Nodes: {nodes}")
    print(f"Execution Times: {execution_times}")
    print(f"Dependencies: {edges}")
    print(f"Deadline: {deadline}")
    
    # Add source and sink to completed nodes since they don't need execution
    completed_nodes.add("source")
    completed_nodes.add("sink")
    
    while len(completed_nodes) < len(nodes):
        # Check if current time exceeds deadline
        if current_time > deadline:
            print(f"\nWarning: Task {task_id} missed its deadline at time {current_time} (deadline was {deadline})")
            return {
                'completed_nodes': completed_nodes,
                'scheduling_log': scheduling_log,
                'total_time': current_time,
                'missed_deadline': True
            }
        
        # Find ready nodes
        ready_nodes = find_ready_nodes(nodes, edges, completed_nodes)
        
        # Prioritize nodes on critical path
        critical_ready_nodes = [node for node in ready_nodes if node in critical_path]
        non_critical_ready_nodes = [node for node in ready_nodes if node not in critical_path]
        
        # Sort critical nodes by their position in critical path
        critical_ready_nodes.sort(key=lambda x: critical_path.index(x))
        
        # Find non-critical nodes that are predecessors of critical nodes
        critical_predecessors = []
        non_critical_others = []
        
        for node in non_critical_ready_nodes:
            is_predecessor = False
            for critical_node in critical_path:
                if (node, critical_node) in edges:
                    is_predecessor = True
            break

            if is_predecessor:
                critical_predecessors.append(node)
            else:
                non_critical_others.append(node)
        
        # Combine ready nodes with priority order
        prioritized_ready_nodes = critical_ready_nodes + critical_predecessors + non_critical_others
        
        # Assign ready nodes to available processors
        available_processors = [p.id for p in processors if p.id not in executing_nodes]
        for node_id in prioritized_ready_nodes:
            if available_processors:
                # Check if node can acquire all required resources
                node_allocations = allocations.get(node_id, [])
                can_acquire_resources = True
                
                for section in node_allocations:
                    if isinstance(section, tuple) and section[0] != "Normal":
                        resource_id = section[0]
                        if not resource_manager.request_resource(node_id, resource_id):
                            can_acquire_resources = False
                            break
                
                if can_acquire_resources:
                    processor_id = available_processors.pop(0)
                    executing_nodes[processor_id] = (node_id, execution_times[node_id], 0)
                    print(f"\nNode {node_id} (Task {task_id}) started execution on processor {processor_id} at time {current_time}")
        else:
                    print(f"\nNode {node_id} is waiting for resources")
        
        # Show current status
        print(f"\n=== Time {current_time} ===")
        print("Executing Nodes:")
        for proc_id, (node_id, remaining, section_idx) in executing_nodes.items():
            print(f"  Processor {proc_id}: Node {node_id} (Task {task_id}) (Remaining Time = {remaining})")
        
        print("\nResource Status:")
        for resource_id, lock in resource_manager.resources.items():
            if lock.is_locked:
                print(f"  Resource {resource_id}: Locked by Node {lock.locked_by}")
                waiting_nodes = resource_manager.get_waiting_nodes(resource_id)
                if waiting_nodes:
                    print(f"    Waiting Queue (FIFO):")
                    for i, waiting_node in enumerate(waiting_nodes):
                        # Find which task this node belongs to
                        for task in tasks:
                            if waiting_node in task["nodes"]:
                                print(f"      {i+1}. Node {waiting_node} (Task {task['task_id']})")
            break

        print("\nCompleted Nodes:")
        for node in sorted(completed_nodes, key=lambda x: str(x)):
            print(f"  Node {node}")
        
        # Update remaining time for executing nodes
        for processor_id, (node_id, remaining, section_idx) in list(executing_nodes.items()):
            node_allocations = allocations.get(node_id, [])
            
            if section_idx < len(node_allocations):
                current_section = node_allocations[section_idx]
                
                if isinstance(current_section, tuple):
                    section_type, section_time = current_section
                    if section_type != "Normal":
                        # Critical section - check if we still have the resource
                        if resource_manager.resources[section_type].locked_by != node_id:
                            continue  # Skip this time unit if we lost the resource
                
                # Update remaining time for current section
                remaining -= 1
                executing_nodes[processor_id] = (node_id, remaining, section_idx)
                
                # Check if current section is completed
                if remaining <= 0:
                    # Move to next section or complete node
                    if section_idx + 1 < len(node_allocations):
                        next_section = node_allocations[section_idx + 1]
                        if isinstance(next_section, tuple):
                            section_type, section_time = next_section
                            executing_nodes[processor_id] = (node_id, node_id, section_time, section_idx + 1, current_time)
                            print(f"\nNode {node_id} (Task {task_id}) moving to next section: {section_type} for {section_time} time units")
                        else:
                            executing_nodes[processor_id] = (node_id, node_id, next_section, section_idx + 1, current_time)
                            print(f"\nNode {node_id} (Task {task_id}) moving to next section: Normal for {next_section} time units")
                    else:
                        # Release all resources held by this node
                        for section in node_allocations:
                            if isinstance(section, tuple) and section[0] != "Normal":
                                resource_manager.release_resource(node_id, section[0])
                        
                        completed_nodes.add(node_id)
                        del executing_nodes[processor_id]
                        print(f"\nNode {node_id} (Task {task_id}) completed on processor {processor_id} at time {current_time}")
                        scheduling_log.append({
                            'time': current_time,
                            'task_id': task_id,
                            'node': node_id,
                            'processor': processor_id,
                            'action': 'completed'
                        })
        
        current_time += 1
        
        # Break if no progress is being made
        if not executing_nodes and not ready_nodes and len(completed_nodes) < len(nodes):
            print(f"\nWarning: Deadlock detected at time {current_time}")
            break
    
    print("\n=== Execution Summary ===")
    print(f"Total Execution Time: {current_time}")
    print(f"Completed Nodes: {sorted(completed_nodes, key=lambda x: str(x))}")
    
    return {
        'completed_nodes': completed_nodes,
        'scheduling_log': scheduling_log,
        'total_time': current_time,
        'missed_deadline': False
    }

def execute_task_with_processors(task: dict, processors: list[Processor]) -> dict:
    """
    Executes a task using the allocated processors with parallel execution support.
    
    Args:
        task (dict): Task information including nodes, edges, and execution times
        processors (list[Processor]): List of processors allocated to the task
        
    Returns:
        dict: Execution results including timeline and processor utilization
    """
    # Schedule the task on allocated processors
    result = schedule_with_processors(task, processors)
    
    # Calculate processor utilization
    total_execution_time = result['total_time']
    processor_utilization = {}
    
    for processor in processors:
        processor_id = processor.id
        execution_time = sum(1 for log in result['scheduling_log'] 
                           if log['processor'] == processor_id)
        utilization = execution_time / total_execution_time if total_execution_time > 0 else 0
        processor_utilization[processor_id] = utilization
    
    result['processor_utilization'] = processor_utilization
    return result

def schedule_multiple_tasks(tasks: list[dict], processors: list[Processor]) -> dict:
    """
    Schedules multiple tasks using federated scheduling and Critical Path scheduling.
    Implements system-wide resource management with FIFO queue and spin lock protocol.
    """
    # Use federated scheduling to allocate processors
    allocated_processors = federated_scheduling(tasks)
    print(f"\n=== Processor Allocation ===")
    for p in allocated_processors:
        print(f"Processor {p.id}: Assigned Tasks {p.assigned_tasks}, Utilization: {p.utilization:.2f}")
    
    # Create task to processor mapping
    task_to_processors = {}
    for processor in allocated_processors:
        for task_id in processor.assigned_tasks:
            if task_id not in task_to_processors:
                task_to_processors[task_id] = []
            task_to_processors[task_id].append(processor.id)
    
    print("\n=== Task to Processor Mapping ===")
    for task_id, processor_ids in task_to_processors.items():
        print(f"Task {task_id} can only execute on processors: {processor_ids}")
    
    # Initialize system-wide resource manager
    resource_manager = ResourceManager()
    resource_manager.initialize_resources([f"R{i+1}" for i in range(6)])  # Assuming 6 resources
    
    # Schedule tasks on their allocated processors
    results = {}
    current_time = 0
    completed_nodes = {task["task_id"]: set() for task in tasks}
    executing_nodes = {}  # {processor_id: (task_id, node_id, remaining_time, current_section_index, section_start_time)}
    scheduling_log = []
    
    # Add source and sink nodes to completed nodes for all tasks
    for task in tasks:
        completed_nodes[task["task_id"]].add("source")
        completed_nodes[task["task_id"]].add("sink")
    
    while any(len(completed_nodes[task["task_id"]]) < len(task["nodes"]) for task in tasks):
        # Check if current time exceeds any task's deadline
        for task in tasks:
            if current_time > task["deadline"]:
                print(f"\nWarning: Task {task['task_id']} missed its deadline at time {current_time} (deadline was {task['deadline']})")
                results[task["task_id"]] = {
                    'completed_nodes': completed_nodes[task["task_id"]],
                    'scheduling_log': [log for log in scheduling_log if log['task_id'] == task["task_id"]],
                    'total_time': current_time,
                    'missed_deadline': True
                }
        
        # Find ready nodes for each task
        ready_nodes_by_task = {}
        for task in tasks:
            if task["task_id"] not in results or not results[task["task_id"]]["missed_deadline"]:
                ready_nodes = find_ready_nodes(task["nodes"], task["edges"], completed_nodes[task["task_id"]])
                ready_nodes_by_task[task["task_id"]] = ready_nodes
        
        # Prioritize nodes on critical paths
        prioritized_nodes = []
        for task in tasks:
            if task["task_id"] not in results or not results[task["task_id"]]["missed_deadline"]:
                task_ready_nodes = ready_nodes_by_task[task["task_id"]]
                critical_path = task["critical_path"]
                
                # Prioritize critical path nodes
                critical_ready_nodes = [node for node in task_ready_nodes if node in critical_path]
                non_critical_ready_nodes = [node for node in task_ready_nodes if node not in critical_path]
                
                # Sort critical nodes by their position in critical path
                critical_ready_nodes.sort(key=lambda x: critical_path.index(x))
                
                # Find non-critical nodes that are predecessors of critical nodes
                critical_predecessors = []
                non_critical_others = []
                
                for node in non_critical_ready_nodes:
                    is_predecessor = False
                    for critical_node in critical_path:
                        if (node, critical_node) in task["edges"]:
                            is_predecessor = True
                            break
                    
                    if is_predecessor:
                        critical_predecessors.append(node)
                    else:
                        non_critical_others.append(node)
                
                # Add nodes to prioritized list with task_id and priority level
                for node in critical_ready_nodes:
                    prioritized_nodes.append((task["task_id"], node, "Critical Path"))
                for node in critical_predecessors:
                    prioritized_nodes.append((task["task_id"], node, "Critical Predecessor"))
                for node in non_critical_others:
                    prioritized_nodes.append((task["task_id"], node, "Non-Critical"))
        
        # Assign ready nodes to available processors
        for task_id, node_id, priority_level in prioritized_nodes:
            # Get available processors for this specific task
            task_processors = task_to_processors.get(task_id, [])
            available_task_processors = [p_id for p_id in task_processors if p_id not in executing_nodes]
            
            if available_task_processors:
                task = next(t for t in tasks if t["task_id"] == task_id)
                # Check if node can acquire all required resources
                node_allocations = task["allocations"].get(node_id, [])
                can_acquire_resources = True
                
                for section in node_allocations:
                    if isinstance(section, tuple) and section[0] != "Normal":
                        resource_id = section[0]
                        if not resource_manager.request_resource(node_id, resource_id):
                            can_acquire_resources = False
                            break
                
                if can_acquire_resources:
                    processor_id = available_task_processors[0]
                    # Start with the first section
                    first_section = node_allocations[0]
                    if isinstance(first_section, tuple):
                        section_type, section_time = first_section
                        executing_nodes[processor_id] = (task_id, node_id, section_time, 0, current_time)
                    else:
                        executing_nodes[processor_id] = (task_id, node_id, first_section, 0, current_time)
                    print(f"\nNode {node_id} (Task {task_id}) started execution on processor {processor_id} at time {current_time}")
                    print(f"  Priority Level: {priority_level}")
                else:
                    print(f"\nNode {node_id} (Task {task_id}) is waiting for resources")
        
        # Show current status
        print(f"\n=== Time {current_time} ===")
        print("Executing Nodes:")
        for proc_id, (task_id, node_id, remaining, section_idx, start_time) in executing_nodes.items():
            task = next(t for t in tasks if t["task_id"] == task_id)
            node_allocations = task["allocations"].get(node_id, [])
            current_section = node_allocations[section_idx]
            if isinstance(current_section, tuple):
                section_type, section_time = current_section
                if section_type != "Normal":
                    elapsed = current_time - start_time
                    remaining_time = max(0, section_time - elapsed)
                    section_info = f" (Resource {section_type}: {remaining_time}/{section_time} time units remaining)"
                else:
                    section_info = f" (Normal Section: {remaining}/{section_time} time units remaining)"
            else:
                section_info = f" (Normal Section: {remaining} time units remaining)"
            
            print(f"  Processor {proc_id}: Node {node_id} (Task {task_id}){section_info}")
        
        print("\nResource Status:")
        for resource_id, lock in resource_manager.resources.items():
            if lock.is_locked:
                print(f"  {resource_id}: Locked by Node {lock.locked_by}")
                if not lock.waiting_queue.empty():
                    waiting = list(lock.waiting_queue.queue)
                    print(f"    Waiting queue: {waiting}")
            else:
                print(f"  {resource_id}: Available")
        
        # Update remaining time for executing nodes
        for processor_id, (task_id, node_id, remaining, section_idx, start_time) in list(executing_nodes.items()):
            task = next(t for t in tasks if t["task_id"] == task_id)
            node_allocations = task["allocations"].get(node_id, [])
            
            if section_idx < len(node_allocations):
                current_section = node_allocations[section_idx]
                
                if isinstance(current_section, tuple):
                    section_type, section_time = current_section
                    if section_type != "Normal":
                        # Check if we need to request the resource at this time
                        elapsed = current_time - start_time
                        if elapsed == 0:  # Request resource at the start of critical section
                            if not resource_manager.request_resource(node_id, section_type):
                                print(f"\nNode {node_id} (Task {task_id}) is waiting for resource {section_type}")
                                continue
                        # Critical section - check if we still have the resource
                        elif resource_manager.resources[section_type].locked_by != node_id:
                            continue  # Skip this time unit if we lost the resource
                
                # Update remaining time for current section
                remaining -= 1
                executing_nodes[processor_id] = (task_id, node_id, remaining, section_idx, start_time)
                
                # Check if current section is completed
                if remaining <= 0:
                    # Move to next section or complete node
                    if section_idx + 1 < len(node_allocations):
                        next_section = node_allocations[section_idx + 1]
                        if isinstance(next_section, tuple):
                            section_type, section_time = next_section
                            executing_nodes[processor_id] = (task_id, node_id, section_time, section_idx + 1, current_time)
                            print(f"\nNode {node_id} (Task {task_id}) moving to next section: {section_type} for {section_time} time units")
                        else:
                            executing_nodes[processor_id] = (task_id, node_id, next_section, section_idx + 1, current_time)
                            print(f"\nNode {node_id} (Task {task_id}) moving to next section: Normal for {next_section} time units")
                    else:
                        # Release all resources held by this node
                        for section in node_allocations:
                            if isinstance(section, tuple) and section[0] != "Normal":
                                resource_manager.release_resource(node_id, section[0])
                        
                        completed_nodes[task_id].add(node_id)
                        del executing_nodes[processor_id]
                        print(f"\nNode {node_id} (Task {task_id}) completed on processor {processor_id} at time {current_time}")
                        scheduling_log.append({
                            'time': current_time,
                            'task_id': task_id,
                            'node': node_id,
                            'processor': processor_id,
                            'action': 'completed'
                        })
        
        current_time += 1
        
        # Break if no progress is being made
        if not executing_nodes and not any(ready_nodes_by_task.values()) and \
           any(len(completed_nodes[task["task_id"]]) < len(task["nodes"]) for task in tasks):
            print(f"\nWarning: Deadlock detected at time {current_time}")
            break
    
    # Add results for tasks that didn't miss deadline
    for task in tasks:
        if task["task_id"] not in results:
            results[task["task_id"]] = {
                'completed_nodes': completed_nodes[task["task_id"]],
                'scheduling_log': [log for log in scheduling_log if log['task_id'] == task["task_id"]],
                'total_time': current_time,
                'missed_deadline': False
            }
    
    print("\n=== Execution Summary ===")
    print(f"Total Execution Time: {current_time}")
    for task_id, result in results.items():
        print(f"Task {task_id}: {'Completed' if not result['missed_deadline'] else 'Missed Deadline'}")
        print(f"  Completed Nodes: {sorted(result['completed_nodes'], key=lambda x: str(x))}")
    
    return results

def run_complete_example():
    """
    Demonstrates the complete workflow:
    1. Generate multiple tasks
    2. Allocate processors using federated scheduling
    3. Execute tasks with the allocated processors
    """
    print("=== Starting Complete Example ===")
    
    # Step 1: Generate tasks
    print("\n1. Generating Tasks...")
    num_tasks = 2
    num_resources = 3
    tasks = []
    
    # Generate accesses and lengths for all tasks
    accesses, lengths = generate_accesses_and_lengths(num_tasks=num_tasks, num_resources=num_resources)
    
    for task_id in range(1, num_tasks + 1):
        task = generate_task(task_id=task_id, accesses=accesses, lengths=lengths)
        if task is not None:
            tasks.append(task)
            print(f"\nGenerated Task {task_id}:")
            print(f"Nodes: {task['nodes']}")
            print(f"Edges: {task['edges']}")
            print(f"Execution Times: {task['execution_times']}")
            print(f"Critical Path: {task['critical_path']}")
            print(f"Critical Path Length: {task['critical_path_length']}")
            print(f"Utilization: {task['utilization']:.2f}")
            print(f"Deadline: {task['deadline']}")
    
    if not tasks:
        print("Failed to generate any tasks")
        return
    
    print(f"\nGenerated {len(tasks)} tasks:")
    for task in tasks:
        print(f"Task {task['task_id']}: U = {task['utilization']:.2f}, Deadline = {task['deadline']}")
    
    # Step 2: Allocate processors using federated scheduling
    print("\n2. Allocating Processors...")
    total_processors = calculate_total_processors(tasks)
    processors = [Processor(id=i+1, assigned_tasks=[]) for i in range(total_processors)]
    print(f"Total Processors: {total_processors}")
    
    # Step 3: Execute tasks
    print("\n3. Executing Tasks...")
    results = schedule_multiple_tasks(tasks, processors)
    
    # Print final results
    print("\n=== Final Results ===")
    for task_id, result in results.items():
        status = "Missed Deadline" if result.get('missed_deadline', False) else "Completed Successfully"
        print(f"Task {task_id}: Total Time = {result['total_time']}, Status = {status}")
    
    return results

# اجرای مثال کامل
if __name__ == "__main__":
    run_complete_example()




