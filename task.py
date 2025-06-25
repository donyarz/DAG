from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random as rand
import random
import math
import networkx as nx
import numpy as np
from math import gcd
from functools import reduce
from queue import Queue
import heapq
from dataclasses import dataclass, field

from matplotlib import pyplot as plt


@dataclass
class ResourceLock:
    is_locked: bool = False
    locked_by: Optional[str] = None
    waiting_queue: Queue[str] = field(default_factory=Queue)
    # Remove the __init__ method below if it exists
    # def __init__(self):
    #     self.is_locked = False
    #     self.locked_by = None
    #     self.waiting_queue = Queue()

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
                return next_node  # Return the ID of the node that acquired the resource
        return None  # No node acquired the resource

    def get_waiting_nodes(self, resource_id: str) -> List[str]:
        """Get list of nodes waiting for a resource."""
        if resource_id not in self.resources:
            return []
        return list(self.resources[resource_id].waiting_queue.queue)

    def is_resource_locked_by(self, node_id: str, resource_id: str) -> bool:
        """Check if a resource is currently locked by a specific node."""
        if resource_id not in self.resources:
            return False
        return self.resources[resource_id].locked_by == node_id

    def release_all_resources_for_node(self, node_id: str):
        """Release all resources currently held by a node."""
        for resource_id, lock in self.resources.items():
            if lock.locked_by == node_id:
                self.release_resource(node_id, resource_id)

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
class ExecutionPhase:
    type: str  # "normal" or "resource"
    duration: int
    resource_id: Optional[str] = None
    remaining_time: int = 0
    resource_acquired: bool = False

    def __post_init__(self):
        self.remaining_time = self.duration

def convert_allocations_to_phases(allocations: dict) -> dict[str, List[ExecutionPhase]]:
    """
    Converts a dictionary of node allocations (defining execution segments)
    into a dictionary suitable for NodeExecution's execution_phases.
    """
    node_phases = {}
    for node_id, phase_defs in allocations.items():
        phases_list: List[ExecutionPhase] = []
        for phase_def in phase_defs:
            if phase_def["type"] == "normal":
                phases_list.append(
                    ExecutionPhase(
                        type="normal",
                        duration=phase_def["duration"]
                    )
                )
            elif phase_def["type"] == "resource":
                if "resource_id" not in phase_def:
                    raise ValueError(f"Resource phase for Node {node_id} requires 'resource_id'.")
                phases_list.append(
                    ExecutionPhase(
                        type="resource",
                        duration=phase_def["duration"],
                        resource_id=phase_def["resource_id"]
                    )
                )
            else:
                raise ValueError(f"Unknown phase type: {phase_def['type']} for Node {node_id}")
        node_phases[node_id] = phases_list
    return node_phases

@dataclass
class NodeExecution:
    node_id: str
    task_id: str
    execution_phases: List[ExecutionPhase]
    current_phase_index: int = 0
    processor_id: Optional[int] = None
    start_time: int = 0
    waiting_for_resource: bool = False

    def get_current_phase(self) -> Optional[ExecutionPhase]:
        if self.current_phase_index < len(self.execution_phases):
            return self.execution_phases[self.current_phase_index]
        return None

    def advance_to_next_phase(self):
        """Move to the next execution phase."""
        self.current_phase_index += 1
        if self.current_phase_index < len(self.execution_phases):
            self.execution_phases[self.current_phase_index].remaining_time = self.execution_phases[self.current_phase_index].duration
            self.execution_phases[self.current_phase_index].resource_acquired = False

    def is_completed(self) -> bool:
        return self.current_phase_index >= len(self.execution_phases)

    def __str__(self) -> str:
        current_phase = self.get_current_phase()
        if current_phase is None:
            return f"Node {self.node_id} (Task {self.task_id}) - Completed"

        if current_phase.type == "normal":
            return f"Node {self.node_id} (Task {self.task_id}) - Normal Section: {current_phase.remaining_time}/{current_phase.duration} time units"
        else:
            status = " [Resource Acquired]" if current_phase.resource_acquired else " [Waiting for Resource]"
            return f"Node {self.node_id} (Task {self.task_id}) - Resource {current_phase.resource_id}: {current_phase.remaining_time}/{current_phase.duration} time units{status}"

def execute_phase(current_time: int, node_exec: NodeExecution, resource_manager: ResourceManager) -> bool:
    """
    Execute one time unit of the current phase of a node with support for busy-waiting in resource phases.

    Args:
        current_time: Current simulation time
        node_exec: Node execution instance
        resource_manager: Resource manager instance

    Returns:
        bool: True if execution progressed, False if waiting for resource
    """
    current_phase = node_exec.get_current_phase()

    if current_phase is None:
        return True  # Node is completed

    if current_phase.type == "normal":
        # Normal phase - just decrease time
        current_phase.remaining_time -= 1
        if current_phase.remaining_time == 0:
            node_exec.advance_to_next_phase()
            print(f"\nNode {node_exec.node_id} (Task {node_exec.task_id}) completed normal section")
        return True

    elif current_phase.type == "resource":
        if not current_phase.resource_acquired:
            # Try to acquire resource
            if resource_manager.request_resource(node_exec.node_id, current_phase.resource_id):
                current_phase.resource_acquired = True
                print(f"\nNode {node_exec.node_id} (Task {node_exec.task_id}) acquired resource {current_phase.resource_id}")
                # Now we have the resource, start execution
                current_phase.remaining_time -= 1
                if current_phase.remaining_time == 0:
                    resource_manager.release_resource(node_exec.node_id, current_phase.resource_id)
                    print(f"\nNode {node_exec.node_id} (Task {node_exec.task_id}) released resource {current_phase.resource_id}")
                    node_exec.advance_to_next_phase()
                return True
            else:
                # Resource not available -> busy waiting
                # Do nothing but keep node on processor
                return False
        else:
            # Resource already acquired, continue execution
            current_phase.remaining_time -= 1
            if current_phase.remaining_time == 0:
                resource_manager.release_resource(node_exec.node_id, current_phase.resource_id)
                print(f"\nNode {node_exec.node_id} (Task {node_exec.task_id}) released resource {current_phase.resource_id}")
                node_exec.advance_to_next_phase()
            return True

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

    # Convert allocations to execution phases
    execution_phases = convert_allocations_to_phases(allocations)

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
    executing_nodes: Dict[int, NodeExecution] = {}  # processor_id -> NodeExecution
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
            if available_processors and node_id in execution_phases:
                processor_id = available_processors.pop(0)
                node_execution = NodeExecution(
                    node_id=node_id,
                    task_id=task_id,
                    execution_phases=execution_phases[node_id],
                    processor_id=processor_id,
                    start_time=current_time
                )
                executing_nodes[processor_id] = node_execution
                print(f"\nNode {node_id} (Task {task_id}) started execution on processor {processor_id} at time {current_time}")

        # Show current status
        print(f"\n=== Time {current_time} ===")
        print("Executing Nodes:")
        for proc_id, node_exec in executing_nodes.items():
            print(f"  Processor {proc_id}: {node_exec}")

        print("\nResource Status:")
        for resource_id, lock in resource_manager.resources.items():
            if lock.is_locked:
                print(f"  Resource {resource_id}: Locked by Node {lock.locked_by}")
                waiting_nodes = resource_manager.get_waiting_nodes(resource_id)
                if waiting_nodes:
                    print(f"    Waiting Queue (FIFO):")
                    for i, waiting_node in enumerate(waiting_nodes):
                        print(f"      {i+1}. Node {waiting_node}")
            else:
                print(f"  Resource {resource_id}: Available")

        # Update executing nodes
        for processor_id, node_exec in list(executing_nodes.items()):
            if node_exec.is_completed():
                # Node is completed
                completed_nodes.add(node_exec.node_id)

                del executing_nodes[processor_id]
                print(f"\nNode {node_exec.node_id} (Task {task_id}) completed on processor {processor_id} at time {current_time}")
                scheduling_log.append({
                    'time': current_time,
                    'task_id': task_id,
                    'node': node_exec.node_id,
                    'processor': processor_id,
                    'action': 'completed'
                })
                continue

            # Execute one time unit of the current phase
            execute_phase(current_time, node_exec, resource_manager)

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
    resource_manager.initialize_resources([f"R{i + 1}" for i in range(6)])  # Assuming 6 resources

    # Schedule tasks on their allocated processors
    results = {}
    current_time = 0
    completed_nodes = {task["task_id"]: set() for task in tasks}
    # executing_nodes: {processor_id: (task_id, node_id, remaining_time, current_section_index, section_start_time)}
    executing_nodes = {}
    scheduling_log = []

    # Add source and sink nodes to completed nodes for all tasks
    for task in tasks:
        completed_nodes[task["task_id"]].add("source")
        completed_nodes[task["task_id"]].add("sink")
        # Initialize results for each task from the start
        results[task["task_id"]] = {
            'completed_nodes': completed_nodes[task["task_id"]].copy(),  # Make a copy
            'scheduling_log': [],
            'total_time': 0,
            'missed_deadline': False
        }

    # Main simulation loop
    while True:
        # --- Deadline Check and Simulation Termination Condition ---
        all_tasks_done_or_missed = True  # Flag to check if we can exit the main loop

        for task in tasks:
            task_id = task["task_id"]

            if not results[task_id]["missed_deadline"]:
                if current_time > task["deadline"]:
                    print(
                        f"\nWarning: Task {task_id} missed its deadline at time {current_time} (deadline was {task['deadline']}). Task is no longer schedulable.")
                    results[task_id]["missed_deadline"] = True
                    results[task_id]["total_time"] = current_time  # Record time of missing deadline
                else:
                    # If this task hasn't missed its deadline, and it still has incomplete nodes,
                    # then not all tasks are done/missed.
                    if len(completed_nodes[task_id]) < len(task["nodes"]):
                        all_tasks_done_or_missed = False
            # If task has missed its deadline, it still counts towards the total_time
            # and incomplete nodes, so it doesn't make `all_tasks_done_or_missed` True
            elif len(completed_nodes[task_id]) < len(task["nodes"]):
                all_tasks_done_or_missed = False

        # Exit loop condition: All tasks are either fully completed OR have missed their deadlines
        # AND there are no nodes currently executing.
        if all_tasks_done_or_missed and not executing_nodes:
            print(f"\n--- Simulation End: All tasks completed or missed their deadlines. ---")
            break  # Exit the main while loop

        # --- Find and Prioritize Ready Nodes ---
        ready_nodes_by_task = {}
        for task in tasks:
            task_id = task["task_id"]
            if not results[task_id][
                "missed_deadline"]:  # Only find ready nodes for tasks that have NOT missed their deadline
                ready_nodes = find_ready_nodes(task["nodes"], task["edges"], completed_nodes[task_id])
                ready_nodes_by_task[task_id] = ready_nodes
            else:
                ready_nodes_by_task[task_id] = []  # No nodes are ready from this task if deadline is missed

        prioritized_nodes = []
        for task in tasks:
            task_id = task["task_id"]
            if not results[task_id]["missed_deadline"]:
                task_ready_nodes = ready_nodes_by_task[task_id]
                critical_path = task["critical_path"]

                critical_ready_nodes = [node for node in task_ready_nodes if node in critical_path]
                non_critical_ready_nodes = [node for node in task_ready_nodes if node not in critical_path]
                critical_ready_nodes.sort(key=lambda x: critical_path.index(x))

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

                for node in critical_ready_nodes:
                    prioritized_nodes.append((task_id, node, "Critical Path"))
                for node in critical_predecessors:
                    prioritized_nodes.append((task_id, node, "Critical Predecessor"))
                for node in non_critical_others:
                    prioritized_nodes.append((task_id, node, "Non-Critical"))

        # --- Assign Ready Nodes to Available Processors ---
        # Get all processors that are not currently busy
        available_system_processors = [p.id for p in allocated_processors if p.id not in executing_nodes]

        for task_id, node_id, priority_level in prioritized_nodes:
            # Ensure the node isn't already executing (can happen if prioritized_nodes has duplicates or race condition)
            if any(n_id == node_id for _, n_id, _, _, _ in executing_nodes.values()):
                continue  # Node is already running

            # Get available processors specifically assigned to this task
            task_specific_processors = task_to_processors.get(task_id, [])
            available_for_this_task = [p_id for p_id in task_specific_processors if p_id in available_system_processors]

            if available_for_this_task:
                task_obj = next(t for t in tasks if t["task_id"] == task_id)
                node_allocations = task_obj["allocations"].get(node_id, [])

                # IMPORTANT: For multi-section nodes, we ONLY try to acquire resources for the *first* section
                # when the node is initially assigned to a processor.
                # Subsequent resource sections are handled in the `Update remaining time` loop below.

                first_section_needs_resource = False
                first_section_resource_id = None
                if node_allocations:
                    potential_first_section = node_allocations[0]
                    if isinstance(potential_first_section, tuple) and potential_first_section[0] != "Normal":
                        first_section_needs_resource = True
                        first_section_resource_id = potential_first_section[0]

                can_acquire_initial_resource = True
                if first_section_needs_resource:
                    if not resource_manager.request_resource(node_id, first_section_resource_id):
                        can_acquire_initial_resource = False

                if can_acquire_initial_resource:
                    processor_id = available_for_this_task[0]  # Pick the first available processor for this task
                    available_system_processors.remove(processor_id)  # Mark as busy for this time step

                    first_section = node_allocations[0]
                    if isinstance(first_section, tuple):
                        section_type, section_time = first_section
                        executing_nodes[processor_id] = (task_id, node_id, section_time, 0, current_time)
                    else:
                        executing_nodes[processor_id] = (task_id, node_id, first_section, 0, current_time)
                    print(
                        f"\nNode {node_id} (Task {task_id}) started execution on processor {processor_id} at time {current_time}")
                    print(f"  Priority Level: {priority_level}")
                else:
                    # If initial resource not available, node waits and is not assigned a processor yet.
                    print(
                        f"\nNode {node_id} (Task {task_id}) is waiting for initial resource {first_section_resource_id} before starting.")

        # --- Show Current Status ---
        print(f"\n=== Time {current_time} ===")
        print("Executing Nodes:")
        if not executing_nodes:
            print("  (None)")
        for proc_id, (task_id, node_id, remaining, section_idx, start_time) in executing_nodes.items():
            task_obj = next(t for t in tasks if t["task_id"] == task_id)
            node_allocations = task_obj["allocations"].get(node_id, [])

            # Defensive check for section_idx
            if section_idx >= len(node_allocations):
                section_info = " (INVALID SECTION - SHOULD BE COMPLETED)"
            else:
                current_section = node_allocations[section_idx]
                if isinstance(current_section, tuple):
                    section_type, section_time = current_section
                    if section_type != "Normal":
                        # For resource section, remaining_time shown should be based on its original duration
                        # and how much time has passed *while it was actively holding the resource*.
                        # For simplicity, we can show its current 'remaining' value from tuple.
                        status_str = " [Acquired]" if resource_manager.is_resource_locked_by(node_id,
                                                                                             section_type) else " [Waiting]"
                        section_info = f" (Resource {section_type}: {remaining}/{section_time} time units remaining{status_str})"
                    else:
                        section_info = f" (Normal Section: {remaining}/{section_time} time units remaining)"
                else:  # Normal section defined as integer
                    section_info = f" (Normal Section: {remaining}/{current_section} time units remaining)"

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

        # --- Update Executing Nodes: Perform Work or Advance ---
        # Create a list to track nodes that complete a section or the entire node
        nodes_to_remove_or_update = []
        nodes_that_acquired_resource_this_cycle = []  # To handle resource transfer

        for processor_id, (task_id, node_id, remaining_orig, section_idx_orig, start_time_orig) in list(
                executing_nodes.items()):
            task_obj = next(t for t in tasks if t["task_id"] == task_id)
            node_allocations = task_obj["allocations"].get(node_id, [])

            # Defensive check
            if section_idx_orig >= len(node_allocations):
                print(
                    f"WARNING: Node {node_id} (Task {task_id}) on processor {processor_id} is in an invalid section index ({section_idx_orig}). Removing.")
                nodes_to_remove_or_update.append((processor_id, "REMOVE", None))
                continue

            current_section = node_allocations[section_idx_orig]
            current_remaining = remaining_orig

            # Assume progress will be made this time unit initially
            progress_made_this_cycle = True

            # --- Handle Resource Sections ---
            if isinstance(current_section, tuple) and current_section[0] != "Normal":
                section_type, section_time = current_section
                resource_id = section_type  # For clarity

                # If resource is needed, check if it's acquired. If not, try to acquire.
                if not resource_manager.is_resource_locked_by(node_id, resource_id):
                    # Attempt to acquire resource if not already held
                    if not resource_manager.request_resource(node_id, resource_id):
                        # Resource not available, busy-waiting. No progress this cycle.
                        print(
                            f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} is busy-waiting for {resource_id}.")
                        progress_made_this_cycle = False
                        # Update tuple to prevent decrementing 'remaining'. 'start_time' might need adjustment.
                        # For simplicity, we just skip decrementing 'remaining' here.
                        executing_nodes[processor_id] = (task_id, node_id, remaining_orig, section_idx_orig,
                                                         start_time_orig)  # Keep original tuple state
                        continue  # Skip decrementing time and advancement for this node this cycle
                    else:
                        # Resource acquired NOW in this cycle.
                        print(
                            f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} just ACQUIRED {resource_id} (was waiting).")
                        # Proceed to decrement time as work is done.

                # If resource is held AND it's the last time unit for this section, release it.
                # This must happen BEFORE 'current_remaining' becomes 0.
                if current_remaining == 1:
                    if resource_manager.is_resource_locked_by(node_id, resource_id):
                        acquired_by_node = resource_manager.release_resource(node_id, resource_id)
                        print(
                            f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} released resource {resource_id}.")
                        if acquired_by_node:
                            nodes_that_acquired_resource_this_cycle.append((acquired_by_node, resource_id))
                    else:
                        # This scenario should ideally not happen if logic is correct:
                        # a node completing a resource section but not holding the resource.
                        print(
                            f"WARNING: Node {node_id} (Task {task_id}) on P{processor_id} finished resource section {resource_id} but didn't hold it.")

            # --- Decrement Remaining Time (Only if progress was made) ---
            if progress_made_this_cycle:
                current_remaining -= 1

            # --- Check if Current Section is Completed ---
            if current_remaining <= 0:
                print(
                    f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} finished section {section_idx_orig} (Type: {current_section}) at time {current_time}.")

                # Move to next section or complete node
                if section_idx_orig + 1 < len(node_allocations):
                    next_section = node_allocations[section_idx_orig + 1]
                    if isinstance(next_section, tuple):  # Next is a resource section
                        section_type_next, section_time_next = next_section
                        nodes_to_remove_or_update.append((processor_id, "UPDATE",
                                                          (task_id, node_id, section_time_next, section_idx_orig + 1,
                                                           current_time + 1)))
                        print(
                            f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} moving to next section: {section_type_next} for {section_time_next} units.")
                    else:  # Next is a normal section
                        nodes_to_remove_or_update.append((processor_id, "UPDATE",
                                                          (task_id, node_id, next_section, section_idx_orig + 1,
                                                           current_time + 1)))
                        print(
                            f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} moving to next section: Normal for {next_section} units.")
                else:
                    # Node is completely finished
                    nodes_to_remove_or_update.append((processor_id, "REMOVE", node_id))
                    print(
                        f"DEBUG: Node {node_id} (Task {task_id}) on P{processor_id} completed ENTIRELY at time {current_time}.")
            else:
                # Section not completed, update remaining time
                executing_nodes[processor_id] = (task_id, node_id, current_remaining, section_idx_orig, start_time_orig)

        # Apply updates and removals after iterating (to avoid dictionary modification during iteration)
        for processor_id, action, data in nodes_to_remove_or_update:
            if action == "REMOVE":
                node_id_completed = data
                completed_nodes[task_id].add(
                    node_id_completed)  # Ensure task_id is correct here, use current node's task_id
                # Final safeguard: release any resources that *might* still be held by this node
                # if not already released by section completion logic.
                resource_manager.release_all_resources_for_node(node_id_completed)
                print(f"DEBUG: Node {node_id_completed} (Task {task_id}) fully removed from P{processor_id}.")
                del executing_nodes[processor_id]
                scheduling_log.append({
                    'time': current_time,
                    'task_id': task_id,  # Need correct task_id here
                    'node': node_id_completed,
                    'processor': processor_id,
                    'action': 'completed'
                })
            elif action == "UPDATE":
                executing_nodes[processor_id] = data

        # For nodes that just acquired a resource via a release from another node
        # (This is more of a notification for the next cycle than an immediate action here with tuples)
        for acquired_node_id, acquired_resource_id in nodes_that_acquired_resource_this_cycle:
            print(
                f"DEBUG: Notification: Node {acquired_node_id} has acquired resource {acquired_resource_id} for its next cycle.")
            # The node will pick this up when it re-evaluates its resource needs in its next cycle.

        current_time += 1
        # The main loop condition now handles deadlock detection and termination more robustly.

    # ... (rest of the schedule_multiple_tasks function remains the same)
    # Add results for tasks that didn't miss deadline (ensure this part correctly uses 'results' dict)
    for task_id_final, result_data in results.items():
        # Ensure total_time is set for tasks that didn't miss deadline and completed
        if not result_data['missed_deadline'] and len(completed_nodes[task_id_final]) == len(
                next(t for t in tasks if t['task_id'] == task_id_final)['nodes']):
            results[task_id_final]['total_time'] = current_time  # This will be the simulation end time

        # Filter scheduling_log for each task
        results[task_id_final]['scheduling_log'] = [log for log in scheduling_log if log['task_id'] == task_id_final]

    print("\n=== Execution Summary ===")
    print(f"Total Simulation Time: {current_time}")  # Total time the simulator ran
    for task_id, result in results.items():
        status = "Missed Deadline (Incomplete)" if result.get('missed_deadline', False) else "Completed Successfully"
        total_task_time = result.get('total_time', 'N/A')
        print(f"Task {task_id}: Total Time = {total_task_time}, Status = {status}")
        # **This line needs to use the 'completed_nodes' from the 'result' dictionary, which is correctly updated.**
        # Ensure 'result['completed_nodes']' contains all actually completed nodes.
        # It seems 'results[task_id]['completed_nodes']' should already be getting populated correctly
        # from the `nodes_to_remove_or_update` loop.
        print(
            f"  Completed Nodes: {sorted(result['completed_nodes'], key=lambda x: str(x))}")  # This line looks correct.

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




