from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import random as rand
import random
import math
import networkx as nx
import numpy as np
from math import gcd
from functools import reduce
from queue import Queue
import heapq


# --- Resource Management Classes ---
@dataclass
class ResourceLock:
    is_locked: bool = False
    locked_by: Optional[str] = None
    waiting_queue: Queue[str] = field(default_factory=Queue)
    # No custom __init__ needed when using default_factory with dataclass


class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, ResourceLock] = {}

    def initialize_resources(self, resource_ids: List[str]):
        for resource_id in resource_ids:
            self.resources[resource_id] = ResourceLock()

    def request_resource(self, node_id: str, resource_id: str) -> bool:
        if resource_id not in self.resources:
            # print(f"WARNING: Resource {resource_id} requested by Node {node_id} does not exist.")
            return True  # Or raise an error if this is an unexpected state

        lock = self.resources[resource_id]

        if not lock.is_locked:
            lock.is_locked = True
            lock.locked_by = node_id
            return True
        elif lock.locked_by == node_id:  # Node already holds the resource
            return True
        else:  # Resource is locked by *another* node
            if node_id not in [item for item in list(lock.waiting_queue.queue)]:  # Avoid duplicates
                lock.waiting_queue.put(node_id)
            return False

    def release_resource(self, node_id: str, resource_id: str) -> Optional[str]:
        if resource_id not in self.resources:
            return None

        lock = self.resources[resource_id]
        if lock.locked_by == node_id:
            lock.is_locked = False
            lock.locked_by = None

            if not lock.waiting_queue.empty():
                next_node = lock.waiting_queue.get()
                lock.is_locked = True
                lock.locked_by = next_node
                return next_node
        return None

    def get_waiting_nodes(self, resource_id: str) -> List[str]:
        if resource_id not in self.resources:
            return []
        return list(self.resources[resource_id].waiting_queue.queue)

    def is_resource_locked_by(self, node_id: str, resource_id: str) -> bool:
        if resource_id not in self.resources:
            return False
        return self.resources[resource_id].locked_by == node_id

    def release_all_resources_for_node(self, node_id: str):
        for resource_id, lock in self.resources.items():
            if lock.locked_by == node_id:
                self.release_resource(node_id, resource_id)


@dataclass
class Resource:
    id: str

    def map_to_color(self) -> str:
        if self.id == "R1": return 'red'
        if self.id == "R2": return 'blue'
        if self.id == "R3": return 'green'
        if self.id == "R4": return 'yellow'
        if self.id == "R5": return 'purple'
        if self.id == "R6": return 'pink'
        return 'black'


# --- Task and Node Definitions (Streamlined) ---
@dataclass
class Node:
    id: str
    wcet: int  # Original WCET from graph generation

    # Other fields (critical_st, critical_en, resources) not directly used in phase-based scheduling

    def __str__(self) -> str:
        return f"Node: {self.id}, WCET: {self.wcet}"


@dataclass
class Edge:
    src: str
    sink: str


@dataclass
class BaseTask:  # Represents the template for periodic tasks
    id: int
    period: int
    wcet: int  # Sum of node WCETs
    nodes: list  # List of node IDs
    edges: list  # List of tuples (src_id, sink_id)
    deadline: int  # Relative deadline (usually equals period for implicit deadlines)
    critical_path: list
    critical_path_length: int
    allocations: Dict[str, list]  # Detailed allocations for each node (phases)
    execution_times: Dict[str, int]  # Map of node_id to its original WCET

    def get_total_wcet(self) -> int:
        return sum(self.execution_times.get(node, 0) for node in self.nodes if node not in ["source", "sink"])

    def utilization(self) -> float:
        total_node_time = sum(sum(section[1] if isinstance(section, tuple) else section for section in sections)
                              for sections in self.allocations.values())
        # If total_node_time is more accurate after allocation, use it, else use get_total_wcet()
        return total_node_time / self.period if self.period > 0 else 0.0

    def __str__(self) -> str:
        return f"Task T{self.id} (Period: {self.period}, WCET: {self.get_total_wcet()}, Deadline: {self.deadline})"


@dataclass
class TaskInstance:  # Represents a single job (instance) of a periodic task
    task_id: int
    instance_num: int  # e.g., 0 for first instance, 1 for second
    instance_id: str  # e.g., "T1-J0"
    release_time: int
    absolute_deadline: int

    nodes: list  # List of node IDs for this instance
    edges: list  # Edges for this instance's graph
    execution_times: Dict[str, int]  # Original WCETs for nodes in this instance
    allocations: Dict[str, list]  # Allocation phases for nodes in this instance
    critical_path: list  # Critical path for this instance
    period: int

    # State specific to this instance's execution
    completed_nodes: set = field(default_factory=set)
    is_completed: bool = False
    deadline_missed: bool = False

    def __post_init__(self):
        self.completed_nodes.add("source")
        self.completed_nodes.add("sink")

    def __lt__(self, other):
        # For EDF (Earliest Deadline First) priority queue: smaller deadline is higher priority
        return self.absolute_deadline < other.absolute_deadline

    def __str__(self) -> str:
        return (f"Instance {self.instance_id} (Task T{self.task_id}) "
                f"Release: {self.release_time}, AbsDL: {self.absolute_deadline}")


# --- Graph Generation and Task Properties ---
def erdos_renyi_graph():
    num_nodes_to_generate = random.randint(5, 20)  # Reduced max nodes for easier debugging
    edge_probability = 0.1  # Increased edge probability for denser graphs

    G = nx.erdos_renyi_graph(num_nodes_to_generate, edge_probability, directed=True)
    # Ensure it's a DAG and relabel nodes
    G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    mapping = {node: f"N{node + 1}" for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    source_node = "source"
    sink_node = "sink"
    G.add_node(source_node)
    G.add_node(sink_node)

    for node in list(G.nodes()):  # Use list(G.nodes()) to allow modification during iteration
        if G.in_degree(node) == 0 and node != source_node:
            G.add_edge(source_node, node)
        if G.out_degree(node) == 0 and node != sink_node and node != source_node:
            G.add_edge(node, sink_node)

    return G


def visualize_task(G):
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", node_size=700, font_size=10,
    #         edge_color="gray")
    # plt.title(f"DAG", fontsize=14)
    # plt.show()
    pass  # Placeholder if matplotlib not installed or not desired


def get_critical_path(nodes: list, edges: list[tuple], execution_times: dict) -> tuple[list, int]:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    source = "source"
    sink = "sink"

    if not G.has_node(source) or not G.has_node(sink):
        return [], 0

    try:
        all_paths = list(nx.all_simple_paths(G, source=source, target=sink))
    except nx.NetworkXNoPath:
        return [], 0
    except nx.NodeNotFound:
        return [], 0

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


def generate_resources(resource_count: int) -> list[Resource]:
    resources = [Resource(id=f"R{i + 1}") for i in range(resource_count)]
    return resources


def generate_accesses_and_lengths(num_tasks: int, num_resources: int = 6) -> tuple[dict, dict]:
    accesses = {f"R{q + 1}": [0] * num_tasks for q in range(num_resources)}
    lengths = {f"R{q + 1}": [[] for _ in range(num_tasks)] for q in range(num_resources)}

    for q in range(num_resources):
        max_accesses_for_resource = random.randint(1, 16)
        max_length_for_resource = random.randint(5, 100)

        remaining_total_accesses = max_accesses_for_resource
        for i in range(num_tasks):
            if remaining_total_accesses > 0:
                task_access_count = random.randint(0, remaining_total_accesses)
                accesses[f"R{q + 1}"][i] = task_access_count
                remaining_total_accesses -= task_access_count

                if task_access_count > 0:
                    lengths[f"R{q + 1}"][i] = [random.randint(1, max_length_for_resource)
                                               for _ in range(task_access_count)]
            else:
                accesses[f"R{q + 1}"][i] = 0
                lengths[f"R{q + 1}"][i] = []

    return accesses, lengths


def allocate_resources_to_nodes(task_info: dict, task_id: int, accesses: dict, lengths: dict) -> tuple[dict, dict]:
    nodes = [node for node in task_info["nodes"] if node != "source" and node != "sink"]

    allocations = {node_id: [] for node_id in nodes}
    execution_times_allocated = task_info["execution_times"].copy()  # This will be the updated execution times

    for node_id in nodes:
        original_wcet = task_info["execution_times"][node_id]  # Use original WCET from task_info

        # Collect all resource sections for this specific node and task
        node_resource_sections = []
        for resource, task_access_counts in accesses.items():
            if task_id - 1 < len(task_access_counts) and task_access_counts[task_id - 1] > 0:
                # Iterate through individual access lengths for this (resource, task) pair
                # You might need a more sophisticated mapping if specific nodes need specific resource instances
                # For now, distributing available resource accesses from `lengths` randomly across nodes.
                node_access_lengths_for_resource = lengths[resource][task_id - 1]

                # Assign a random number of these accesses to the current node
                num_accesses_for_node = random.randint(0, len(node_access_lengths_for_resource))

                # Take these accesses and remove them from the general pool for this task
                for _ in range(num_accesses_for_node):
                    if node_access_lengths_for_resource:
                        access_length = node_access_lengths_for_resource.pop(0)
                        node_resource_sections.append((resource, access_length))

        # Calculate total resource time assigned to this node
        total_resource_time_for_node = sum(sec[1] for sec in node_resource_sections)

        # Distribute remaining normal execution time (original_wcet - total_resource_time_for_node)
        # among segments. If original_wcet is less than total_resource_time, this implies critical section
        # duration adds to total WCET, not just splitting existing WCET.

        # For simplicity, let's assume resource sections ADD to WCET if they are more than original_wcet,
        # otherwise they might consume part of it.

        remaining_normal_time = max(0, original_wcet - total_resource_time_for_node)

        # Create final allocation for this node
        final_allocation_for_node = []

        # Interleave normal and resource sections
        all_sections_to_interleave = [(section, 'resource') for section in node_resource_sections]

        # Add a placeholder for normal sections, then distribute remaining_normal_time
        num_segments = len(node_resource_sections) + 1  # num_resource_sections + 1 normal sections at most

        normal_segment_lengths = [0] * num_segments
        if remaining_normal_time > 0:
            # Distribute remaining_normal_time among the normal segments
            # A simple way: randomly assign pieces.
            points = sorted(random.sample(range(remaining_normal_time + num_segments - 1), num_segments - 1))
            points = [0] + points + [remaining_normal_time + num_segments - 1]
            normal_segment_lengths = [points[i + 1] - points[i] for i in range(num_segments)]

            # Clean up the `normal_segment_lengths` for better distribution (avoiding very small ones)
            # This part can be complex, for now let's use the random distribution directly.

        for i in range(num_segments):
            if normal_segment_lengths[i] > 0:
                final_allocation_for_node.append(("Normal", normal_segment_lengths[i]))
            if i < len(node_resource_sections):
                final_allocation_for_node.append(node_resource_sections[i])  # Add resource section

        # Filter out zero-duration normal sections if they appear consecutively or at ends
        filtered_allocation = []
        for sec in final_allocation_for_node:
            if isinstance(sec, tuple) and sec[0] == "Normal" and sec[1] == 0:
                continue  # Skip normal sections with zero duration
            filtered_allocation.append(sec)

        # If a node ends up with no normal or resource sections, give it a minimal normal section.
        if not filtered_allocation:
            filtered_allocation.append(("Normal", 1))  # Default to 1 time unit if empty

        allocations[node_id] = filtered_allocation

        # Update overall execution_times based on allocated phases (total allocated time)
        # This is important as resource sections might increase effective WCET
        total_allocated_time_for_node = sum(sec[1] if isinstance(sec, tuple) else sec for sec in filtered_allocation)
        execution_times_allocated[node_id] = total_allocated_time_for_node

    return allocations, execution_times_allocated


def generate_base_task(task_id: int, accesses_data: dict, lengths_data: dict) -> Optional[BaseTask]:
    """
    Generates a base task template with random DAG, execution times, and resource allocations.
    """
    G = erdos_renyi_graph()
    nodes = list(G.nodes())
    edges = list(G.edges())

    # Filter out source/sink for WCET assignment
    actual_nodes = [n for n in nodes if n not in ["source", "sink"]]
    if not actual_nodes:  # If only source/sink, skip task generation
        # print(f"Skipping Task {task_id}: No functional nodes other than source and sink.")
        return None

    # Assign base execution times to actual nodes
    execution_times_initial = {node: random.randint(10, 20) for node in actual_nodes}  # Smaller range for periods

    # Allocate resources to nodes, which also modifies effective execution_times
    allocations, execution_times_final = allocate_resources_to_nodes(
        {"nodes": nodes, "edges": edges, "execution_times": execution_times_initial},
        task_id,
        accesses_data,
        lengths_data
    )

    # Calculate properties based on FINAL execution times after resource allocation
    critical_path, critical_path_length = get_critical_path(nodes, edges, execution_times_final)

    # If critical path is zero due to graph issues, assign a minimal length
    if critical_path_length == 0 and len(actual_nodes) > 0:
        critical_path_length = sum(execution_times_final.values()) / len(actual_nodes)  # Estimate for period calc
        if critical_path_length == 0: critical_path_length = 10  # Minimum to avoid div by zero

    total_execution_time = sum(execution_times_final.values())  # Sum of all allocated section times

    # Adjust period calculation for tighter deadlines for schedulability testing
    # Period should be related to critical path, but not too loose.
    critical_pathth = critical_path_length
    x = 1
    while x < critical_pathth:
        x *= 2

    period = rand.choice([x, 2 * x])
   # period = int(critical_path_length * rand.uniform(1.2, 2.5))  # Adjust factor for schedulability
    if period == 0: period = 1  # Minimum period

    deadline = period  # Implicit deadline

    base_task = BaseTask(
        id=task_id,
        period=period,
        wcet=total_execution_time,  # Total WCET including resource parts
        nodes=nodes,
        edges=edges,
        deadline=deadline,
        critical_path=critical_path,
        critical_path_length=critical_path_length,
        allocations=allocations,
        execution_times=execution_times_final  # Final WCETs after allocation
    )

    print(f"\nGenerated Base Task T{task_id}: {base_task}")
    print(f"  Nodes: {base_task.nodes}")
    print(f"  Edges: {base_task.edges}")
    print(f"  Execution Times (after allocation): {base_task.execution_times}")
    print(f"  Critical Path: {base_task.critical_path}")
    print(f"  Critical Path Length: {base_task.critical_path_length}")
    print(f"  Total Allocated Time (WCET): {base_task.wcet}")
    print(f"  Utilization: {base_task.utilization():.2f}")
    print(f"  Period/Deadline: {base_task.period}")

    return base_task


def generate_tasks(num_base_tasks: int, num_resources: int) -> List[BaseTask]:
    tasks_list = []
    # Generate global access/length patterns for all base tasks
    accesses_data, lengths_data = generate_accesses_and_lengths(num_base_tasks, num_resources)

    for i in range(1, num_base_tasks + 1):
        task = generate_base_task(task_id=i, accesses_data=accesses_data, lengths_data=lengths_data)
        if task is not None:
            tasks_list.append(task)
    return tasks_list


# --- Periodicity and Hyperperiod Calculation ---
def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def hyperperiod_lcm(tasks: List[BaseTask]) -> int:
    if not tasks:
        return 1
    periods = [task.period for task in tasks]
    if not periods:
        return 1
    return reduce(lcm, periods)


def generate_periodic_task_instances(base_tasks: List[BaseTask], hyper_period: int) -> List[TaskInstance]:
    all_instances: List[TaskInstance] = []
    for task in base_tasks:
        num_instances = hyper_period // task.period
        # Add an extra instance if hyperperiod is an exact multiple, to ensure the end point is covered
        if hyper_period % task.period == 0:
            num_instances -= 1  # The last instance's release time would be exactly hyperperiod

        for i in range(num_instances + 1):  # +1 to include instance at time 0 and up to HP - period
            release_t = task.period * i
            abs_deadline = release_t + task.deadline  # Relative deadline == Period

            # Create a deep copy of nodes/edges/allocations to avoid shared state issues between instances
            # Although your existing code copies when assigning, this explicit copy is safer
            instance_nodes = list(task.nodes)
            instance_edges = list(task.edges)
            instance_execution_times = task.execution_times.copy()
            instance_allocations = {k: list(v) for k, v in task.allocations.items()}  # Copy lists of sections too

            instance = TaskInstance(
                task_id=task.id,
                instance_num=i,
                instance_id=f"T{task.id}-J{i}",
                release_time=release_t,
                absolute_deadline=abs_deadline,
                nodes=instance_nodes,
                edges=instance_edges,
                execution_times=instance_execution_times,
                allocations=instance_allocations,
                critical_path=list(task.critical_path),  # Copy critical path
                period=task.period
            )
            all_instances.append(instance)

    # Sort all instances by their release time for easier processing
    all_instances.sort(key=lambda inst: inst.release_time)
    print(f"\nGenerated {len(all_instances)} total task instances over hyperperiod {hyper_period}.")
    return all_instances


# --- Processor and Scheduling Utilities ---
@dataclass
class Processor:
    id: int
    assigned_tasks: List[int]  # Base task IDs
    utilization: float = 0.0


def calculate_total_processors(tasks: List[BaseTask]):
    # This calculation is for initial allocation, based on base task properties
    sum_utilization = sum(t.utilization() for t in tasks)

    # Define U_norm as per your new requirement (a value between 0.1 and 1)
    # If U_norm is a fixed system parameter, pass it as an argument.
    # If it's a random value per simulation, generate it here.
    # Based on your previous code snippet: U_norm = rand.uniform(0.1, 1)
    U_norm = rand.uniform(0.1, 1)  # Example: Randomly choose a normalized utilization factor

    # Apply the new formula: sum_utilization / U_norm, then ceiling
    m_total = math.ceil(sum_utilization / U_norm)

    if m_total == 0:
        m_total = 1  # Ensure at least one processor even if utilization is extremely low

    print(
        f"DEBUG: Sum Utilization: {sum_utilization:.2f}, U_norm: {U_norm:.2f}, Calculated Total Processors (m_total): {m_total}")
    return m_total


def calculate_asap_cores(nodes: List[str], edges: List[Tuple[str, str]], execution_times: Dict[str, int]) -> Tuple[
    Dict[str, int], int]:
    # This uses execution_times which are the WCETs of the original nodes.
    # It does not directly account for resource critical sections increasing length.
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    asap_schedule = {}

    for node in nx.topological_sort(G):
        if node == "source":
            asap_schedule[node] = 0
        else:
            # max end time of predecessors
            pred_end_times = [asap_schedule[pred] + execution_times.get(pred, 0) for pred in G.predecessors(node)]
            asap_schedule[node] = max(pred_end_times) if pred_end_times else 0

    max_parallel_tasks = 0
    # To calculate max_parallel_tasks, we'd need a full discrete event simulation or a more complex ASAP calculation
    # For now, a rough estimate or simply using total_processors might be more practical for federated.
    # A more accurate ASAP for parallelism:
    active_nodes_count = {}
    sorted_nodes = list(nx.topological_sort(G))

    # Initialize all nodes to be "not active" before source time
    current_active_nodes = set()

    # Simulate time step by time step based on ASAP start/finish times
    all_event_times = sorted(list(set(asap_schedule.values()) | set(
        asap_schedule[n] + execution_times.get(n, 0) for n in nodes if n not in ["source", "sink"])))

    for t_step in all_event_times:
        nodes_starting_at_t = {node for node, start_t in asap_schedule.items() if
                               start_t == t_step and node not in ["source", "sink"]}
        nodes_finishing_at_t = {node for node, end_t in asap_schedule.items() if
                                (asap_schedule[node] + execution_times.get(node, 0)) == t_step and node not in [
                                    "source", "sink"]}

        current_active_nodes.update(nodes_starting_at_t)
        current_active_nodes.difference_update(nodes_finishing_at_t)

        max_parallel_tasks = max(max_parallel_tasks, len(current_active_nodes))

    return asap_schedule, max_parallel_tasks


def federated_scheduling(base_tasks: List[BaseTask]) -> List[Processor]:
    total_processors = calculate_total_processors(base_tasks)
    processors = [Processor(id=i + 1, assigned_tasks=[]) for i in range(total_processors)]

    print(f"\n--- Federated Scheduling Allocation ---")
    print(f"Total Processors to allocate: {total_processors}")

    # Step 1: Assign processors to U_i > 1 tasks (fully utilize m_i processors)
    remaining_processors = total_processors
    high_util_tasks = []
    low_util_tasks = []

    for task in base_tasks:
        if task.utilization() > 1.0:  # Using task.utilization() which is WCET/Period
            high_util_tasks.append(task)
        else:
            low_util_tasks.append(task)

    # Sort high_util_tasks by utilization descending
    high_util_tasks.sort(key=lambda t: t.utilization(), reverse=True)

    for task in high_util_tasks:
        _, max_parallel_cores_for_task = calculate_asap_cores(task.nodes, task.edges, task.execution_times)

        # Ensure max_parallel_cores_for_task is at least 1, if task has real nodes
        if max_parallel_cores_for_task == 0 and len([n for n in task.nodes if n not in ["source", "sink"]]) > 0:
            max_parallel_cores_for_task = 1  # Needs at least one processor

        if remaining_processors >= max_parallel_cores_for_task:
            print(
                f"Assigning {max_parallel_cores_for_task} processors to Task T{task.id} (U_i={task.utilization():.2f} > 1)")
            # Find available processors and assign
            assigned_count = 0
            for p in processors:
                if not p.assigned_tasks:  # If processor is unassigned
                    p.assigned_tasks.append(task.id)
                    p.utilization += task.utilization() / max_parallel_cores_for_task  # Distribute utilization
                    assigned_count += 1
                    if assigned_count == max_parallel_cores_for_task:
                        break
            remaining_processors -= max_parallel_cores_for_task
        else:
            print(
                f"WARNING: Not enough processors for Task T{task.id} (U_i={task.utilization():.2f}). Needs {max_parallel_cores_for_task}, but only {remaining_processors} left.")
            # This task might not be schedulable or will contend for processors in lower utilization pool.
            low_util_tasks.append(task)  # Add to low util tasks to try WFD

    # Step 2: Assign remaining processors to U_i <= 1 tasks using Worst Fit Decreasing (WFD)
    # Sort low_util_tasks by utilization descending
    low_util_tasks.sort(key=lambda t: t.utilization(), reverse=True)

    for task in low_util_tasks:
        # Sort processors by remaining capacity (largest capacity first)
        # Processors assigned to high_util_tasks are now "busy" or dedicated.
        # We only consider processors that have remaining capacity or are completely free.
        available_processors_for_wfd = sorted([p for p in processors if p.utilization <= 1.0],
                                              key=lambda p: 1.0 - p.utilization,
                                              reverse=True)  # Sort by remaining capacity

        assigned = False
        for processor in available_processors_for_wfd:
            if processor.utilization + task.utilization() <= 1.0:
                processor.assigned_tasks.append(task.id)
                processor.utilization += task.utilization()
                assigned = True
                print(
                    f"Assigning Task T{task.id} (U_i={task.utilization():.2f}) to Processor {processor.id} (New U: {processor.utilization:.2f})")
                break

        if not assigned:
            print(
                f"WARNING: Task T{task.id} (U_i={task.utilization():.2f}) could not be assigned to any processor in WFD. It might miss its deadline.")
            # Mark this task as potentially unschedulable if it's crucial.

    print("\n--- Final Processor Allocation Summary ---")
    for p in processors:
        print(f"Processor {p.id}: Assigned Tasks {p.assigned_tasks}, Utilization: {p.utilization:.2f}")

    return processors


# --- EDF Scheduler Core ---

def find_ready_nodes(nodes: list, edges: list[tuple], completed_nodes: set) -> list:
    """
    Finds nodes that are ready for execution (all predecessors are completed).
    Assumes source node is initially in completed_nodes.
    """
    ready_nodes = []

    for node in nodes:
        if node == "sink" or node in completed_nodes:
            continue

        predecessors_of_node = [src for src, dest in edges if dest == node]

        if not predecessors_of_node or all(pred in completed_nodes for pred in predecessors_of_node):
            ready_nodes.append(node)

    return ready_nodes


def schedule_multiple_tasks_periodic(base_tasks: List[BaseTask], system_processors: List[Processor]) -> Dict[int, Dict]:
    """
    Schedules multiple periodic tasks over a hyperperiod using EDF-like priority.
    Determines overall schedulability.
    """
    print("\n=== Starting Periodic Task Scheduling ===")

    # Calculate Hyperperiod
    hyper_period = hyperperiod_lcm(base_tasks)
    print(f"Calculated Hyperperiod: {hyper_period}")

    # Generate all task instances (jobs) for the hyperperiod
    all_instances: List[TaskInstance] = generate_periodic_task_instances(base_tasks, hyper_period)

    # Initialize System-wide Resource Manager
    resource_manager = ResourceManager()
    resource_manager.initialize_resources([f"R{i + 1}" for i in range(6)])

    # Scheduling state variables
    current_time = 0
    # active_jobs_pq: Jobs that have been released and are not yet completed
    # Stored as (absolute_deadline, release_time, instance_id, TaskInstance object) for priority queue
    active_jobs_pq: List[Tuple[int, int, str, TaskInstance]] = []
    heapq.heapify(active_jobs_pq)  # Min-heap based on absolute_deadline

    # executing_nodes: {processor_id: (instance_id, node_id, remaining_time, current_section_index, section_start_time, job_obj)}
    executing_nodes: Dict[int, Tuple[str, str, int, int, int, TaskInstance]] = {}

    # Map job_id to its TaskInstance object for quick lookup
    job_map: Dict[str, TaskInstance] = {instance.instance_id: instance for instance in all_instances}

    # Keep track of the results for each job
    job_results: Dict[str, Dict] = {inst.instance_id: {'completed': False, 'deadline_missed': False, 'total_time': 0}
                                    for inst in all_instances}

    overall_schedulable = True  # Flag for final schedulability determination
    scheduling_log = []  # Initialize scheduling log

    # --- Main Simulation Loop (over Hyperperiod) ---
    while True:  # Keep as True for now, exit conditions inside
        print(f"\n--- Time {current_time} --- (Hyperperiod: {hyper_period})")

        # 1. Release new job instances
        jobs_released_this_cycle = False
        for instance in all_instances:
            if instance.release_time == current_time and not instance.is_completed and not instance.deadline_missed:
                if not any(job_tuple[2] == instance.instance_id for job_tuple in active_jobs_pq):
                    heapq.heappush(active_jobs_pq,
                                   (instance.absolute_deadline, instance.release_time, instance.instance_id, instance))
                    print(
                        f"INFO: {instance.instance_id} released and added to active_jobs_pq at time {current_time}. AbsDL: {instance.absolute_deadline}")
                    jobs_released_this_cycle = True
                else:
                    print(f"DEBUG: {instance.instance_id} already in active_jobs_pq, skipping re-add.")

        if jobs_released_this_cycle:
            print(f"DEBUG: Active jobs PQ size after release: {len(active_jobs_pq)}")
            # For inspection: print some active jobs
            # for i, (dl, _, inst_id, _) in enumerate(active_jobs_pq[:5]):
            #     print(f"  PQ Top {i}: {inst_id} (DL: {dl})")

        # 2. Check for missed deadlines among active and executing jobs
        jobs_to_remove_from_active_pq = []
        for deadline, release, inst_id, job_obj in list(active_jobs_pq):  # Iterate over a copy
            if current_time > job_obj.absolute_deadline and not job_obj.is_completed and not job_obj.deadline_missed:
                job_obj.deadline_missed = True
                job_obj.is_schedulable = False  # Mark the instance as unschedulable
                job_obj.is_completed = True  # Consider it 'completed' for processing, but failed
                overall_schedulable = False
                job_results[inst_id]['deadline_missed'] = True
                job_results[inst_id]['total_time'] = current_time
                print(
                    f"WARNING: {job_obj.instance_id} MISSED DEADLINE at time {current_time} (was {job_obj.absolute_deadline}). Job status updated.")

                # If this job was executing, remove it and release resources
                for proc_id, exec_data in list(executing_nodes.items()):
                    if exec_data[0] == inst_id:  # exec_data[0] is instance_id
                        print(f"DEBUG: Removing {inst_id} (node {exec_data[1]}) from P{proc_id} due to deadline miss.")
                        del executing_nodes[proc_id]
                        # Release any resources held by this node
                        # Note: exec_data[1] is the node_id string for ResourceManager
                        resource_manager.release_all_resources_for_node(exec_data[1])
                jobs_to_remove_from_active_pq.append((deadline, release, inst_id, job_obj))

        if jobs_to_remove_from_active_pq:
            active_jobs_pq = [(d, r, i, j) for d, r, i, j in active_jobs_pq if not j.deadline_missed]
            heapq.heapify(active_jobs_pq)

        # --- Loop Termination Condition ---
        all_jobs_processed = True
        for job_obj in all_instances:
            if not job_obj.is_completed and not job_obj.deadline_missed:
                all_jobs_processed = False
                break

        # Terminate if all jobs are processed OR if we've passed hyperperiod and no jobs are executing.
        if all_jobs_processed and not executing_nodes and current_time >= hyper_period:
            print(f"\n--- Simulation End Condition Met: All jobs processed or hyperperiod reached. ---")
            break

        # Fail-safe break: if no work can be done, and no new jobs will arrive within hyperperiod.
        # This prevents infinite loops if scheduling stalls before hyperperiod end.
        if not executing_nodes and not active_jobs_pq and current_time >= hyper_period:
            print(f"WARNING: No executing nodes and no active jobs in PQ. Hyperperiod reached. Terminating.")
            break

        # 3. Select jobs/nodes to execute for this time step (EDF & CPC)
        available_processors = [p.id for p in system_processors if p.id not in executing_nodes]
        candidates_for_scheduling = []  # List of (job_priority_value, node_priority_value, job_obj, node_id, processor_id)

        nodes_currently_running_or_selected = set()
        for _, node_id_exec, _, _, _, _ in executing_nodes.values():
            nodes_currently_running_or_selected.add(node_id_exec)

            # Iterate through active_jobs_pq (prioritized by deadline/EDF)
        for _, _, _, job_obj in active_jobs_pq:
            if job_obj.is_completed or job_obj.deadline_missed:
                continue

            # Find processors assigned to this job's base task ID
            # This is crucial for federated scheduling to ensure jobs only run on their allocated cores.
            processors_assigned_to_this_job_base_task = []
            for p in system_processors:
                if job_obj.task_id in p.assigned_tasks:
                    processors_assigned_to_this_job_base_task.append(p.id)

            # Only consider processors that are both available system-wide AND assigned to this task's base ID
            suitable_and_available_processors_for_this_job = [p_id for p_id in available_processors if
                                                              p_id in processors_assigned_to_this_job_base_task]

            if not suitable_and_available_processors_for_this_job:
                continue  # No suitable and available processor for this job right now

            # Find ready nodes for this job instance
            ready_nodes_for_job = find_ready_nodes(job_obj.nodes, job_obj.edges, job_obj.completed_nodes)
            actual_ready_nodes = [n for n in ready_nodes_for_job if n not in ["source", "sink"]]

            if not actual_ready_nodes:
                continue

                # --- Apply Critical Path Scheduling (CPC) for node selection within this job ---
            critical_path_for_job = job_obj.critical_path
            edges_for_job = job_obj.edges

            critical_ready_nodes = [node for node in actual_ready_nodes if node in critical_path_for_job]
            non_critical_ready_nodes = [node for node in actual_ready_nodes if node not in critical_path_for_job]

            # Sort critical nodes by position on CP
            critical_ready_nodes.sort(key=lambda x: critical_path_for_job.index(x))

            critical_predecessors = []
            non_critical_others = []

            for node in non_critical_ready_nodes:
                is_predecessor = False
                for critical_node in critical_path_for_job:
                    if (node, critical_node) in edges_for_job:
                        is_predecessor = True
                        break
                if is_predecessor:
                    critical_predecessors.append(node)
                else:
                    non_critical_others.append(node)

            # Combine ready nodes with priority order for this job:
            prioritized_nodes_for_job = critical_ready_nodes + critical_predecessors + non_critical_others

            # --- Iterate through prioritized nodes to find one that can be scheduled ---
            for node_to_schedule_candidate in prioritized_nodes_for_job:
                # If this node is already running or selected for this cycle, skip it.
                if node_to_schedule_candidate in nodes_currently_running_or_selected:
                    continue

                    # Try to assign to the first suitable and available processor
                processor_to_assign = None
                if suitable_and_available_processors_for_this_job:  # Check if list is not empty
                    processor_to_assign = suitable_and_available_processors_for_this_job[0]  # Pick the first available

                if processor_to_assign is None:
                    continue  # No suitable and available processor found for this node, try next node in prioritized list

                # Resource acquisition check for the first section of this node
                node_allocations = job_obj.allocations.get(node_to_schedule_candidate, [])
                first_section_needs_resource = False
                first_section_resource_id = None
                if node_allocations:
                    potential_first_section = node_allocations[0]
                    if isinstance(potential_first_section, tuple) and potential_first_section[0] != "Normal":
                        first_section_needs_resource = True
                        first_section_resource_id = potential_first_section[0]

                can_acquire_initial_resource = True
                if first_section_needs_resource:
                    if not resource_manager.request_resource(node_to_schedule_candidate, first_section_resource_id):
                        can_acquire_initial_resource = False
                        # If resource not acquired, this node can't start this cycle, so continue
                        # print(f"DEBUG: Node {node_to_schedule_candidate} ({job_obj.instance_id}) cannot acquire initial resource {first_section_resource_id}. Waiting.")
                        continue  # Skip this node candidate, try next one.

                if can_acquire_initial_resource:
                    # FOUND A MATCH! Add to candidates_for_scheduling.
                    candidates_for_scheduling.append(
                        (job_obj.absolute_deadline, job_obj, node_to_schedule_candidate, processor_to_assign))

                    # Mark this node as selected for this cycle to prevent re-selection
                    nodes_currently_running_or_selected.add(node_to_schedule_candidate)

                    # Mark this processor as taken for this cycle (remove from available pool)
                    available_processors.remove(processor_to_assign)

                    # Break from `prioritized_nodes_for_job` loop as we assigned a node for this job
                    break

                    # Assign all selected candidates to processors
        # Sort candidates by their deadline again (as a safeguard for EDF)
        candidates_for_scheduling.sort(key=lambda x: x[0])

        for _, job_obj_selected, node_to_schedule_selected, processor_to_assign_selected in candidates_for_scheduling:
            # Check if processor is still available (it should be, as we removed it from `available_processors` above)
            if processor_to_assign_selected not in executing_nodes:  # This check is redundant with correct `available_processors` management
                first_section = job_obj_selected.allocations[node_to_schedule_selected][0]
                section_time = first_section[1] if isinstance(first_section, tuple) else first_section
                section_idx = 0

                executing_nodes[processor_to_assign_selected] = (
                job_obj_selected.instance_id, node_to_schedule_selected, section_time, section_idx, current_time,
                job_obj_selected)
                print(
                    f"INFO: Node {node_to_schedule_selected} ({job_obj_selected.instance_id}) started execution on processor {processor_to_assign_selected} at time {current_time}. (First section: {first_section})")

        # --- Show Current Status (for debugging) ---
        print(f"\n=== Time {current_time} === (Hyperperiod: {hyper_period})")
        print("Executing Nodes:")
        if not executing_nodes:
            print("  (None)")
        for proc_id, (inst_id, node_id, remaining, section_idx, start_time, job_obj) in executing_nodes.items():
            node_allocations = job_obj.allocations.get(node_id, [])

            if section_idx >= len(node_allocations):
                section_info = " (INVALID SECTION - SHOULD BE COMPLETED)"
            else:
                current_section_def_for_display = node_allocations[section_idx]
                if isinstance(current_section_def_for_display, tuple):
                    section_type, section_time_orig = current_section_def_for_display[0], \
                    current_section_def_for_display[1]
                    if section_type != "Normal":
                        status_str = " [Acquired]" if resource_manager.is_resource_locked_by(node_id,
                                                                                             section_type) else " [Waiting]"
                        section_info = f" (Resource {section_type}: {remaining}/{section_time_orig} time units remaining{status_str})"
                    else:
                        section_info = f" (Normal Section: {remaining}/{section_time_orig} time units remaining)"
                else:  # Normal section defined as integer (legacy or simple)
                    section_info = f" (Normal Section: {remaining}/{current_section_def_for_display} time units remaining)"

            print(f"  Processor {proc_id}: Node {node_id} ({inst_id}){section_info}")

        print("\nResource Status:")
        for resource_id, lock in resource_manager.resources.items():
            if lock.is_locked:
                print(f"  {resource_id}: Locked by Node {lock.locked_by}")
                if not lock.waiting_queue.empty():
                    waiting = list(lock.waiting_queue.queue)
                    print(f"    Waiting queue: {waiting}")
            else:
                print(f"  {resource_id}: Available")

        # 5. Update Executing Nodes: Perform Work or Advance
        nodes_to_remove_from_executing_this_cycle = []

        for processor_id, (inst_id, node_id, remaining_orig, section_idx_orig, start_time_orig, job_obj) in list(
                executing_nodes.items()):
            node_allocations = job_obj.allocations.get(node_id, [])

            if section_idx_orig >= len(node_allocations):
                print(
                    f"WARNING: Node {node_id} ({inst_id}) on P{processor_id} is in invalid section. Marking for removal.")
                nodes_to_remove_from_executing_this_cycle.append(processor_id)
                continue

            current_section_def = node_allocations[section_idx_orig]
            current_remaining = remaining_orig

            progress_made_this_cycle = True

            # --- Handle Resource Sections for currently executing nodes ---
            if isinstance(current_section_def, tuple) and current_section_def[0] != "Normal":
                resource_id = current_section_def[0]

                if not resource_manager.is_resource_locked_by(node_id, resource_id):
                    if not resource_manager.request_resource(node_id, resource_id):
                        print(
                            f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} is busy-waiting for {resource_id} (already on processor).")
                        progress_made_this_cycle = False
                        executing_nodes[processor_id] = (
                        inst_id, node_id, remaining_orig, section_idx_orig, start_time_orig, job_obj)
                        continue
                    else:
                        print(
                            f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} just ACQUIRED {resource_id} (was waiting on processor).")

                if current_remaining == 1:
                    if resource_manager.is_resource_locked_by(node_id, resource_id):
                        acquired_by_node = resource_manager.release_resource(node_id, resource_id)
                        print(f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} released resource {resource_id}.")
                        if acquired_by_node:
                            print(f"DEBUG: Resource {resource_id} transferred to {acquired_by_node}.")
                    else:
                        print(
                            f"WARNING: Node {node_id} ({inst_id}) on P{processor_id} finished resource section {resource_id} but didn't hold it. (Logical error)")

            # --- Decrement Remaining Time (Only if progress was made) ---
            if progress_made_this_cycle:
                current_remaining -= 1

            # --- Check if Current Section is Completed ---
            if current_remaining <= 0:
                print(
                    f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} finished section {section_idx_orig} at time {current_time}.")

                job_obj.completed_nodes.add(node_id)

                if section_idx_orig + 1 < len(node_allocations):
                    next_section_def = node_allocations[section_idx_orig + 1]
                    next_section_time = next_section_def[1] if isinstance(next_section_def, tuple) else next_section_def

                    executing_nodes[processor_id] = (
                    inst_id, node_id, next_section_time, section_idx_orig + 1, current_time + 1, job_obj)
                    print(
                        f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} moving to next section: {next_section_def} for {next_section_time} units.")
                else:
                    print(
                        f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} completed ENTIRELY at time {current_time}.")
                    resource_manager.release_all_resources_for_node(node_id)

                    job_obj.is_completed = True
                    job_results[inst_id]['completed'] = True
                    job_results[inst_id]['total_time'] = current_time

                    nodes_to_remove_from_executing_this_cycle.append(processor_id)

                    scheduling_log.append({
                        'time': current_time,
                        'task_id': job_obj.task_id,
                        'instance_id': inst_id,
                        'node': node_id,
                        'processor': processor_id,
                        'action': 'completed_node_and_job_part'
                    })
            else:
                executing_nodes[processor_id] = (
                inst_id, node_id, current_remaining, section_idx_orig, start_time_orig, job_obj)

        for proc_id_to_remove in nodes_to_remove_from_executing_this_cycle:
            if proc_id_to_remove in executing_nodes:
                del executing_nodes[proc_id_to_remove]

        current_time += 1

    total_simulation_time = current_time

    final_results_summary = {}
    overall_system_schedulable = True

    for task in base_tasks:
        task_id = task.id
        final_results_summary[task_id] = {
            'total_instances': 0,
            'completed_instances': 0,
            'missed_deadline_instances': 0,
            'schedulable': True,
            'overall_time_simulated': total_simulation_time
        }

    for inst_id, res in job_results.items():
        job_obj = job_map[inst_id]
        final_results_summary[job_obj.task_id]['total_instances'] += 1

        if job_obj.is_completed and not job_obj.deadline_missed:
            final_results_summary[job_obj.task_id]['completed_instances'] += 1
        else:
            final_results_summary[job_obj.task_id]['missed_deadline_instances'] += 1
            final_results_summary[job_obj.task_id]['schedulable'] = False
            overall_system_schedulable = False

    print("\n=== Overall Simulation Summary ===")
    print(f"Total Simulation Time: {total_simulation_time}")
    print(f"Total Processors Used: {len([p for p in system_processors if p.assigned_tasks])}")
    print(f"Overall System Schedulability: {'YES' if overall_system_schedulable else 'NO'}")

    for task_id, summary in final_results_summary.items():
        print(f"\nTask T{task_id} Summary:")
        print(f"  Total Instances: {summary['total_instances']}")
        print(f"  Completed Instances: {summary['completed_instances']}")
        print(f"  Missed Deadline Instances: {summary['missed_deadline_instances']}")
        print(f"  Task Schedulability: {'YES' if summary['schedulable'] else 'NO'}")

    return final_results_summary


def run_complete_example():
    print("=== Starting Complete Example ===")

    # 1. Generate Base Tasks (Task templates)
    print("\n1. Generating Base Tasks...")
    num_base_tasks = 2  # Number of distinct periodic task types
    num_resources = 6

    base_tasks_list = generate_tasks(num_base_tasks=num_base_tasks, num_resources=num_resources)

    if not base_tasks_list:
        print("Failed to generate any base tasks. Exiting.")
        return

    # 2. Allocate Processors for the Base Tasks (Federated Scheduling)
    print("\n2. Allocating Processors using Federated Scheduling...")
    # This step uses base task properties to allocate processors
    system_processors = federated_scheduling(base_tasks_list)

    # 3. Schedule all Instances of Periodic Tasks over Hyperperiod
    print("\n3. Scheduling Periodic Task Instances over Hyperperiod...")
    # This is where the main scheduling logic happens
    results_summary = schedule_multiple_tasks_periodic(base_tasks_list, system_processors)

    print("\n=== Final Results: Schedulability Analysis ===")
    overall_system_schedulable = True
    for task_id, summary in results_summary.items():
        if not summary['schedulable']:
            overall_system_schedulable = False
        print(
            f"Task T{task_id}: Schedulable = {'YES' if summary['schedulable'] else 'NO'} (Completed {summary['completed_instances']}/{summary['total_instances']} instances)")

    print(f"\nOverall System Schedulable: {'YES' if overall_system_schedulable else 'NO'}")


# --- Main execution block ---
if __name__ == "__main__":
    run_complete_example()