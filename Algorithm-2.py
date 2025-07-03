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


class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, ResourceLock] = {}

    def initialize_resources(self, resource_ids: List[str]):
        for resource_id in resource_ids:
            self.resources[resource_id] = ResourceLock()

    def request_resource(self, node_id: str, resource_id: str) -> bool:
        if resource_id not in self.resources:
            return True

        lock = self.resources[resource_id]

        if not lock.is_locked:
            lock.is_locked = True
            lock.locked_by = node_id
            return True
        elif lock.locked_by == node_id:
            return True
        else:
            if node_id not in [item for item in list(lock.waiting_queue.queue)]:
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
    wcet: int

    def __str__(self) -> str:
        return f"Node: {self.id}, WCET: {self.wcet}"


@dataclass
class Edge:
    src: str
    sink: str


@dataclass
class BaseTask:
    id: int
    period: int
    wcet: int
    nodes: list
    edges: list
    deadline: int
    critical_path: list
    critical_path_length: int
    allocations: Dict[str, list] = field(default_factory=dict)
    execution_times: Dict[str, int] = field(default_factory=dict)

    def get_total_wcet(self) -> int:
        return sum(self.execution_times.get(node, 0) for node in self.nodes if node not in ["source", "sink"])

    def utilization(self) -> float:
        total_node_time = 0
        for node_id, sections in self.allocations.items():
            for section in sections:
                if isinstance(section, tuple):
                    total_node_time += section[1]
                else:
                    total_node_time += section

        return total_node_time / self.period if self.period > 0 else 0.0

    def __str__(self) -> str:
        return f"Task T{self.id} (Period: {self.period}, WCET: {self.get_total_wcet()}, Deadline: {self.deadline})"


@dataclass
class TaskInstance:
    task_id: int
    instance_num: int
    instance_id: str
    release_time: int
    absolute_deadline: int

    nodes: list
    edges: list
    execution_times: Dict[str, int]
    allocations: Dict[str, list]
    critical_path: list
    period: int

    completed_nodes: set = field(default_factory=set)
    is_completed: bool = False
    deadline_missed: bool = False

    def __post_init__(self):
        self.completed_nodes.add("source")
        self.completed_nodes.add("sink")

    def __lt__(self, other):
        return self.absolute_deadline < other.absolute_deadline

    def __str__(self) -> str:
        return (f"Instance {self.instance_id} (Task T{self.task_id}) "
                f"Release: {self.release_time}, AbsDL: {self.absolute_deadline}")


# --- Graph Generation and Task Properties ---
def erdos_renyi_graph(num_nodes: int, density: float):
    G = nx.erdos_renyi_graph(num_nodes, density, directed=True)
    G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    mapping = {node: f"N{node + 1}" for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    source_node = "source"
    sink_node = "sink"
    G.add_node(source_node)
    G.add_node(sink_node)

    for node in list(G.nodes()):
        if G.in_degree(node) == 0 and node != source_node:
            G.add_edge(source_node, node)
        if G.out_degree(node) == 0 and node != sink_node and node != source_node:
            G.add_edge(node, sink_node)

    return G


def visualize_task(G):
    pass


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


def generate_accesses_and_lengths(num_tasks: int, num_resources: int = 32) -> tuple[dict, dict]:
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
    execution_times_allocated = task_info["execution_times"].copy()

    for node_id in nodes:
        original_wcet = task_info["execution_times"][node_id]

        node_resource_sections = []
        for resource, task_access_counts in accesses.items():
            if task_id - 1 < len(task_access_counts) and task_access_counts[task_id - 1] > 0:
                node_access_lengths_for_resource = lengths[resource][task_id - 1]

                num_accesses_for_node = random.randint(0, len(node_access_lengths_for_resource))

                for _ in range(num_accesses_for_node):
                    if node_access_lengths_for_resource:
                        access_length = node_access_lengths_for_resource.pop(0)
                        node_resource_sections.append((resource, access_length))

        total_resource_time_for_node = sum(sec[1] for sec in node_resource_sections)

        remaining_normal_time = max(0, original_wcet - total_resource_time_for_node)

        final_allocation_for_node = []

        num_normal_segments = len(node_resource_sections) + 1

        normal_segment_lengths = [0] * num_normal_segments
        if remaining_normal_time > 0:
            if num_normal_segments > 0:
                distribution_base = [1] * num_normal_segments
                remaining_to_distribute = remaining_normal_time - num_normal_segments
                if remaining_to_distribute < 0:
                    distribution_base = [0] * num_normal_segments
                    remaining_to_distribute = remaining_normal_time
                    for i in range(remaining_to_distribute):
                        distribution_base[i % num_normal_segments] += 1
                else:
                    for _ in range(remaining_to_distribute):
                        distribution_base[random.randint(0, num_normal_segments - 1)] += 1
                normal_segment_lengths = distribution_base
            else:
                normal_segment_lengths = [remaining_normal_time]

        for i in range(num_normal_segments):
            if normal_segment_lengths[i] > 0:
                final_allocation_for_node.append(("Normal", normal_segment_lengths[i]))
            if i < len(node_resource_sections):
                final_allocation_for_node.append(node_resource_sections[i])

        filtered_allocation = []
        for sec in final_allocation_for_node:
            if isinstance(sec, tuple) and sec[0] == "Normal" and sec[1] == 0:
                continue
            filtered_allocation.append(sec)

        if not filtered_allocation:
            filtered_allocation.append(("Normal", 1))

        allocations[node_id] = filtered_allocation

        total_allocated_time_for_node = sum(sec[1] if isinstance(sec, tuple) else sec for sec in filtered_allocation)
        execution_times_allocated[node_id] = total_allocated_time_for_node

    return allocations, execution_times_allocated


def generate_base_task(task_id: int, accesses_data: dict, lengths_data: dict, num_nodes: int, density: float,
                       verbose: bool = False) -> Optional[BaseTask]:
    """
    Generates a base task template with random DAG, execution times, and resource allocations.
    Accepts num_nodes and density for graph generation.
    Added 'verbose' flag to control printing.
    """
    G = erdos_renyi_graph(num_nodes=num_nodes, density=density)
    nodes = list(G.nodes())
    edges = list(G.edges())

    actual_nodes = [n for n in nodes if n not in ["source", "sink"]]
    if not actual_nodes:
        return None

    execution_times_initial = {node: random.randint(10, 20) for node in actual_nodes}

    allocations, execution_times_final = allocate_resources_to_nodes(
        {"nodes": nodes, "edges": edges, "execution_times": execution_times_initial},
        task_id,
        accesses_data,
        lengths_data
    )

    critical_path, critical_path_length = get_critical_path(nodes, edges, execution_times_final)

    if critical_path_length == 0 and len(actual_nodes) > 0:
        critical_path_length = sum(execution_times_final.values()) / len(actual_nodes)
        if critical_path_length == 0: critical_path_length = 10

    total_execution_time = sum(execution_times_final.values())

    critical_pathth = critical_path_length
    x = 1
    while x < critical_pathth:
        x *= 2

    period = rand.choice([x, 2 * x])
    if period == 0: period = 1

    deadline = period

    base_task = BaseTask(
        id=task_id,
        period=period,
        wcet=total_execution_time,
        nodes=nodes,
        edges=edges,
        deadline=deadline,
        critical_path=critical_path,
        critical_path_length=critical_path_length,
        allocations=allocations,
        execution_times=execution_times_final
    )

    if verbose:
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


def generate_tasks(num_base_tasks: int, num_resources: int, num_nodes_per_task: int, density: float,
                   verbose: bool = False) -> List[BaseTask]:
    tasks_list = []
    all_resource_ids = [f"R{i + 1}" for i in range(num_resources)]

    accesses_data, lengths_data = generate_accesses_and_lengths(num_base_tasks, num_resources)

    for i in range(1, num_base_tasks + 1):
        task = generate_base_task(task_id=i, accesses_data=accesses_data, lengths_data=lengths_data,
                                  num_nodes=num_nodes_per_task, density=density, verbose=verbose)
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


def generate_periodic_task_instances(base_tasks: List[BaseTask], hyper_period: int, verbose: bool = False) -> List[
    TaskInstance]:
    all_instances: List[TaskInstance] = []
    for task in base_tasks:
        num_instances = hyper_period // task.period
        if hyper_period % task.period == 0 and hyper_period != 0:
            num_instances -= 1

        for i in range(num_instances + 1):
            release_t = task.period * i
            abs_deadline = release_t + task.deadline

            instance_nodes = list(task.nodes)
            instance_edges = list(task.edges)
            instance_execution_times = task.execution_times.copy()
            instance_allocations = {k: [tuple(section) if isinstance(section, tuple) else section for section in v] for
                                    k, v in task.allocations.items()}

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
                critical_path=list(task.critical_path),
                period=task.period
            )
            all_instances.append(instance)

    all_instances.sort(key=lambda inst: inst.release_time)
    if verbose:
        print(f"\nGenerated {len(all_instances)} total task instances over hyperperiod {hyper_period}.")
    return all_instances


# --- Processor and Scheduling Utilities ---
@dataclass
class Processor:
    id: int
    assigned_tasks: List[int]
    utilization: float = 0.0


def calculate_total_processors(tasks: List[BaseTask], U_norm: float) -> int:
    sum_utilization = sum(t.utilization() for t in tasks)
    m_total = math.ceil(sum_utilization / U_norm)
    if m_total == 0:
        m_total = 1
    return m_total


def federated_scheduling(base_tasks: List[BaseTask], all_resource_ids: List[str], U_norm: float,
                         verbose: bool = False) -> List[Processor]:
    m_total_system_processors = calculate_total_processors(base_tasks, U_norm)
    if verbose:
        print(f"DEBUG: Total system processors available (m_total): {m_total_system_processors}")

    if verbose:
        print(f"\n--- Federated Scheduling (Resource-Based Grouping) Allocation ---")

    task_resource_access_map: Dict[int, Dict[str, int]] = {}
    for task in base_tasks:
        task_resource_access_map[task.id] = {res_id: 0 for res_id in all_resource_ids}
        for node_id, allocations in task.allocations.items():
            for section in allocations:
                if isinstance(section, tuple) and section[0] != "Normal":
                    resource_id = section[0]
                    if resource_id in all_resource_ids:
                        task_resource_access_map[task.id][resource_id] += 1

    if verbose:
        print("\nTask Resource Access Map:")
        for task_id, accesses in task_resource_access_map.items():
            print(f"  Task T{task_id}: {accesses}")

    resource_total_access_counts: Dict[str, int] = {res_id: 0 for res_id in all_resource_ids}
    for task_id, accesses in task_resource_access_map.items():
        for res_id, count in accesses.items():
            resource_total_access_counts[res_id] += count

    sorted_resources_by_access = sorted(resource_total_access_counts.items(), key=lambda item: item[1], reverse=True)

    task_groups: List[List[BaseTask]] = []

    unassigned_task_ids = {task.id for task in base_tasks}
    task_id_map: Dict[int, BaseTask] = {task.id: task for task in base_tasks}

    if verbose:
        print("\nResource Grouping Process:")
    for resource_id, _ in sorted_resources_by_access:
        if resource_id not in all_resource_ids:
            continue

        current_group_task_ids = []

        for task_id in list(unassigned_task_ids):
            if task_resource_access_map[task_id].get(resource_id, 0) > 0:
                current_group_task_ids.append(task_id)

        if current_group_task_ids:
            newly_added_to_group = []
            for task_id in current_group_task_ids:
                if task_id in unassigned_task_ids:
                    newly_added_to_group.append(task_id_map[task_id])

            if newly_added_to_group:
                task_groups.append(newly_added_to_group)
                for task in newly_added_to_group:
                    unassigned_task_ids.remove(task.id)
                if verbose:
                    print(f"  Group created for Resource {resource_id}: {[t.id for t in newly_added_to_group]}")

    if unassigned_task_ids:
        for task_id in unassigned_task_ids:
            task_groups.append([task_id_map[task_id]])
            if verbose:
                print(f"  Individual group created for Task T{task_id}")

    if verbose:
        print("\nFinal Task Groups:")
        for i, group in enumerate(task_groups):
            print(f"  Group {i + 1}: {[task.id for task in group]}")

    processors: List[Processor] = []
    processor_id_counter = 1
    total_processors_assigned_by_federated = 0

    for i, group in enumerate(task_groups):
        group_utilization = sum(task.utilization() for task in group)

        num_cores_for_group = math.ceil(group_utilization)

        if num_cores_for_group == 0 and group:
            num_cores_for_group = 1

        total_processors_assigned_by_federated += num_cores_for_group

        assigned_task_ids = [task.id for task in group]
        for _ in range(num_cores_for_group):
            processor = Processor(id=processor_id_counter, assigned_tasks=assigned_task_ids, utilization=0.0)
            processors.append(processor)
            processor_id_counter += 1

        if num_cores_for_group > 0:
            util_per_processor = group_utilization / num_cores_for_group
            for p in processors[-num_cores_for_group:]:
                p.utilization = util_per_processor

    if total_processors_assigned_by_federated > m_total_system_processors:
        if verbose:
            print(
                f"INFO: Federated allocation failed! Requested {total_processors_assigned_by_federated} processors, but only {m_total_system_processors} available.")
        return []

    if verbose:
        print("\n--- Final Processor Allocation Summary (Resource-Based Grouping) ---")
        for p in processors:
            print(f"Processor {p.id}: Assigned Tasks {p.assigned_tasks}, Utilization: {p.utilization:.2f}")

    return processors


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


def schedule_multiple_tasks_periodic(base_tasks: List[BaseTask], system_processors: List[Processor],
                                     max_simulation_time_factor: int, verbose: bool = False) -> bool:
    if verbose:
        print("\n=== Starting Periodic Task Scheduling ===")

    hyper_period = hyperperiod_lcm(base_tasks)
    if verbose:
        print(f"Calculated Hyperperiod: {hyper_period}")
    max_simulation_time = hyper_period * max_simulation_time_factor

    if hyper_period > 10000:
        max_simulation_time = 10000 * max_simulation_time_factor
        if verbose:
            print(
                f"WARNING: Hyperperiod ({hyper_period}) is very large. Capping simulation time to {max_simulation_time}.")

    all_instances: List[TaskInstance] = generate_periodic_task_instances(base_tasks, hyper_period, verbose=verbose)

    resource_manager = ResourceManager()
    all_resource_ids = set()
    for task in base_tasks:
        for node_id, allocations in task.allocations.items():
            for section in allocations:
                if isinstance(section, tuple) and section[0] != "Normal":
                    all_resource_ids.add(section[0])
    resource_manager.initialize_resources(list(all_resource_ids))

    current_time = 0
    active_jobs_pq: List[Tuple[int, int, str, TaskInstance]] = []
    heapq.heapify(active_jobs_pq)

    executing_nodes: Dict[int, Tuple[str, str, int, int, int, TaskInstance]] = {}

    job_map: Dict[str, TaskInstance] = {instance.instance_id: instance for instance in all_instances}

    job_results: Dict[str, Dict] = {inst.instance_id: {'completed': False, 'deadline_missed': False, 'total_time': 0}
                                    for inst in all_instances}

    overall_schedulable = True

    while current_time < max_simulation_time:
        if verbose:
            print(f"\n--- Time {current_time} --- (Hyperperiod: {hyper_period})")

        # 1. Release new job instances
        jobs_released_this_cycle = False
        for instance in all_instances:
            if instance.release_time == current_time and not instance.is_completed and not instance.deadline_missed:
                if instance.instance_id not in {j[2] for j in active_jobs_pq}:
                    heapq.heappush(active_jobs_pq,
                                   (instance.absolute_deadline, instance.release_time, instance.instance_id, instance))
                    if verbose:
                        print(
                            f"INFO: {instance.instance_id} released and added to active_jobs_pq at time {current_time}. AbsDL: {instance.absolute_deadline}")
                    jobs_released_this_cycle = True
                else:
                    if verbose:
                        print(f"DEBUG: {instance.instance_id} already in active_jobs_pq, skipping re-add.")

        if verbose and jobs_released_this_cycle:
            print(f"DEBUG: Active jobs PQ size after release: {len(active_jobs_pq)}")

        # 2. Check for missed deadlines among active and executing jobs
        jobs_to_remove_from_active_pq = []
        for deadline, release, inst_id, job_obj in list(active_jobs_pq):
            if current_time > job_obj.absolute_deadline and not job_obj.is_completed and not job_obj.deadline_missed:
                job_obj.deadline_missed = True
                job_obj.is_completed = True
                overall_schedulable = False
                job_results[inst_id]['deadline_missed'] = True
                job_results[inst_id]['total_time'] = current_time
                if verbose:
                    print(
                        f"WARNING: {job_obj.instance_id} MISSED DEADLINE at time {current_time} (was {job_obj.absolute_deadline}). Job status updated.")

                for proc_id, exec_data in list(executing_nodes.items()):
                    if exec_data[0] == inst_id:
                        if verbose:
                            print(
                                f"DEBUG: Removing {inst_id} (node {exec_data[1]}) from P{proc_id} due to deadline miss.")
                        del executing_nodes[proc_id]
                        resource_manager.release_all_resources_for_node(exec_data[1])
                jobs_to_remove_from_active_pq.append((deadline, release, inst_id, job_obj))

        if jobs_to_remove_from_active_pq:
            active_jobs_pq = [(d, r, i, j) for d, r, i, j in active_jobs_pq if
                              not j.deadline_missed and not j.is_completed]
            heapq.heapify(active_jobs_pq)

        # --- Check for early loop termination ---
        all_jobs_in_instances_list_processed = True
        for job_obj in all_instances:
            if not job_obj.is_completed and job_obj.release_time <= current_time:
                all_jobs_in_instances_list_processed = False
                break

        if all_jobs_in_instances_list_processed and not executing_nodes and current_time >= hyper_period:
            if verbose:
                print(f"\n--- Simulation End Condition Met: All relevant jobs processed or hyperperiod reached. ---")
            break

        if current_time >= max_simulation_time - 1 and (len(active_jobs_pq) > 0 or len(executing_nodes) > 0):
            overall_schedulable = False
            if verbose:
                print(
                    f"WARNING: Simulation reached max_simulation_time ({max_simulation_time}) with pending work. System likely unschedulable.")
            break

        # 3. Select jobs/nodes to execute for this time step (EDF & NO CPC for node selection)
        current_tick_available_processors = [p.id for p in system_processors if p.id not in executing_nodes]
        candidates_for_scheduling = []

        nodes_being_scheduled_this_cycle = set()
        for proc_id, (inst_id, node_id, _, _, _, _) in executing_nodes.items():
            nodes_being_scheduled_this_cycle.add(node_id)

        jobs_to_consider_for_scheduling = list(active_jobs_pq)
        jobs_to_consider_for_scheduling.sort(key=lambda x: x[0])

        for _, _, _, job_obj in jobs_to_consider_for_scheduling:
            if job_obj.is_completed or job_obj.deadline_missed:
                continue

            processors_assigned_to_this_job_base_task = []
            for p in system_processors:
                if job_obj.task_id in p.assigned_tasks:
                    processors_assigned_to_this_job_base_task.append(p.id)

            suitable_and_available_processors_for_this_job = [
                p_id for p_id in current_tick_available_processors
                if p_id in processors_assigned_to_this_job_base_task
            ]

            if not suitable_and_available_processors_for_this_job:
                continue

            ready_nodes_for_job = find_ready_nodes(job_obj.nodes, job_obj.edges, job_obj.completed_nodes)
            actual_ready_nodes = [n for n in ready_nodes_for_job if n not in ["source", "sink"]]

            if not actual_ready_nodes:
                continue

            prioritized_nodes_for_job = sorted(actual_ready_nodes)

            for node_to_schedule_candidate in prioritized_nodes_for_job:
                if node_to_schedule_candidate in nodes_being_scheduled_this_cycle:
                    continue

                processor_to_assign = None
                if suitable_and_available_processors_for_this_job:
                    processor_to_assign = suitable_and_available_processors_for_this_job.pop(0)
                else:
                    break

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
                        suitable_and_available_processors_for_this_job.insert(0, processor_to_assign)
                        continue

                if can_acquire_initial_resource:
                    candidates_for_scheduling.append(
                        (job_obj.absolute_deadline, job_obj, node_to_schedule_candidate, processor_to_assign))
                    nodes_being_scheduled_this_cycle.add(node_to_schedule_candidate)
                    current_tick_available_processors.remove(processor_to_assign)

        candidates_for_scheduling.sort(key=lambda x: x[0])

        for _, job_obj_selected, node_to_schedule_selected, processor_to_assign_selected in candidates_for_scheduling:
            if processor_to_assign_selected not in executing_nodes:
                first_section = job_obj_selected.allocations[node_to_schedule_selected][0]
                section_time = first_section[1] if isinstance(first_section, tuple) else first_section
                section_idx = 0

                executing_nodes[processor_to_assign_selected] = (
                    job_obj_selected.instance_id, node_to_schedule_selected, section_time, section_idx, current_time,
                    job_obj_selected)
                if verbose:
                    print(
                        f"INFO: Node {node_to_schedule_selected} ({job_obj_selected.instance_id}) started execution on processor {processor_to_assign_selected} at time {current_time}. (First section: {first_section})")

        if verbose:
            print("\nExecuting Nodes:")
            if not executing_nodes:
                print("  (None)")
            for proc_id, (inst_id, node_id, remaining, section_idx, start_time, job_obj) in executing_nodes.items():
                node_allocations = job_obj.allocations.get(node_id, [])
                if section_idx < len(node_allocations):
                    current_section_def_for_display = node_allocations[section_idx]
                    section_type, section_time_orig = current_section_def_for_display[0], \
                    current_section_def_for_display[1]
                    status_str = " [Acquired]" if resource_manager.is_resource_locked_by(node_id,
                                                                                         section_type) else " [Waiting]"
                    section_info = f" (Resource {section_type}: {remaining}/{section_time_orig} time units remaining{status_str})" if section_type != "Normal" else f" (Normal Section: {remaining}/{section_time_orig} time units remaining)"
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

        nodes_to_remove_from_executing_this_cycle = []

        for processor_id, (inst_id, node_id, remaining_orig, section_idx_orig, start_time_orig, job_obj) in list(
                executing_nodes.items()):
            node_allocations = job_obj.allocations.get(node_id, [])

            if section_idx_orig >= len(node_allocations):
                if verbose:
                    print(
                        f"WARNING: Node {node_id} ({inst_id}) on P{processor_id} is in invalid section. Marking for removal.")
                nodes_to_remove_from_executing_this_cycle.append(processor_id)
                continue

            current_section_def = node_allocations[section_idx_orig]
            current_remaining = remaining_orig

            progress_made_this_cycle = True

            if isinstance(current_section_def, tuple) and current_section_def[0] != "Normal":
                resource_id = current_section_def[0]

                if not resource_manager.is_resource_locked_by(node_id, resource_id):
                    if not resource_manager.request_resource(node_id, resource_id):
                        if verbose:
                            print(
                                f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} is busy-waiting for {resource_id} (already on processor).")
                        progress_made_this_cycle = False
                        executing_nodes[processor_id] = (
                            inst_id, node_id, remaining_orig, section_idx_orig, start_time_orig, job_obj)
                        continue

            if progress_made_this_cycle:
                current_remaining -= 1

            if current_remaining <= 0:
                if verbose:
                    print(
                        f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} finished section {section_idx_orig} at time {current_time}.")

                if isinstance(current_section_def, tuple) and current_section_def[0] != "Normal":
                    resource_id_to_release = current_section_def[0]
                    if resource_manager.is_resource_locked_by(node_id, resource_id_to_release):
                        acquired_by_node = resource_manager.release_resource(node_id, resource_id_to_release)
                        if verbose:
                            print(
                                f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} released resource {resource_id_to_release}.")
                            if acquired_by_node:
                                print(f"DEBUG: Resource {resource_id_to_release} transferred to {acquired_by_node}.")

                if section_idx_orig + 1 < len(node_allocations):
                    next_section_def = node_allocations[section_idx_orig + 1]
                    next_section_time = next_section_def[1] if isinstance(next_section_def, tuple) else next_section_def

                    executing_nodes[processor_id] = (
                        inst_id, node_id, next_section_time, section_idx_orig + 1, current_time + 1, job_obj)
                    if verbose:
                        print(
                            f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} moving to next section: {next_section_def} for {next_section_time} units.")
                else:
                    if verbose:
                        print(
                            f"DEBUG: Node {node_id} ({inst_id}) on P{processor_id} completed ENTIRELY at time {current_time}.")

                    job_obj.completed_nodes.add(node_id)

                    all_functional_nodes = [n for n in job_obj.nodes if n not in ["source", "sink"]]
                    if all(n in job_obj.completed_nodes for n in all_functional_nodes):
                        job_obj.is_completed = True
                        job_results[inst_id]['completed'] = True
                        job_results[inst_id]['total_time'] = current_time
                        if verbose:
                            print(f"INFO: Job {inst_id} (T{job_obj.task_id}) completed at time {current_time}.")

                    nodes_to_remove_from_executing_this_cycle.append(processor_id)

            else:
                executing_nodes[processor_id] = (
                    inst_id, node_id, current_remaining, section_idx_orig, start_time_orig, job_obj)

        for proc_id_to_remove in nodes_to_remove_from_executing_this_cycle:
            if proc_id_to_remove in executing_nodes:
                del executing_nodes[proc_id_to_remove]

        current_time += 1

    final_overall_schedulable = True
    for inst_id, res in job_results.items():
        job_obj = job_map[inst_id]
        if job_obj.release_time < hyper_period:
            if res['deadline_missed'] or not res['completed']:
                final_overall_schedulable = False
                break

    if verbose:
        print("\n=== Overall Simulation Summary ===")
        print(f"Total Simulation Time: {current_time}")
        unique_processors_used = len(set(p.id for p in system_processors if p.assigned_tasks))
        print(f"Total Processors Used (Assigned to at least one task): {unique_processors_used}")
        print(f"Overall System Schedulability: {'YES' if final_overall_schedulable else 'NO'}")

    return final_overall_schedulable


# MODIFIED: run_single_simulation now accepts U_norm_val again
def run_single_simulation(U_norm_val: float, num_tasks_val: int, num_resources_val: int, density_val: float) -> bool:
    """
    Runs a single simulation with specified parameters and returns True if schedulable, False otherwise.
    The number of nodes per task is randomly determined within the function.
    This function is intended for statistical runs, hence 'verbose' is False by default.
    """
    # Define num_nodes_for_each_task here as a random value between 5 and 20
    num_nodes_for_each_task = random.randint(5, 20)  # Corrected as per user's request

    # Generate Base Tasks
    base_tasks_list = generate_tasks(num_base_tasks=num_tasks_val, num_resources=num_resources_val,
                                     num_nodes_per_task=num_nodes_for_each_task, density=density_val, verbose=False)

    if not base_tasks_list:
        return False

    all_resource_ids = [f"R{i + 1}" for i in range(num_resources_val)]

    system_processors = federated_scheduling(base_tasks_list, all_resource_ids, U_norm_val, verbose=False)

    if not system_processors:
        return False

    is_schedulable = schedule_multiple_tasks_periodic(base_tasks_list, system_processors, max_simulation_time_factor=5,
                                                      verbose=False)

    return is_schedulable


# New function for detailed, single-run debugging
def run_single_verbose_simulation(U_norm_val: float, num_tasks_val: int, num_resources_val: int, density_val: float,
                                  num_nodes_per_task_val: int):
    """
    Runs a single simulation with specified parameters and prints all verbose details.
    Intended for debugging and understanding scheduling flow.
    NOTE: num_nodes_per_task_val is provided as an explicit argument for this verbose run,
    allowing you to fix task size for a specific debug scenario.
    """
    print("=== Starting a single VERBOSE Simulation ===")

    print("\n1. Generating Base Tasks...")
    base_tasks_list = generate_tasks(num_base_tasks=num_tasks_val, num_resources=num_resources_val,
                                     num_nodes_per_task=num_nodes_per_task_val, density=density_val, verbose=True)

    if not base_tasks_list:
        print("Failed to generate any base tasks. Exiting verbose simulation.")
        return

    all_resource_ids = [f"R{i + 1}" for i in range(num_resources_val)]

    print(f"DEBUG: Using U_norm for verbose run: {U_norm_val:.2f}")
    print("\n2. Allocating Processors using Resource-Based Grouping...")
    system_processors = federated_scheduling(base_tasks_list, all_resource_ids, U_norm_val, verbose=True)

    if not system_processors:
        print("\n=== Federated Allocation Failed: Not enough processors for groups. System is NOT schedulable. ===")
        print(f"\nOverall System Schedulability: NO")
        return

    print("\n3. Scheduling Periodic Task Instances over Hyperperiod...")
    is_schedulable = schedule_multiple_tasks_periodic(base_tasks_list, system_processors, max_simulation_time_factor=5,
                                                      verbose=True)

    print("\n=== Final Results: Schedulability Analysis (VERBOSE) ===")
    print(f"\nOverall System Schedulability: {'YES' if is_schedulable else 'NO'}")
    print("\n=== End of single VERBOSE Simulation ===")


def run_all_scenarios(num_runs_per_scenario: int = 100):
    """
    Runs simulations for all specified scenarios.
    """
    U_norm_values = [0.1, 0.5, 0.7, 1.0]
    num_tasks_values = [2, 4, 8, 16]
    num_resource_values = [2, 4, 8, 16, 32]
    density_values = [0.1, 0.3, 0.7]
    # num_nodes_per_task_values is now implicitly handled by random.randint in run_single_simulation
    # The scenarios where num_nodes_per_task was varied will now just use the random value.

    # Default values for scenarios where one parameter is varied
    # num_nodes_per_task_val is no longer a fixed default here, it's random per run in run_single_simulation
    default_U_norm = 0.5
    default_num_tasks = 4
    default_num_resources = 8
    default_density = 0.1

    scenario_count = 0

    print("=== شروع اجرای سناریوهای زمان‌بندی‌پذیری ===")

    # Scenario 1: Vary U_norm
    for U_norm_val in U_norm_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: U_norm = {U_norm_val} ---")
        # Removed explicit num_nodes_per_task_val from print since it's random per run
        print(
            f"پارامترهای ثابت: Num_tasks={default_num_tasks}, Num_resources={default_num_resources}, Density={default_density}, Num_nodes_per_task=RANDOM(5-20)")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            current_run_schedulable = run_single_simulation(
                U_norm_val=U_norm_val,
                num_tasks_val=default_num_tasks,
                num_resources_val=default_num_resources,
                density_val=default_density
                # num_nodes_per_task_val is now generated inside run_single_simulation
            )
            if current_run_schedulable:
                schedulable_count += 1

        print(f"خلاصه سناریو U_norm = {U_norm_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    # Scenario 2: Vary Num_tasks
    for num_tasks_val in num_tasks_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: تعداد تسک‌ها = {num_tasks_val} ---")
        print(
            f"پارامترهای ثابت: U_norm={default_U_norm}, منابع={default_num_resources}, چگالی={default_density}, Num_nodes_per_task=RANDOM(5-20)")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            current_run_schedulable = run_single_simulation(
                U_norm_val=default_U_norm,
                num_tasks_val=num_tasks_val,
                num_resources_val=default_num_resources,
                density_val=default_density
            )
            if current_run_schedulable:
                schedulable_count += 1

        print(f"خلاصه سناریو تعداد تسک‌ها = {num_tasks_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    # Scenario 3: Vary Num_resource
    for num_resources_val in num_resource_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: تعداد منابع = {num_resources_val} ---")
        print(
            f"پارامترهای ثابت: U_norm={default_U_norm}, تسک‌ها={default_num_tasks}, چگالی={default_density}, Num_nodes_per_task=RANDOM(5-20)")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            current_run_schedulable = run_single_simulation(
                U_norm_val=default_U_norm,
                num_tasks_val=default_num_tasks,
                num_resources_val=num_resources_val,
                density_val=default_density
            )
            if current_run_schedulable:
                schedulable_count += 1

        print(f"خلاصه سناریو تعداد منابع = {num_resources_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    # Scenario 4: Vary Density
    for density_val in density_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: چگالی = {density_val} ---")
        print(
            f"پارامترهای ثابت: U_norm={default_U_norm}, تسک‌ها={default_num_tasks}, منابع={default_num_resources}, Num_nodes_per_task=RANDOM(5-20)")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            current_run_schedulable = run_single_simulation(
                U_norm_val=default_U_norm,
                num_tasks_val=default_num_tasks,
                num_resources_val=default_num_resources,
                density_val=density_val
            )
            if current_run_schedulable:
                schedulable_count += 1

        print(f"خلاصه سناریو چگالی = {density_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    # Scenario 5: Vary Num_nodes_per_task (THIS SCENARIO IS NO LONGER MEANINGFUL WITH RANDOM NODE COUNT)
    # Since num_nodes_per_task is now always random, varying it explicitly in a loop
    # as [5, 10, 20] doesn't make sense if each run is already random.
    # If you still want to test *specific fixed* node counts, run_single_simulation
    # would need its num_nodes_per_task_val parameter re-added, or this scenario
    # could be removed/rethought based on your exact experimental design.
    # For now, I will remove this scenario's loop for clarity, as it conflicts with the new approach.
    print(
        f"\nNOTE: Scenario for 'Vary Num_nodes_per_task' is removed. Num_nodes_per_task is now random (5-20) in all runs.")

    print("\n=== پایان اجرای تمامی سناریوها ===")


# --- Main execution block ---
# Define run_complete_example here (it's not used by run_all_scenarios, but for manual debug runs)
def run_complete_example():
    # This function is intended for single-run debugging, not for silent scenario testing.
    print("=== Starting Complete Example ===")

    print("\n1. Generating Base Tasks...")
    num_base_tasks = 3
    num_resources = 3
    # num_nodes is now defined randomly for the single example run
    num_nodes_for_each_task = random.randint(5, 20)
    density = 0.1  # Example density

    base_tasks_list = generate_tasks(num_base_tasks=num_base_tasks, num_resources=num_resources,
                                     num_nodes_per_task=num_nodes_for_each_task, density=density, verbose=True)

    if not base_tasks_list:
        print("Failed to generate any base tasks. Exiting.")
        return

    all_resource_ids = [f"R{i + 1}" for i in range(num_resources)]

    U_norm_for_example = rand.uniform(0.1, 1.0)
    print(f"DEBUG: Using U_norm for example run: {U_norm_for_example:.2f}")

    print("\n2. Allocating Processors using Resource-Based Grouping...")
    system_processors = federated_scheduling(base_tasks_list, all_resource_ids, U_norm_for_example, verbose=True)

    if not system_processors:
        print("\n=== Federated Allocation Failed: Not enough processors for groups. System is NOT schedulable. ===")
        print(f"\nOverall System Schedulability: NO")
        return

    print("\n3. Scheduling Periodic Task Instances over Hyperperiod...")
    is_schedulable = schedule_multiple_tasks_periodic(base_tasks_list, system_processors, max_simulation_time_factor=5,
                                                      verbose=True)

    print("\n=== Final Results: Schedulability Analysis ===")
    print(f"\nOverall System Schedulability: {'YES' if is_schedulable else 'NO'}")


if __name__ == "__main__":
    # --- For detailed, step-by-step debugging (manual run) ---
    # Uncomment the line below to run a single verbose simulation.
    # You can change the parameters here to test specific scenarios.
    #print("\n--- Running a single verbose simulation for debugging ---")
    # For a verbose simulation, you might still want to fix the number of nodes for repeatable debugging.
    # So, num_nodes_per_task_val is provided here.
    #run_single_verbose_simulation(U_norm_val=0.5, num_tasks_val=2, num_resources_val=3, density_val=0.3,
    #                              num_nodes_per_task_val=10)
    #print("\n--- End of single verbose simulation ---")

    # --- For running all scenarios (statistical results) ---
    # Comment out run_single_verbose_simulation() above and uncomment the lines below.
    print("\n--- Running all scenarios (may take time) ---")
    run_all_scenarios(num_runs_per_scenario=200)
    print("\n--- All scenarios finished ---")