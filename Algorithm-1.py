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
    allocations: Dict[str, list]
    execution_times: Dict[str, int]

    def get_total_wcet(self) -> int:
        return sum(self.execution_times.get(node, 0) for node in self.nodes if node not in ["source", "sink"])

    def utilization(self) -> float:
        total_node_time = sum(sum(section[1] if isinstance(section, tuple) else section for section in sections)
                              for sections in self.allocations.values())
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
def erdos_renyi_graph(num_nodes_for_task: int, density: float):
    # Ensure a minimum of 2 nodes for actual task logic (excluding source/sink)
    if num_nodes_for_task < 2:
        num_nodes_to_generate = 2
    else:
        num_nodes_to_generate = num_nodes_for_task

    G = nx.erdos_renyi_graph(num_nodes_to_generate, density, directed=True)
    G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])  # Ensure DAG
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

    # If after adding source/sink, a node has become isolated, ensure it's connected
    for node in list(G.nodes()):
        if node not in [source_node, sink_node]:
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                # If isolated, connect from source and to sink
                G.add_edge(source_node, node)
                G.add_edge(node, sink_node)

    return G


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
    except (nx.NetworkXNoPath, nx.NodeNotFound):
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


def generate_accesses_and_lengths(num_tasks: int, num_resources: int) -> tuple[dict, dict]:
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
    nodes = [node for node in task_info["nodes"] if node not in ["source", "sink"]]

    allocations = {node_id: [] for node_id in nodes}
    execution_times_allocated = task_info["execution_times"].copy()

    for node_id in nodes:
        original_wcet = task_info["execution_times"][node_id]

        node_resource_sections = []
        for resource, task_access_counts in accesses.items():
            if task_id - 1 < len(task_access_counts) and task_access_counts[task_id - 1] > 0:
                node_access_lengths_for_resource = list(lengths[resource][task_id - 1])  # Use a copy for pop operations

                num_accesses_for_node = random.randint(0, len(node_access_lengths_for_resource))

                for _ in range(num_accesses_for_node):
                    if node_access_lengths_for_resource:
                        access_length = node_access_lengths_for_resource.pop(0)
                        node_resource_sections.append((resource, access_length))

        total_resource_time_for_node = sum(sec[1] for sec in node_resource_sections)
        remaining_normal_time = max(0, original_wcet - total_resource_time_for_node)

        final_allocation_for_node = []
        num_segments = len(node_resource_sections) + 1

        normal_segment_lengths = [0] * num_segments
        if remaining_normal_time > 0:
            if num_segments > 1:
                # Distribute remaining time among segments
                points = sorted(random.sample(range(remaining_normal_time + num_segments - 1), num_segments - 1))
                points = [0] + points + [remaining_normal_time + num_segments - 1]
                normal_segment_lengths = [points[i + 1] - points[i] for i in range(num_segments)]
            else:  # Only one normal segment
                normal_segment_lengths = [remaining_normal_time]

        # Interleave normal and resource sections
        for i in range(num_segments):
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


def generate_base_task(task_id: int, accesses_data: dict, lengths_data: dict, num_nodes_for_task: int,
                       density: float) -> Optional[BaseTask]:
    G = erdos_renyi_graph(num_nodes_for_task, density)
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

    if critical_path_length == 0:
        if actual_nodes:  # If there are actual nodes but no path, sum their WCET
            critical_path_length = sum(execution_times_final.values())
        if critical_path_length == 0: critical_path_length = 10  # Minimum to avoid div by zero

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
    return base_task


def generate_tasks(num_base_tasks: int, num_resources: int, num_nodes_per_task: int, density: float) -> List[BaseTask]:
    tasks_list = []
    accesses_data, lengths_data = generate_accesses_and_lengths(num_base_tasks, num_resources)

    for i in range(1, num_base_tasks + 1):
        task = generate_base_task(task_id=i, accesses_data=accesses_data, lengths_data=lengths_data,
                                  num_nodes_for_task=num_nodes_per_task, density=density)
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
        if hyper_period % task.period == 0 and num_instances > 0:  # Ensure we don't double count the last period's start
            num_instances -= 1

        for i in range(num_instances + 1):
            release_t = task.period * i
            abs_deadline = release_t + task.deadline

            instance_nodes = list(task.nodes)
            instance_edges = list(task.edges)
            instance_execution_times = task.execution_times.copy()
            instance_allocations = {k: list(v) for k, v in task.allocations.items()}

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


def calculate_asap_cores(nodes: List[str], edges: List[Tuple[str, str]], execution_times: Dict[str, int]) -> Tuple[
    Dict[str, int], int]:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    asap_schedule = {}

    for node in nx.topological_sort(G):
        if node == "source":
            asap_schedule[node] = 0
        else:
            pred_end_times = [asap_schedule[pred] + execution_times.get(pred, 0) for pred in G.predecessors(node) if
                              pred in asap_schedule]
            asap_schedule[node] = max(pred_end_times) if pred_end_times else 0

    max_parallel_tasks = 0
    active_nodes_count = {}

    # Collect all unique start and end times
    all_event_times = set()
    for node in nodes:
        if node not in ["source", "sink"]:
            if node in asap_schedule:
                all_event_times.add(asap_schedule[node])
                all_event_times.add(asap_schedule[node] + execution_times.get(node, 0))
    all_event_times = sorted(list(all_event_times))

    current_active_nodes = set()
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


def federated_scheduling(base_tasks: List[BaseTask], U_norm: float) -> List[Processor]:
    total_processors = calculate_total_processors(base_tasks, U_norm)
    processors = [Processor(id=i + 1, assigned_tasks=[]) for i in range(total_processors)]

    high_util_tasks = []
    low_util_tasks = []

    for task in base_tasks:
        if task.utilization() > 1.0:
            high_util_tasks.append(task)
        else:
            low_util_tasks.append(task)

    high_util_tasks.sort(key=lambda t: t.utilization(), reverse=True)

    for task in high_util_tasks:
        _, max_parallel_cores_for_task = calculate_asap_cores(task.nodes, task.edges, task.execution_times)

        if max_parallel_cores_for_task == 0 and len([n for n in task.nodes if n not in ["source", "sink"]]) > 0:
            max_parallel_cores_for_task = 1

        if max_parallel_cores_for_task > 0 and total_processors >= max_parallel_cores_for_task:
            assigned_count = 0
            for p in processors:
                if not p.assigned_tasks:
                    p.assigned_tasks.append(task.id)
                    p.utilization += task.utilization() / max_parallel_cores_for_task
                    assigned_count += 1
                    if assigned_count == max_parallel_cores_for_task:
                        break
            total_processors -= max_parallel_cores_for_task
        else:
            # If not enough processors for high util task, push to low util for WFD if possible
            low_util_tasks.append(task)

    low_util_tasks.sort(key=lambda t: t.utilization(), reverse=True)

    for task in low_util_tasks:
        available_processors_for_wfd = sorted([p for p in processors if p.utilization <= 1.0],
                                              key=lambda p: 1.0 - p.utilization,
                                              reverse=True)

        assigned = False
        for processor in available_processors_for_wfd:
            if processor.utilization + task.utilization() <= 1.0:
                processor.assigned_tasks.append(task.id)
                processor.utilization += task.utilization()
                assigned = True
                break
    return processors


# --- EDF Scheduler Core ---

def find_ready_nodes(nodes: list, edges: list[tuple], completed_nodes: set) -> list:
    ready_nodes = []
    for node in nodes:
        if node == "sink" or node in completed_nodes:
            continue
        predecessors_of_node = [src for src, dest in edges if dest == node]
        if not predecessors_of_node or all(pred in completed_nodes for pred in predecessors_of_node):
            ready_nodes.append(node)
    return ready_nodes


def schedule_multiple_tasks_periodic(base_tasks: List[BaseTask], system_processors: List[Processor],
                                     max_simulation_time_factor: int = 50) -> bool:
    hyper_period = hyperperiod_lcm(base_tasks)

    # Calculate a sensible maximum simulation time based on hyperperiod and task WCETs
    if not base_tasks:
        max_time_estimate = 1000  # Default if no tasks
    else:
        # Sum of all WCETs in the system for a single iteration, multiplied by a factor and number of processors
        total_wcet_sum = sum(t.wcet for t in base_tasks)
        # Max simulation time should be enough for hyperperiod and to clear any backlog + a safety margin
        max_time_estimate = hyper_period * max_simulation_time_factor
        # Ensure it's not excessively small if hyperperiod is tiny
        if max_time_estimate < total_wcet_sum * 2:
            max_time_estimate = total_wcet_sum * 2

    MAX_SIMULATION_TIME = int(max_time_estimate)

    all_instances: List[TaskInstance] = generate_periodic_task_instances(base_tasks, hyper_period)

    resource_manager = ResourceManager()
    resource_manager.initialize_resources([f"R{i + 1}" for i in range(32)])  # Max 6 resources

    current_time = 0
    active_jobs_pq: List[Tuple[int, int, str, TaskInstance]] = []
    heapq.heapify(active_jobs_pq)

    executing_nodes: Dict[int, Tuple[str, str, int, int, int, TaskInstance]] = {}

    job_map: Dict[str, TaskInstance] = {instance.instance_id: instance for instance in all_instances}

    overall_schedulable = True

    while True:
        # --- Time Limit Check ---
        if current_time > MAX_SIMULATION_TIME:
            # print(f"Simulation exceeded MAX_SIMULATION_TIME ({MAX_SIMULATION_TIME}). Declaring unschedulable.")
            overall_schedulable = False
            break

        # 1. Release new job instances
        for instance in all_instances:
            if instance.release_time == current_time and not instance.is_completed and not instance.deadline_missed:
                if not any(job_tuple[2] == instance.instance_id for job_tuple in active_jobs_pq):
                    heapq.heappush(active_jobs_pq,
                                   (instance.absolute_deadline, instance.release_time, instance.instance_id, instance))

        # 2. Check for missed deadlines among active and executing jobs
        jobs_to_remove_from_active_pq = []
        for deadline, release, inst_id, job_obj in list(active_jobs_pq):
            if current_time > job_obj.absolute_deadline and not job_obj.is_completed and not job_obj.deadline_missed:
                job_obj.deadline_missed = True
                job_obj.is_completed = True
                overall_schedulable = False

                for proc_id, exec_data in list(executing_nodes.items()):
                    if exec_data[0] == inst_id:
                        del executing_nodes[proc_id]
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

        if all_jobs_processed and not executing_nodes and current_time >= hyper_period:
            break

        if not executing_nodes and not active_jobs_pq and current_time >= hyper_period:
            break

        # 3. Select jobs/nodes to execute for this time step (EDF & CPC)
        available_processors = [p.id for p in system_processors if p.id not in executing_nodes]
        candidates_for_scheduling = []

        nodes_currently_running_or_selected = set()
        for _, node_id_exec, _, _, _, _ in executing_nodes.values():
            nodes_currently_running_or_selected.add(node_id_exec)

        for _, _, _, job_obj in active_jobs_pq:
            if job_obj.is_completed or job_obj.deadline_missed:
                continue

            processors_assigned_to_this_job_base_task = []
            for p in system_processors:
                if job_obj.task_id in p.assigned_tasks:
                    processors_assigned_to_this_job_base_task.append(p.id)

            suitable_and_available_processors_for_this_job = [p_id for p_id in available_processors if
                                                              p_id in processors_assigned_to_this_job_base_task]

            if not suitable_and_available_processors_for_this_job:
                continue

            ready_nodes_for_job = find_ready_nodes(job_obj.nodes, job_obj.edges, job_obj.completed_nodes)
            actual_ready_nodes = [n for n in ready_nodes_for_job if n not in ["source", "sink"]]

            if not actual_ready_nodes:
                continue

            critical_path_for_job = job_obj.critical_path
            edges_for_job = job_obj.edges

            critical_ready_nodes = [node for node in actual_ready_nodes if node in critical_path_for_job]
            non_critical_ready_nodes = [node for node in actual_ready_nodes if node not in critical_path_for_job]

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

            prioritized_nodes_for_job = critical_ready_nodes + critical_predecessors + non_critical_others

            for node_to_schedule_candidate in prioritized_nodes_for_job:
                if node_to_schedule_candidate in nodes_currently_running_or_selected:
                    continue

                processor_to_assign = None
                if suitable_and_available_processors_for_this_job:
                    processor_to_assign = suitable_and_available_processors_for_this_job[0]

                if processor_to_assign is None:
                    continue

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
                        continue

                if can_acquire_initial_resource:
                    candidates_for_scheduling.append(
                        (job_obj.absolute_deadline, job_obj, node_to_schedule_candidate, processor_to_assign))
                    nodes_currently_running_or_selected.add(node_to_schedule_candidate)
                    available_processors.remove(processor_to_assign)
                    break

        candidates_for_scheduling.sort(key=lambda x: x[0])

        for _, job_obj_selected, node_to_schedule_selected, processor_to_assign_selected in candidates_for_scheduling:
            if processor_to_assign_selected not in executing_nodes:
                first_section = job_obj_selected.allocations[node_to_schedule_selected][0]
                section_time = first_section[1] if isinstance(first_section, tuple) else first_section
                section_idx = 0

                executing_nodes[processor_to_assign_selected] = (
                    job_obj_selected.instance_id, node_to_schedule_selected, section_time, section_idx, current_time,
                    job_obj_selected)

        nodes_to_remove_from_executing_this_cycle = []

        for processor_id, (inst_id, node_id, remaining_orig, section_idx_orig, start_time_orig, job_obj) in list(
                executing_nodes.items()):
            node_allocations = job_obj.allocations.get(node_id, [])

            if section_idx_orig >= len(node_allocations):
                nodes_to_remove_from_executing_this_cycle.append(processor_id)
                continue

            current_section_def = node_allocations[section_idx_orig]
            current_remaining = remaining_orig

            progress_made_this_cycle = True

            if isinstance(current_section_def, tuple) and current_section_def[0] != "Normal":
                resource_id = current_section_def[0]

                if not resource_manager.is_resource_locked_by(node_id, resource_id):
                    if not resource_manager.request_resource(node_id, resource_id):
                        progress_made_this_cycle = False
                        executing_nodes[processor_id] = (
                            inst_id, node_id, remaining_orig, section_idx_orig, start_time_orig, job_obj)
                        continue

                if current_remaining == 1:
                    if resource_manager.is_resource_locked_by(node_id, resource_id):
                        resource_manager.release_resource(node_id, resource_id)

            if progress_made_this_cycle:
                current_remaining -= 1

            if current_remaining <= 0:
                job_obj.completed_nodes.add(node_id)

                if section_idx_orig + 1 < len(node_allocations):
                    next_section_def = node_allocations[section_idx_orig + 1]
                    next_section_time = next_section_def[1] if isinstance(next_section_def, tuple) else next_section_def

                    executing_nodes[processor_id] = (
                        inst_id, node_id, next_section_time, section_idx_orig + 1, current_time + 1, job_obj)
                else:
                    resource_manager.release_all_resources_for_node(node_id)
                    job_obj.is_completed = True
                    nodes_to_remove_from_executing_this_cycle.append(processor_id)

            else:
                executing_nodes[processor_id] = (
                    inst_id, node_id, current_remaining, section_idx_orig, start_time_orig, job_obj)

        for proc_id_to_remove in nodes_to_remove_from_executing_this_cycle:
            if proc_id_to_remove in executing_nodes:
                del executing_nodes[proc_id_to_remove]

        current_time += 1

    for inst in all_instances:
        if not inst.is_completed or inst.deadline_missed:
            overall_schedulable = False
            break

    return overall_schedulable


def run_single_simulation(U_norm_val: float, num_tasks_val: int, num_resources_val: int, density_val: float) -> bool:
    """
    Runs a single simulation with specified parameters and returns True if schedulable, False otherwise.
    """
    # Generate Base Tasks
    # num_nodes_per_task: A reasonable heuristic might be to relate it to num_tasks_val, e.g., random from (5, 20)
    # or (num_tasks_val * 2, num_tasks_val * 5) for more complex tasks. Let's keep a reasonable range.
    num_nodes_for_each_task = random.randint(5, 20)  # Keep a consistent node count range for tasks

    base_tasks_list = generate_tasks(num_base_tasks=num_tasks_val, num_resources=num_resources_val,
                                     num_nodes_per_task=num_nodes_for_each_task, density=density_val)

    if not base_tasks_list:
        return False

    # Allocate Processors
    system_processors = federated_scheduling(base_tasks_list, U_norm_val)

    # Schedule all Instances of Periodic Tasks
    # max_simulation_time_factor: A larger value for more complex scenarios or smaller periods
    # Defaulting to 50, but might need tuning if hyperperiods are very large
    is_schedulable = schedule_multiple_tasks_periodic(base_tasks_list, system_processors, max_simulation_time_factor=50)

    return is_schedulable


def run_all_scenarios(num_runs_per_scenario: int = 100):
    """
    Runs simulations for all specified scenarios.
    """
    U_norm_values = [0.1, 0.5, 0.7, 1.0]
    num_tasks_values = [2, 4, 8, 16]
    num_resource_values = [2, 4, 8, 16, 32]
    density_values = [0, 0.1, 0.3, 0.7, 0.9, 1]

    default_U_norm = 0.5
    default_num_tasks = 2
    default_num_resources = 6
    default_density = 0.1  # graph density

    scenario_count = 0

    print("=== شروع اجرای سناریوهای زمان‌بندی‌پذیری ===")

    # Scenario 1: Vary U_norm
    '''for U_norm_val in U_norm_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: U_norm = {U_norm_val} ---")
        print(
            f"پارامترهای ثابت: Num_tasks={default_num_tasks}, Num_resources={default_num_resources}, Density={default_density}")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            # No random seed for independent trials
            is_current_run_schedulable = run_single_simulation(
                U_norm_val=U_norm_val,
                num_tasks_val=default_num_tasks,
                num_resources_val=default_num_resources,
                density_val=default_density
            )
            if is_current_run_schedulable:
                schedulable_count += 1
            # print(f"  اجرای {i+1}/{num_runs_per_scenario}: {'زمان‌بندی‌پذیر' if is_current_run_schedulable else 'غیرزمان‌بندی‌پذیر'}")

        print(f"خلاصه سناریو U_norm = {U_norm_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    # Scenario 2: Vary Num_tasks
    for num_tasks_val in num_tasks_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: Num_tasks = {num_tasks_val} ---")
        print(
            f"پارامترهای ثابت: U_norm={default_U_norm}, Num_resources={default_num_resources}, Density={default_density}")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            is_current_run_schedulable = run_single_simulation(
                U_norm_val=default_U_norm,
                num_tasks_val=num_tasks_val,
                num_resources_val=default_num_resources,
                density_val=default_density
            )
            if is_current_run_schedulable:
                schedulable_count += 1
            # print(f"  اجرای {i+1}/{num_runs_per_scenario}: {'زمان‌بندی‌پذیر' if is_current_run_schedulable else 'غیرزمان‌بندی‌پذیر'}")

        print(f"خلاصه سناریو Num_tasks = {num_tasks_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    # Scenario 3: Vary Num_resource
    for num_resources_val in num_resource_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: Num_resources = {num_resources_val} ---")
        print(f"پارامترهای ثابت: U_norm={default_U_norm}, Num_tasks={default_num_tasks}, Density={default_density}")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            is_current_run_schedulable = run_single_simulation(
                U_norm_val=default_U_norm,
                num_tasks_val=default_num_tasks,
                num_resources_val=num_resources_val,
                density_val=default_density
            )
            if is_current_run_schedulable:
                schedulable_count += 1
            # print(f"  اجرای {i+1}/{num_runs_per_scenario}: {'زمان‌بندی‌پذیر' if is_current_run_schedulable else 'غیرزمان‌بندی‌پذیر'}")

        print(f"خلاصه سناریو Num_resources = {num_resources_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)'''

    # Scenario 4: Vary Density
    for density_val in density_values:
        scenario_count += 1
        print(f"\n--- سناریو {scenario_count}: Density = {density_val} ---")
        print(
            f"پارامترهای ثابت: U_norm={default_U_norm}, Num_tasks={default_num_tasks}, Num_resources={default_num_resources}")

        schedulable_count = 0
        for i in range(num_runs_per_scenario):
            is_current_run_schedulable = run_single_simulation(
                U_norm_val=default_U_norm,
                num_tasks_val=default_num_tasks,
                num_resources_val=default_num_resources,
                density_val=density_val
            )
            if is_current_run_schedulable:
                schedulable_count += 1
            # print(f"  اجرای {i+1}/{num_runs_per_scenario}: {'زمان‌بندی‌پذیر' if is_current_run_schedulable else 'غیرزمان‌بندی‌پذیر'}")

        print(f"خلاصه سناریو Density = {density_val}:")
        print(f"  تعداد اجراهای زمان‌بندی‌پذیر: {schedulable_count} از {num_runs_per_scenario}")
        print(f"  درصد زمان‌بندی‌پذیری: {(schedulable_count / num_runs_per_scenario) * 100:.2f}%")
        print("-" * 50)

    print("\n=== پایان اجرای تمامی سناریوها ===")


if __name__ == "__main__":
    run_all_scenarios(num_runs_per_scenario=20)