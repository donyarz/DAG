from typing import List, Tuple, Dict
import random as rand
import random
import math
import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt


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
    def __init__(self, id: int, period: int, wcet: int, deadline: int, nodes: list, edges: list):
        self.id = id
        self.period = period
        self.wcet = wcet
        self.deadline = deadline
        self.nodes = nodes
        self.edges = edges
        #self.U = self.wcet / self.period  # Utilization


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
    mapping = {node: node + 1 for node in G.nodes()}  # افزایش شماره نودها
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
    pos = nx.spring_layout(G, seed=42)  # تعیین موقعیت نودها
    nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", node_size=700, font_size=10,
            edge_color="gray")
    plt.title(f"Task {task['task_id']} DAG", fontsize=14)
    plt.show()


def get_critical_path(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    # ایجاد ساختار برای نگهداری درجه ورودی و زمان
    dp = {node: 0 for node in nodes}  # حداکثر زمان رسیدن به هر نود
    degree_in = {node: 0 for node in nodes}  # درجه ورودی هر نود

    # محاسبه درجه ورودی نودها
    for src, sink in edges:
        degree_in[sink] += 1

    # یافتن نودهای منبع (درجه ورودی صفر)
    sources = [node for node in nodes if degree_in[node] == 0]

    # پردازش گراف برای محاسبه زمان حداکثر برای هر نود
    queue = sources[:]
    while queue:
        current = queue.pop(0)  # نود فعلی
        for src, sink in edges:
            if src == current:  # بررسی اینکه یال از نود فعلی خارج شده باشد
                dp[sink] = max(dp[sink], dp[current] + 1)  # بروزرسانی حداکثر زمان مسیر
                degree_in[sink] -= 1  # کاهش درجه ورودی
                if degree_in[sink] == 0:
                    queue.append(sink)

    # طول مسیر بحرانی
    return max(dp.values())


def __repr__(self):
        return f"Task(ID: {self.task_id}, C_i: {self.C_i}, Nodes: {self.num_nodes})"


def generate_resources(resource_count: int) -> list[Resource]:
    resources = [Resource(id=f"R{i + 1}") for i in range(resource_count)]
    return resources

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


def generate_task(task_id: int, accesses: dict, lengths: dict) -> dict:
    nodes, edges = erdos_renyi_graph()
    # بررسی اینکه آیا فقط 'source' و 'sink' در گراف وجود دارند یا خیر
    if len(nodes) <= 2:  # یعنی فقط 'source' و 'sink' وجود دارند
        print(f"Skipping Task {task_id}: No nodes other than source and sink.")
        return None  # می‌توانید به دلخواه None یا یک تسک خالی برگردانید

    execution_times = {node: random.randint(13, 30) for node in nodes if node not in ["source", "sink"]}
    critical_path_length = get_critical_path(nodes, edges)
    total_execution_time = sum(execution_times.values())
    period = int(total_execution_time * rand.uniform(0.125, 0.25))
    deadline = period
    asap_schedule, max_parallel_tasks = calculate_asap_cores(nodes, edges, execution_times)

    return {
        "task_id": task_id,
        "nodes": nodes,
        "edges": edges,
        "execution_times": execution_times,
        "total_execution_time": total_execution_time,
        "period": period,
        "deadline": deadline,
        "accesses": accesses,  # اطلاعات منابع و دسترسی
        "lengths": lengths,
        "Critical Path Length": critical_path_length,
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
    execution_times = task["execution_times"]  # مقادیر ثابت از تسک

    for node in nodes:
        execution_time = execution_times[node]  # زمان اجرای نود ثابت و از قبل تعریف شده است
        critical_sections = []  # سکشن‌های کریتیکال
        normal_sections = []  # سکشن‌های نرمال

        # تخصیص منابع مختص به این تسک
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





class algorithm:

    def calculate_total_processors(tasks):
        U_norm = rand.uniform([0.1,1])
        U_sum = sum(task.U for task in tasks)  # مجموع بهره‌وری کل وظایف
        m_total = math.ceil(U_sum / U_norm)  # فرمول تعداد کل پردازنده‌ها
        return m_total

def calculate_asap_cores(nodes: List[int], edges: List[Tuple[int, int]], execution_times: Dict[int, int]) -> Tuple[Dict[int, int], int]:
    """
    محاسبه ASAP برای هر نود با توجه به زمان‌های اجرایی هر نود و برگرداندن تعداد هسته‌های موردنیاز.
    """
    # ساخت گراف جهت‌دار از نودها و یال‌ها
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # محاسبه ASAP (زمان شروع و پایان)
    asap_schedule = {}
    end_times = {}

    for node in nx.topological_sort(G):
        if node == "source":
            # زمان شروع "source" همیشه صفر است
            asap_schedule[node] = 0
            end_times[node] = 0  # زمان پایان "source" نیز 0 است
        else:
            # زمان شروع هر نود برابر است با بیشترین زمان پایان از نودهای پیشنیاز + 1
            start_time = max(
                [asap_schedule[pred] + execution_times.get(pred, 0) for pred in G.predecessors(node)],
                default=0
            )
            asap_schedule[node] = start_time
            # محاسبه زمان پایان نود جاری (start + execution_time)
            end_times[node] = start_time + execution_times.get(node, 0)

    # محاسبه تعداد هسته‌ها بر اساس زمان‌بندی ASAP
    max_parallel_tasks = 0
    time_slots = {}

    for node, start_time in asap_schedule.items():
        if node not in ("source", "sink"):
            # در این قسمت زمان‌های آغاز هر نود و تعداد نودهای همزمان در آن زمان را محاسبه می‌کنیم
            time_slots.setdefault(start_time, []).append(node)
            # محاسبه حداکثر تعداد نودهایی که در یک زمان همزمان اجرا می‌شوند
            max_parallel_tasks = max(max_parallel_tasks, len(time_slots[start_time]))

    return asap_schedule, max_parallel_tasks

def federated_scheduling(tasks):
    scheduling_result = []

    for task in tasks:
        # محاسبه U_i
        total_execution_time = task["total_execution_time"]
        period = task["period"]
        U_i = total_execution_time / period

        _, max_parallel_tasks = calculate_asap_cores(task["nodes"], task["edges"], task["execution_times"])

        if U_i > 1:
            num_processors = max_parallel_tasks  # تعداد هسته‌ها باید برابر max_parallel_tasks باشد
        else:
            num_processors = 1

        scheduling_result.append({
            "task_id": task["task_id"],
            "U_i": U_i,
            "num_processors": num_processors
        })

    return scheduling_result


