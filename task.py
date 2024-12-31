# Resource -> gere
# Nemoodar -> Percentage of missed deadlines


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


def get_critical_path(nodes: list[Node], edges: list[Edge]) -> list[Node]:
    dp: dict[str, int] = {}
    degree_in: dict[str, int] = {}
    for edge in edges:
        degree_in[edge.sink.id] = degree_in.get(edge.sink.id, 0) + 1
    sources = [node for node in nodes if degree_in.get(node.id, 0) == 0]
    for node in sources:
        dp[node.id] = node.wcet
    while sources:
        node = sources.pop()
        for edge in edges:
            if edge.src.id == node.id:
                degree_in[edge.sink.id] -= 1
                dp[edge.sink.id] = max(dp.get(edge.sink.id, 0), dp[node.id] + edge.sink.wcet)
                if degree_in.get(edge.sink.id, 0) == 0:
                    sources.append(edge.sink)
    return max(dp.values())

def __repr__(self):
        return f"Task(ID: {self.task_id}, C_i: {self.C_i}, Nodes: {self.num_nodes})"


def generate_resources(resource_count: int) -> list[Resource]:
    resources = [Resource(id=f"R{i + 1}") for i in range(resource_count)]
    return resources

import random
from typing import Dict, List


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


def generate_task(task_id: int, nodes: list[str], edges: list[tuple[str, str]]) -> dict:
    """
    تولید دیکشنری مربوط به هر تسک شامل نودها و یال‌ها.
    """
    return {
        "task_id": task_id,
        "nodes": nodes,
        "edges": edges
    }

def generate_tasks(resources: list[str], task_count: int) -> list[dict]:
    tasks = []
    for i in range(task_count):
        tasks.append(generate_task(i + 1, resources))
    return tasks


def allocate_resources_to_nodes(task: dict, task_id: int, accesses: dict, lengths: dict) -> tuple[dict, dict]:
    """
    تخصیص منابع به نودها با توجه به دسترسی‌ها و طول‌های مختص تسک مشخص.
    """
    nodes = [node for node in task["nodes"] if node != "source" and node != "sink"]

    allocations = {node: [] for node in nodes}
    execution_times = {}  # ذخیره زمان اجرای هر نود

    # تخصیص منابع به نودهای تسک
    for node_idx, node in enumerate(nodes):
        execution_time = random.randint(13, 30)  # زمان اجرای تصادفی برای نود
        execution_times[node] = execution_time  # ذخیره زمان اجرا
        critical_sections = []  # سکشن‌های کریتیکال
        normal_sections = []  # سکشن‌های نرمال

        # تخصیص منابع مختص به این تسک
        for resource, task_accesses in accesses.items():
            # بررسی دسترسی‌های مرتبط با تسک فعلی
            if task_accesses[task_id - 1] > 0:  # index تسک
                node_access_lengths = lengths[resource][task_id - 1]  # تخصیص طول‌ها

                # بررسی طول‌ها و تطابق با زمان اجرا
                while node_access_lengths and execution_time > 0:
                    access_time = node_access_lengths[0]
                    if execution_time >= access_time:
                        critical_sections.append((resource, access_time))
                        execution_time -= access_time
                        node_access_lengths.pop(0)
                    else:
                        break

        # محاسبه زمان نرمال سکشن‌ها
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
                    normal_sections.append(0)  # در صورت عدم وجود زمان باقی‌مانده

            # تخصیص باقی‌مانده به آخرین نرمال سکشن
            normal_sections.append(remaining_time)
        else:
            # وقتی که کریتیکال سکشنی وجود ندارد
            normal_sections.append(remaining_time)

        # ترکیب نرمال و کریتیکال سکشن‌ها
        allocation = []
        for i, critical in enumerate(critical_sections):
            allocation.append(("Normal", normal_sections[i]))
            allocation.append(critical)
        if normal_sections:
            allocation.append(("Normal", normal_sections[-1]))

        allocations[node] = allocation

    return allocations, execution_times




num_tasks = 3  # تعداد تسک‌ها
accesses, lengths = generate_accesses_and_lengths(num_tasks)  # یکبار تولید دسترسی‌ها و طول‌ها

tasks = []
for i in range(num_tasks):
    nodes, edges = erdos_renyi_graph()  # گراف برای هر تسک
    task = generate_task(i + 1, nodes, edges)
    task["accesses"] = accesses  # اضافه کردن دسترسی‌ها
    task["lengths"] = lengths    # اضافه کردن طول‌ها
    tasks.append(task)

# تخصیص منابع به نودها
for task in tasks:
    print(f"Task {task['task_id']}:")
    print(f"Nodes: {task['nodes']}")
    print(f"Edges: {task['edges']}")
    print(f"Accesses: {task['accesses']}")
    print(f"Lengths: {task['lengths']}")

    allocations, execution_times = allocate_resources_to_nodes(task, task["task_id"], accesses, lengths)

    print("\nAllocations and Execution Times:")
    for node, allocation in allocations.items():
        print(f"Node {node} (Execution Time: {execution_times[node]}): {allocation}")
    print("\n")


class algorithm:

    def calculate_total_processors(tasks):
        U_norm = rand.uniform([0.1,1])
        U_sum = sum(task.U for task in tasks)  # مجموع بهره‌وری کل وظایف
        m_total = math.ceil(U_sum / U_norm)  # فرمول تعداد کل پردازنده‌ها
        return m_total

    def federated_scheduling(tasks, total_processors):
        high_utilization_tasks = [task for task in tasks if task.U > 1]
        low_utilization_tasks = [task for task in tasks if task.U <= 1]

        high_utilization_allocations = {}
        for task in high_utilization_tasks:
            li = get_critical_path(task.nodes, task.edges)
            m_i = math.ceil((task.wcet - li) / (task.period - li))  # فرمول m_i
            high_utilization_allocations[task.id] = m_i

        low_utilization_allocations = {task.id: 1 for task in low_utilization_tasks}

        total_allocated_processors = sum(high_utilization_allocations.values()) + len(low_utilization_tasks)

        # بررسی زمان‌بندپذیری
        if total_allocated_processors > total_processors:
            return None, f"Tasks are not schedulable with {total_processors} processors."

        # ترکیب نتایج
        allocations = {**high_utilization_allocations, **low_utilization_allocations}
        return allocations


