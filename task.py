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
        self.U = self.wcet / self.period  # Utilization


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


def erdos_renyi_graph(n: int, p: float) -> tuple[list[int], list[tuple[int, int]]]:

    nodes = list(range(1, n + 1))  # Node labels start from 1
    edges = []
    for i in nodes:
        for j in nodes:
            if i != j:  # Avoid self-loops
                if random.random() < p:  # Create an edge with probability p
                    edges.append((i, j))
    return nodes, edges

def get_critical_path(nodes: list[Node], edges: list[Edge]) -> int:
    dp: dict[str, int] = {}
    degree_in: dict[str, int] = {}

    # محاسبه درجه ورودی گره‌ها
    for edge in edges:
        degree_in[edge.sink.id] = degree_in.get(edge.sink.id, 0) + 1

    # پیدا کردن گره‌های منبع
    sources = [node for node in nodes if degree_in.get(node.id, 0) == 0]
    if not sources:
        raise ValueError("No source node found. The graph may not be a DAG.")

    # مقداردهی اولیه برای منابع
    for node in sources:
        dp[node.id] = node.wcet

    # پردازش گره‌ها
    while sources:
        node = sources.pop()
        for edge in edges:
            if edge.src.id == node.id:
                degree_in[edge.sink.id] -= 1
                dp[edge.sink.id] = max(dp.get(edge.sink.id, 0), dp[node.id] + edge.sink.wcet)
                if degree_in.get(edge.sink.id, 0) == 0:
                    sources.append(edge.sink)

    if not dp:
        raise ValueError("No critical path could be determined. Check the graph structure.")

    return max(dp.values())

def __repr__(self):
        return f"Task(ID: {self.task_id}, C_i: {self.C_i}, Nodes: {self.num_nodes})"


def generate_resources(resource_count: int) -> list[Resource]:
    resources = [Resource(id=f"R{i + 1}") for i in range(resource_count)]
    return resources


def generate_task(task_id: int, resources: list[Resource]) -> Task:
    # تنظیم تعداد گره‌ها و احتمال ایجاد یال
    num_nodes = random.randint(5, 10)  # تعداد گره‌ها
    edge_probability = 0.2  # احتمال یال

    # ایجاد گره‌ها و یال‌ها
    graph_nodes, graph_edges = erdos_renyi_graph(num_nodes, edge_probability)

    nodes: list[Node] = []

    critical_nodes_count = rand.randint(1, min(16, len(graph_nodes)))
    critical_nodes = rand.sample(graph_nodes, k=critical_nodes_count)

    for node_id in graph_nodes:
        wcet = rand.randint(13, 30)

        node_resources = []
        critical_st = []
        critical_en = []

        if node_id in critical_nodes:
            m = rand.randint(1, wcet // 2)
            node_resources = rand.choices(resources, k=m)
            random_list = rand.sample(range(0, wcet + 1), 2 * m)
            random_list = sorted(random_list)
            critical_st = [i for idx, i in enumerate(random_list) if idx % 2 == 0]
            critical_en = [i for idx, i in enumerate(random_list) if idx % 2 == 1]

        nodes.append(Node(id=f"T{task_id}-J{node_id}", wcet=wcet, resources=node_resources,
                          critical_st=critical_st, critical_en=critical_en))

    edges: list[Edge] = []
    for edge in graph_edges:
        src = nodes[edge[0] - 1]
        sink = nodes[edge[1] - 1]
        edges.append(Edge(src=src, sink=sink))

    critical_path = get_critical_path(nodes, edges)
    x = 1
    while x < critical_path:
        x *= 2

    helper = get_critical_path(nodes, edges)
    period = int(helper * rand.uniform(0.125, 0.25))
    wcet = sum([node.wcet for node in nodes])

    # تعیین deadline (برای سادگی برابر با period)
    deadline = period

    return Task(id=f"T{task_id}", period=period, wcet=wcet, deadline=deadline, nodes=nodes, edges=edges)


def generate_tasks(resources: list[Resource], task_count: int) -> list[Task]:
    tasks = []
    for i in range(task_count):

        tasks.append(generate_task(i, resources))

    return tasks

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


