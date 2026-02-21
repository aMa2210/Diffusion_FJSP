import json
import random
import os
from typing import List, Dict


def generate_random_fjsp_problem(
        filename: str,
        num_machines: int,
        num_workpieces: int,
        min_ops: int = 4,
        max_ops: int = 8,
        min_opts: int = 1,
        max_opts: int = 4,
        min_time: int = 8,
        max_time: int = 15,
        seed: int = None
):
    """
    生成随机的 FJSP 问题定义并保存为 JSON 文件。

    参数:
    - filename: 保存路径 (例如 "TestSet/random_10_machines.json")
    - num_machines: 机器总数 (例如 10)
    - num_workpieces: 工件总数 (例如 20)
    - min_ops / max_ops: 每个工件包含的工序数量范围
    - min_opts / max_opts: 每个工序可选机器的数量范围 (柔性程度)
    - min_time / max_time: 加工时间的范围
    - seed: 随机种子 (保证可复现)
    """
    if seed is not None:
        random.seed(seed)

    # 1. 生成机器列表 (ID 从 1 开始)
    machines = list(range(1, num_machines + 1))

    workpieces_data = []

    for i in range(1, num_workpieces + 1):
        wp_name = f"Workpiece{i}"

        # 随机决定该工件有多少道工序
        current_num_ops = random.randint(min_ops, max_ops)

        optional_machines = []
        processing_times = []

        for _ in range(current_num_ops):
            # 随机决定该工序有几个可选机器 (不能超过机器总数)
            num_options = random.randint(min_opts, min(max_opts, num_machines))

            # 随机选择具体的机器 (无放回采样)
            selected_machines = sorted(random.sample(machines, num_options))

            # 为每台选中的机器生成对应的加工时间
            # (通常假设不同机器加工同一工序的时间是相近的，但也允许波动)
            times = [random.randint(min_time, max_time) for _ in range(num_options)]

            optional_machines.append(selected_machines)
            processing_times.append(times)

        workpieces_data.append({
            "name": wp_name,
            "optional_machines": optional_machines,
            "processing_time": processing_times
        })

    problem_def = {
        "machines": machines,
        "workpieces": workpieces_data
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(problem_def, f, indent=2)

    print(f"✅ Problem instance saved to: {filename} with {num_machines} machines and {num_workpieces} workpieces")


if __name__ == "__main__":
    generate_random_fjsp_problem(
        filename="TestSet/random_simple.json",
        num_machines=8,
        num_workpieces=5,
        min_ops=6, max_ops=6,  # 固定每个工件 6 道工序
        min_opts=2, max_opts=4,  # 每个工序 2-4 个可选机器
        seed=42
    )

    generate_random_fjsp_problem(
        filename="TestSet/random_hard.json",
        num_machines=20,  # 20 台机器
        num_workpieces=50,  # 50 个工件
        min_ops=10, max_ops=20,  # 每个工件 10-20 道工序 (总节点数可能破千)
        min_opts=1, max_opts=5,  # 有些工序可能只有 1 个机器可选 (瓶颈)
        min_time=10, max_time=100,  # 时间跨度大
        seed=100
    )