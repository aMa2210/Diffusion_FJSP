"""
读取 milp_expert_batches 目录下指定的 .pt 文件。

每个 .pt 文件由 torch.save 保存，内容为 list[dict]，每个 dict 为一条专家数据，包含：
  - problem_file: str  实例文件名
  - machine_ids: list  机器 ID 列表
  - expert_edges: Tensor [2, num_ops]  (工序节点索引, 机器节点索引)
  - expert_priorities: Tensor [num_ops]  工序优先级
  - makespan: float  完工时间

在下方配置要读取的文件名，然后直接运行: python read_milp_expert_batches.py
"""

from pathlib import Path

import torch

# ---------- 在脚本中指定要读取的文件 ----------
BATCH_DIR = "milp_expert_batches"   # .pt 文件所在目录
PT_FILES = [                        # 要读取的 .pt 文件名
    "milp_expert_data_batch_0.pt",
    "milp_expert_data_batch_1.pt",
    # "milp_expert_data_batch_2.pt",
]
SHOW_FIRST_N = 10                    # 每个批次打印前 N 条详情，0 表示只打摘要


def load_batch_pt(path: Path, map_location=None):
    """加载单个 .pt 文件，返回 list[dict]。"""
    if map_location is None:
        map_location = "cpu"
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def summarize_batch(data: list, batch_label: str = ""):
    """打印一个批次的摘要。"""
    if not data:
        print(f"  {batch_label} 空批次")
        return
    n = len(data)
    problem_files = [d.get("problem_file", "?") for d in data]
    makespans = [d.get("makespan") for d in data]
    print(f"  {batch_label} 共 {n} 条")
    print(f"    问题文件示例: {problem_files[0]}" + (f" ... {problem_files[-1]}" if n > 1 else ""))
    if makespans and makespans[0] is not None:
        print(f"    makespan 范围: {min(makespans):.2f} ~ {max(makespans):.2f}")
    e0 = data[0]
    if "expert_edges" in e0:
        ee = e0["expert_edges"]
        print(f"    首条 expert_edges shape: {ee.shape}")
    if "expert_priorities" in e0:
        ep = e0["expert_priorities"]
        print(f"    首条 expert_priorities shape: {ep.shape}")


def show_entry(entry: dict, index: int = 0):
    """打印单条条目的关键字段。"""
    print(f"  [{index}] problem_file: {entry.get('problem_file')}")
    print(f"      machine_ids: {entry.get('machine_ids')}")
    print(f"      expert_edges: {entry.get('expert_edges')}")
    if entry.get("expert_priorities") is not None:
        p = entry["expert_priorities"]
        print(f"      expert_priorities: shape={p.shape}, min={p.min().item():.4f}, max={p.max().item():.4f}")
    print(f"      makespan: {entry.get('makespan')}")


def main():
    base = Path(BATCH_DIR)
    if not base.exists():
        print(f"目录不存在: {base}")
        return

    paths = [base / f for f in PT_FILES]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"在 {base} 下未找到指定的 .pt 文件: {PT_FILES}")
        return

    print(f"找到 {len(paths)} 个文件:")
    for p in paths:
        print(f"  {p}")
    print()

    all_loaded = []
    for path in paths:
        try:
            data = load_batch_pt(path)
            if not isinstance(data, list):
                print(f"  {path.name}: 内容不是 list，类型为 {type(data)}")
                continue
            label = path.name
            summarize_batch(data, label)
            if SHOW_FIRST_N > 0:
                for i, entry in enumerate(data[: SHOW_FIRST_N]):
                    show_entry(entry, i)
                if len(data) > SHOW_FIRST_N:
                    print(f"  ... 共 {len(data)} 条，仅显示前 {SHOW_FIRST_N} 条")
            all_loaded.append((path, data))
        except Exception as e:
            print(f"  {path.name}: 加载失败 - {e}")

    return all_loaded


if __name__ == "__main__":
    main()
