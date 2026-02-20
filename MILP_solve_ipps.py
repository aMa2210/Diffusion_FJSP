import json
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

# import matplotlib.pyplot as plt  # 甘特图已关闭
import numpy as np
import pulp
import highspy
import torch

def load_instance(path):
    """
    读取由 `generate_random_fjsp_problem` 生成的 JSON 实例。

    返回:
    - machines: List[int]
    - workpieces: List[dict]，其中每个元素形如:
        {
          "name": str,
          "optional_machines": List[List[int]],   # len = 工序数
          "processing_time":   List[List[int]]    # 同形状，时间与机器一一对应
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["machines"], data["workpieces"]


def load_ga_makespan_lookup_one_batch(ga_expert_batches_dir: str, batch_id: int) -> dict:
    """
    仅加载指定批次的 ga_expert_data_batch_{batch_id}.pt，返回 problem_file -> makespan 的映射。
    按需调用，避免一次性加载所有 GA 批次占用过多内存。若文件不存在或读取出错则返回 {}。
    """
    batch_dir = Path(ga_expert_batches_dir)
    pt_file = batch_dir / f"ga_expert_data_batch_{batch_id}.pt"
    if not pt_file.exists():
        return {}
    try:
        batch = torch.load(pt_file, map_location="cpu")
        if not isinstance(batch, list):
            return {}
        lookup = {}
        for entry in batch:
            if isinstance(entry, dict) and "problem_file" in entry and "makespan" in entry:
                lookup[entry["problem_file"]] = float(entry["makespan"])
        return lookup
    except Exception as e:
        print(f"Warning: could not load {pt_file}: {e}")
        return {}


# def create_gantt_chart_from_milp(operations, title="MILP Gantt", filename=None):
#     """
#     根据 MILP 解中的 operations（含 job_name, machine, start, proc_time）绘制甘特图并保存。
#     operations 可为按 start 排序的 schedule，或任意含上述字段的列表。
#     """
#     if not operations:
#         return
#     fig, ax = plt.subplots(figsize=(14, 8))
#     raw_workpieces = list(set(op["job_name"] for op in operations))
#
#     def extract_number(text):
#         match = re.search(r"\d+", str(text))
#         return int(match.group()) if match else 0
#
#     workpieces = sorted(raw_workpieces, key=extract_number)
#     colors = plt.cm.tab20(np.linspace(0, 1, len(workpieces)))
#     color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}
#
#     for op in operations:
#         m_id = op["machine"]
#         wp = op["job_name"]
#         start = float(op["start"])
#         dur = float(op["proc_time"] or 0)
#         wp_num = extract_number(wp)
#         label_text = f"J{wp_num - 1}"
#         ax.barh(
#             y=m_id, width=dur, left=start,
#             height=0.6, align="center",
#             color=color_map.get(wp, "gray"), edgecolor="black", alpha=0.9,
#         )
#         ax.text(
#             start + dur / 2, m_id, label_text,
#             ha="center", va="center", color="white", fontweight="bold", fontsize=8,
#         )
#
#     machines = sorted(set(op["machine"] for op in operations))
#     ax.set_yticks(machines)
#     ax.set_yticklabels([f"M-{m}" for m in machines])
#     ax.set_ylabel("Machines")
#     ax.set_xlabel("Time")
#     ax.set_title(title)
#     ax.grid(True, axis="x", linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     if filename:
#         Path(filename).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(filename, dpi=150)
#         plt.close(fig)
#     else:
#         plt.show()


def build_and_solve_milp(
    instance_path: str,
    time_limit: int = None,
    msg: bool = True,
    gap_rel: float = None,
    threads: int = None,
    ga_makespan_ub: float = None,
    # gantt_path: str = None,  # 甘特图已关闭
):
    """
    针对一个 FJSP 实例构建 MILP 模型，求解最小完工时间（makespan）的调度。

    建模假设（标准柔性 Job-Shop / FJSP 形式）:
    - 每个工件内部工序按给定顺序依次加工。
    - 每道工序必须选择其候选机器集合里的恰好一台机器。
    - 同一时间每台机器最多加工一道工序。
    - 目标：最小化系统完工时间 C_max。

    - ga_makespan_ub: 若提供（如从 GA 的 ga_expert_batches 读取的 makespan），用作 Big-M 上界，可加快求解。
    """
    machines, workpieces = load_instance(instance_path)

    # 将工件索引化：job_id = 0..J-1
    num_jobs = len(workpieces)

    # 预处理：记录每个 (job, op) 的候选机器和对应加工时间
    # ops_per_job[j] = 工件 j 的工序数
    ops_per_job = [len(wp["optional_machines"]) for wp in workpieces]

    # cand_machines[(j, o)] = [m1, m2, ...]
    # proc_time[(j, o, m)] = p
    cand_machines = {}
    proc_time = {}

    for j, wp in enumerate(workpieces):
        opt_machs_list = wp["optional_machines"]
        proc_times_list = wp["processing_time"]
        for o, (mach_list, time_list) in enumerate(zip(opt_machs_list, proc_times_list)):
            cand_machines[(j, o)] = list(mach_list)
            for m, p in zip(mach_list, time_list):
                proc_time[(j, o, m)] = p

    # 为每台物理机器，列出所有可能在其上加工的工序 (j, o)
    ops_on_machine = {m: [] for m in machines}
    for (j, o), mach_list in cand_machines.items():
        for m in mach_list:
            ops_on_machine[m].append((j, o))

    # Big-M 上界：需不小于任意可行 makespan。若提供 GA 的 makespan 则用之（可收紧松弛、加快求解）；否则用“每工序取最慢机器”之和作为保守上界。
    total_max = 0
    for j, wp in enumerate(workpieces):
        for p_list in wp["processing_time"]:
            total_max += max(p_list) if p_list else 0
    if ga_makespan_ub is not None and ga_makespan_ub > 0:
        big_m = max(total_max, ga_makespan_ub)
    else:
        big_m = total_max
    # -------------------- 定义 MILP 模型 --------------------
    model = pulp.LpProblem("FJSP_Makespan_Minimization", pulp.LpMinimize)

    # 变量:
    # S[j, o]: 工件 j 的第 o 道工序的开始时间 (连续非负)
    S = {
        (j, o): pulp.LpVariable(f"S_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
        for o in range(ops_per_job[j])
    }

    # Y[j, o, m]: 如果工序 (j, o) 在机器 m 上加工，则为 1（仅当 m 是候选机器）
    Y = {
        (j, o, m): pulp.LpVariable(f"Y_{j}_{o}_{m}", lowBound=0, upBound=1, cat="Binary")
        for (j, o), mach_list in cand_machines.items()
        for m in mach_list
    }

    # C_max: 系统完工时间
    C_max = pulp.LpVariable("C_max", lowBound=0, cat="Continuous")

    # 对每台机器 m 上的每一对工序 a=(j1,o1), b=(j2,o2)，
    # 使用二元变量 Z[m, a, b] 表示在机器 m 上 a 是否先于 b。
    Z = {}
    for m in machines:
        ops_m = ops_on_machine[m]
        for (j1, o1), (j2, o2) in combinations(ops_m, 2):
            if j1 == j2:
                continue
            Z[(m, j1, o1, j2, o2)] = pulp.LpVariable(
                f"Z_{m}_{j1}_{o1}_{j2}_{o2}", lowBound=0, upBound=1, cat="Binary"
            )

    # -------------------- 目标函数 --------------------
    model += C_max, "Minimize_makespan"

    # -------------------- 约束 --------------------

    # 1) 每道工序必须在其候选机器中选择恰好一台机器
    for (j, o), mach_list in cand_machines.items():
        model += (
            pulp.lpSum(Y[(j, o, m)] for m in mach_list) == 1,
            f"Select_one_machine_job{j}_op{o}",
        )

    # 2) 工件内部工序顺序：S[j,o+1] >= S[j,o] + p(j,o)
    #    其中 p(j,o) = sum_m proc_time(j,o,m) * Y[j,o,m]
    for j in range(num_jobs):
        for o in range(ops_per_job[j] - 1):
            mach_list = cand_machines[(j, o)]
            processing_expr = pulp.lpSum(
                proc_time[(j, o, m)] * Y[(j, o, m)] for m in mach_list
            )
            model += (
                S[(j, o + 1)]
                >= S[(j, o)] + processing_expr,
                f"Job_precedence_j{j}_o{o}",
            )

    # 3) 机器不重叠约束 (disjunctive constraints)
    #    仅当两道工序 a,b 都分配到机器 m 时，才强制其先后顺序；
    #    否则通过 bigM * (1 - Y_a_m) 和 bigM * (1 - Y_b_m) 放松约束。
    for m in machines:
        ops_m = ops_on_machine[m]
        for (j1, o1), (j2, o2) in combinations(ops_m, 2):
            if j1 == j2:
                continue
            z_var = Z[(m, j1, o1, j2, o2)]

            # a = (j1,o1) 的加工时间表达式
            p_a = pulp.lpSum(
                proc_time[(j1, o1, m2)] * Y[(j1, o1, m2)]
                for m2 in cand_machines[(j1, o1)]
            )
            # b = (j2,o2)
            p_b = pulp.lpSum(
                proc_time[(j2, o2, m2)] * Y[(j2, o2, m2)]
                for m2 in cand_machines[(j2, o2)]
            )

            # 仅当 Y[(j1,o1,m)]=1 且 Y[(j2,o2,m)]=1 时约束生效
            relax = big_m * (1 - Y[(j1, o1, m)]) + big_m * (1 - Y[(j2, o2, m)])
            model += (
                S[(j1, o1)] + p_a
                <= S[(j2, o2)] + big_m * (1 - z_var) + relax,
                f"Mach_{m}_pair_{j1}_{o1}_before_{j2}_{o2}",
            )
            model += (
                S[(j2, o2)] + p_b
                <= S[(j1, o1)] + big_m * z_var + relax,
                f"Mach_{m}_pair_{j2}_{o2}_before_{j1}_{o1}",
            )

    # 4) 定义 C_max：所有工件最后一道工序的完成时间都不能超过 C_max
    for j in range(num_jobs):
        last_op = ops_per_job[j] - 1
        mach_list = cand_machines[(j, last_op)]
        p_last = pulp.lpSum(
            proc_time[(j, last_op, m)] * Y[(j, last_op, m)] for m in mach_list
        )
        model += (
            C_max >= S[(j, last_op)] + p_last,
            f"Define_Cmax_job{j}",
        )

    # -------------------- 求解 --------------------
    # HiGHS: Python API (需安装 highspy)；HiGHS_CMD: 命令行接口 (需 highs 可执行文件)
    solver_kw = {"timeLimit": time_limit, "msg": msg}
    if gap_rel is not None:
        solver_kw["gapRel"] = gap_rel
    if threads is not None:
        solver_kw["threads"] = threads
    solver = pulp.HiGHS(**solver_kw)
    result_status = model.solve(solver)

    status_str = pulp.LpStatus[result_status]

    # 问题数据均为整数时，最优 C_max 理论为整数；求解器浮点运算会产生 107.99... 等，需取整
    def _round_if_integer(x):
        if x is None:
            return None
        r = round(x)
        return r if abs(r - x) < 1e-6 else x

    if msg:
        print(f"Solve status: {status_str}")
        raw_cmax = pulp.value(C_max)
        print(f"Objective (C_max): {_round_if_integer(raw_cmax) if raw_cmax is not None else raw_cmax}")

    # 提取解（若最优/可行）
    if status_str not in ("Optimal", "Feasible"):
        return {
            "status": status_str,
            "C_max": None,
            "operations": [],
        }

    raw_cmax = pulp.value(C_max)
    cmax_rounded = _round_if_integer(raw_cmax)

    schedule = []
    for j, wp in enumerate(workpieces):
        job_name = wp["name"]
        for o in range(ops_per_job[j]):
            start_time = pulp.value(S[(j, o)])
            start_rounded = _round_if_integer(start_time)

            # 找出该工序分配到的机器及其加工时间
            assigned_m = None
            proc = None
            for m in cand_machines[(j, o)]:
                if pulp.value(Y[(j, o, m)]) > 0.5:
                    assigned_m = m
                    proc = proc_time[(j, o, m)]
                    break

            end_time = start_rounded + (proc if proc is not None else 0)
            schedule.append(
                {
                    "job_index": j,
                    "job_name": job_name,
                    "op_index": o,
                    "machine": assigned_m,
                    "start": start_rounded,
                    "proc_time": proc,
                    "end": end_time,
                }
            )

    # 按开始时间排序，方便查看
    schedule.sort(key=lambda x: x["start"])

    # if gantt_path:  # 甘特图已关闭
    #     cmax_str = int(cmax_rounded) if cmax_rounded is not None else 0
    #     create_gantt_chart_from_milp(
    #         schedule,
    #         title=f"MILP Solution (C_max={cmax_str})",
    #         filename=gantt_path,
    #     )

    return {
        "status": status_str,
        "C_max": cmax_rounded,
        "operations": schedule,
    }


def milp_result_to_expert_entry(instance_path: str, milp_result: dict, problem_filename: str = None) -> dict:
    """
    将 MILP 求解结果转为与 Comparison_heuristic_algorithm.py 生成的 ga_expert_data.pt 完全一致的
    专家数据条目，供 Supervised_train.py 的 SupervisedDataset 使用。

    约定：图节点顺序与 GA 的 GA_to_Diffusion_Converter 一致——
    前 num_ops 个节点为工序（按 job0_op0, job0_op1, ..., job1_op0, ...），
    随后为机器节点（按 machines 列表顺序）。
    优先级与 GA 一致：按机器分组，每台机器上按开始时间排序后 1.0->0.0 线性插值。
    """
    if milp_result["status"] not in ("Optimal", "Feasible") or not milp_result.get("operations"):
        print(f"Warning: No expert data for {instance_path} (milp_result_to_expert_entry returned None)")
        return None

    machines, workpieces = load_instance(instance_path)
    ops_per_job = [len(wp["optional_machines"]) for wp in workpieces]
    num_ops = sum(ops_per_job)
    C_max = milp_result["C_max"]
    if C_max is None:
        print(f"Warning: No C_max for {instance_path} (milp_result_to_expert_entry returned None)")
        return None

    # 按 (job_index, op_index) 排序，得到与图一致的工序顺序
    ops_sorted = sorted(milp_result["operations"], key=lambda x: (x["job_index"], x["op_index"]))
    op_node_indices = []
    machine_node_indices = []

    for op in ops_sorted:
        j, o, m = op["job_index"], op["op_index"], op["machine"]
        op_node_i = sum(ops_per_job[:j]) + o
        machine_node_i = num_ops + machines.index(m)
        op_node_indices.append(op_node_i)
        machine_node_indices.append(machine_node_i)

    # 与 GA 一致的优先级：按机器分组，按开始时间排序，Rank 0 -> 1.0, Rank N-1 -> 0.0
    priorities = [0.0] * num_ops
    machine_queues = {m: [] for m in machines}
    for op in milp_result["operations"]:
        j, o, m = op["job_index"], op["op_index"], op["machine"]
        g_idx = sum(ops_per_job[:j]) + o
        machine_queues[m].append((op["start"], g_idx))
    for m_id, queue in machine_queues.items():
        if not queue:
            continue
        queue.sort(key=lambda x: x[0])
        n = len(queue)
        for rank, (_, g_idx) in enumerate(queue):
            priorities[g_idx] = 1.0 - (rank / (n - 1)) if n > 1 else 1.0

    problem_file = problem_filename or Path(instance_path).name
    return {
        "problem_file": problem_file,
        "machine_ids": list(machines),
        "expert_edges": torch.tensor([op_node_indices, machine_node_indices], dtype=torch.long),
        "expert_priorities": torch.tensor(priorities, dtype=torch.float),
        "makespan": float(C_max),
    }


def _solve_one_instance(
    json_path_str: str,
    time_limit: int,
    gap_rel: float,
    threads: int,
    ga_makespan_ub: float = None,
) -> tuple:
    """
    供多进程调用的单实例求解，返回完整解以便主进程可构建 expert .pt。
    返回 (filename, best_makespan, status, C_max, full_result)。
    full_result 含 operations，用于 milp_result_to_expert_entry。
    """
    res = build_and_solve_milp(
        instance_path=json_path_str,
        time_limit=time_limit,
        msg=False,
        gap_rel=gap_rel,
        threads=threads,
        ga_makespan_ub=ga_makespan_ub,
    )
    cmax = res["C_max"]
    best_makespan = int(round(cmax)) if cmax is not None else ""
    return (
        Path(json_path_str).name,
        best_makespan,
        res["status"],
        cmax,
        res,
    )


def solve_all_in_dir_and_save(
    folder: str = "TestSet/Generalization_Temp",
    output_csv: str = "result_milp_generalization_temp.csv",
    time_limit: int = None,
    gap_rel: float = 0.01,
    threads: int = None,
    parallel_workers: int = 0,
    save_expert_pt: str = None,
    # gantt_dir: str = None,  # 甘特图已关闭
):
    """
    对指定文件夹中的所有 JSON 实例求解 MILP，并按
    `Filename,Best_Makespan` 的格式保存到 CSV。

    - folder: 存放 JSON 实例的目录，相对项目根目录。
    - output_csv: 输出的 CSV 文件路径（相对项目根目录）。
    - time_limit: 求解时间上限（秒），None 表示不限制。
    - gap_rel: MIP 相对间隙，达到即停止（如 0.01 表示 1%），None 表示求到最优。
    - threads: 每个实例求解时使用的线程数，None 为求解器默认。
    - parallel_workers: 并行求解的进程数；0 或 1 表示串行，>1 时多进程批量加速。
    - save_expert_pt: 若指定路径（如 "milp_expert_data.pt"），将把每个实例的最优解转为与
      ga_expert_data.pt 相同格式并保存，供 Supervised_train.py 训练 diffusion 使用。
      可与 parallel_workers 同时使用（并行时也会传回完整解并写入 .pt）。
    """
    folder_path = Path(folder)
    json_files = sorted(folder_path.glob("*.json"), reverse=True)
    # if gantt_dir:
    #     Path(gantt_dir).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(json_files)} JSON instances in '{folder_path}'.")
    use_parallel = parallel_workers and parallel_workers > 1
    if use_parallel:
        print(f"Using {parallel_workers} parallel workers (batch solving).")

    rows = [("Filename", "Best_Makespan")]
    expert_list = [] if save_expert_pt else None

    if use_parallel:
        # 多进程批量求解（每个 worker 返回完整 res，主进程据此写 CSV 与 expert .pt）
        n_workers = min(parallel_workers, len(json_files), os.cpu_count() or 2)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _solve_one_instance,
                    str(json_path),
                    time_limit,
                    gap_rel,
                    threads,
                ): json_path
                for json_path in json_files
            }
            results = []
            for fut in as_completed(futures):
                json_path = futures[fut]
                try:
                    name, best_makespan, status, cmax, res = fut.result()
                    results.append((json_path, name, best_makespan, status, cmax, res))
                    print(f"  Done: {name} -> status={status}, C_max={cmax}")
                except Exception as e:
                    name = json_path.name
                    results.append((json_path, name, "", "Error", None, None))
                    print(f"  Error on {name}: {e}")
            # 按文件名排序，与串行顺序一致
            results.sort(key=lambda x: x[1])
            rows.extend((name, best_makespan) for _, name, best_makespan, _, _, _ in results)
            if save_expert_pt:
                for json_path, name, _bm, _st, _cmax, res in results:
                    if res is not None:
                        entry = milp_result_to_expert_entry(str(json_path), res)
                        if entry is not None:
                            expert_list.append(entry)
            # if gantt_dir:  # 甘特图已关闭
            #     for json_path, name, _bm, _st, cmax, res in results:
            #         if res and res.get("operations"):
            #             cmax_int = int(cmax) if cmax is not None else 0
            #             create_gantt_chart_from_milp(...)
    else:
        # 串行（含需要写 save_expert_pt 时）
        for json_path in json_files:
            print(f"Solving instance: {json_path.name}")
            res = build_and_solve_milp(
                instance_path=str(json_path),
                time_limit=time_limit,
                msg=False,
                gap_rel=gap_rel,
                threads=threads,
            )
            cmax = res["C_max"]
            best_makespan = int(round(cmax)) if cmax is not None else ""
            rows.append((json_path.name, best_makespan))
            print(f"  -> status={res['status']}, C_max={cmax}")

            # if gantt_dir and res.get("operations"):  # 甘特图已关闭
            #     ...
            if save_expert_pt:
                entry = milp_result_to_expert_entry(str(json_path), res)
                if entry is not None:
                    expert_list.append(entry)

    output_path = Path(output_csv)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n✅ MILP results saved to '{output_path}'.")
    print(f"   Total instances: {len(json_files)}")

    if save_expert_pt and expert_list:
        torch.save(expert_list, save_expert_pt)
        print(f"✅ Expert data for supervision saved to '{save_expert_pt}' ({len(expert_list)} instances).")


def get_instance_index(fname: str) -> int:
    """从 Trainset 文件名提取 index，如 job5_m4_12345.json -> 12345（与 Comparison_heuristic_algorithm 一致）."""
    base = os.path.splitext(fname)[0]
    try:
        return int(base.split("_")[-1])
    except (ValueError, IndexError):
        return -1


def solve_trainset_and_save_batches(
    folder: str = "Trainset",
    ga_expert_batches_dir: str = "ga_expert_batches",
    milp_batch_dir: str = "milp_expert_batches",
    BATCH_SIZE: int = 1000,
    time_limit: int = None,
    gap_rel: float = None,
    threads: int = None,
    parallel_workers: int = 0,
    chunk_size: int = None,
):
    """
    对 Trainset 中每个任务求解 MILP，每 1000 个结果保存为一个 .pt 文件（与 ga_expert_batches 格式一致），支持断点续跑。

    - folder: Trainset 目录。
    - ga_expert_batches_dir: GA 专家数据批次目录；若存在则读取每个实例的 GA makespan 作为 Big-M 上界以加速 MILP。
    - milp_batch_dir: 输出目录，保存 milp_expert_data_batch_0.pt, milp_expert_data_batch_1.pt, ...
    - BATCH_SIZE: 每批实例数（默认 1000）。
    - time_limit / gap_rel / threads: 传给 HiGHS。
    - parallel_workers: 并行进程数，0 表示串行。
    - chunk_size: 并行时每批提交的任务数，默认 max(parallel_workers*4, 32)。
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"错误: 找不到文件夹 '{folder_path}'")
        return

    def _instance_index(f):
        return get_instance_index(f.name)

    json_files = sorted(folder_path.glob("*.json"), key=_instance_index)
    if not json_files:
        print(f"在 '{folder_path}' 下未找到 .json 文件")
        return

    Path(milp_batch_dir).mkdir(parents=True, exist_ok=True)
    existing_batches = set()
    for pt_file in Path(milp_batch_dir).glob("milp_expert_data_batch_*.pt"):
        try:
            bid = int(pt_file.stem.split("_")[-1])
            existing_batches.add(bid)
        except (ValueError, IndexError):
            pass
    if existing_batches:
        print(f"断点续跑: 以下批次已存在，将跳过: {sorted(existing_batches)}")

    # 按 batch_id 分组待处理实例，保证顺序
    todo_by_batch = {}
    for jpath in json_files:
        fname = jpath.name
        idx = get_instance_index(fname)
        batch_id = idx // BATCH_SIZE if idx >= 0 else -1
        if batch_id >= 0 and batch_id not in existing_batches:
            todo_by_batch.setdefault(batch_id, []).append((jpath, fname))
    for bid in todo_by_batch:
        todo_by_batch[bid].sort(key=lambda x: get_instance_index(x[1]))

    total_todo = sum(len(v) for v in todo_by_batch.values())
    batch_ids = sorted(todo_by_batch.keys())
    print(f"待处理 {total_todo} 个实例（共 {len(json_files)} 个），{len(batch_ids)} 个批次. BATCH_SIZE={BATCH_SIZE}.")
    print("按批处理：每批仅加载对应 GA .pt 的 makespan，求解并保存后释放，再加载下一批.")

    use_parallel = parallel_workers and parallel_workers > 1
    if use_parallel:
        chunk_size = chunk_size or max(parallel_workers * 4, 32)
        n_workers = min(parallel_workers, os.cpu_count() or 2)

    processed = 0
    for batch_id in batch_ids:
        todo_list = todo_by_batch[batch_id]
        # 仅加载当前批次的 GA makespan，用完后本轮循环结束即释放
        ga_lookup = load_ga_makespan_lookup_one_batch(ga_expert_batches_dir, batch_id)
        n_ga = len(ga_lookup)
        print(f"  [批次 {batch_id}] 已加载 GA makespan {n_ga} 条，待求解 {len(todo_list)} 个实例.")

        current_batch = []
        if use_parallel:
            n_workers_batch = min(n_workers, len(todo_list))
            with ProcessPoolExecutor(max_workers=n_workers_batch) as executor:
                for start in range(0, len(todo_list), chunk_size):
                    chunk = todo_list[start : start + chunk_size]
                    args = [
                        (str(jpath), time_limit, gap_rel, threads, ga_lookup.get(fname))
                        for (jpath, fname) in chunk
                    ]
                    results = list(executor.map(lambda a: _solve_one_instance(*a), args))
                    for (jpath, fname), (name, bm, status, cmax, res) in zip(chunk, results):
                        processed += 1
                        if res and res.get("status") in ("Optimal", "Feasible"):
                            entry = milp_result_to_expert_entry(str(jpath), res)
                            if entry is not None:
                                current_batch.append(entry)
                        print(f"[{processed}/{total_todo}] {fname} -> status={status}, C_max={cmax}")
        else:
            for jpath, fname in todo_list:
                processed += 1
                ga_ub = ga_lookup.get(fname)
                res = build_and_solve_milp(
                    instance_path=str(jpath),
                    time_limit=time_limit,
                    msg=False,
                    gap_rel=gap_rel,
                    threads=threads,
                    ga_makespan_ub=ga_ub,
                )
                if res and res.get("status") in ("Optimal", "Feasible"):
                    entry = milp_result_to_expert_entry(str(jpath), res)
                    if entry is not None:
                        current_batch.append(entry)
                print(f"[{processed}/{total_todo}] {fname} -> status={res['status']}, C_max={res.get('C_max')}")

        if current_batch:
            batch_path = Path(milp_batch_dir) / f"milp_expert_data_batch_{batch_id}.pt"
            torch.save(current_batch, batch_path)
            print(f"   -> 已保存批次 {batch_id} ({len(current_batch)} 条)")
        # ga_lookup 在此处离开作用域，仅保留当前批的 MILP 结果，下一批会重新 load 下一个 .pt

    print(f"✅ MILP 专家数据已保存到 '{milp_batch_dir}'.")


def run_generalization_temp_benchmark(
    time_limit: int = 120,
    gap_rel: float = 0.01,
    parallel_workers: int = 0,
    save_expert_pt: str = None,
    # gantt_dir: str = None,  # 甘特图已关闭
):
    """
    方便在 IDE 里直接运行的一键函数：
    - 读取 `TestSet/Generalization_Temp/` 下全部 json
    - 用 MILP 求 makespan
    - 结果保存为 `result_milp_generalization_temp.csv`

    - time_limit: 每个实例最大求解秒数，None 表示不限制。
    - gap_rel: 相对间隙（如 0.01=1%），达到即停止，加快速度。
    - parallel_workers: 并行进程数，0=串行，4 表示同时解 4 个实例（大幅缩短总时间）。
    - save_expert_pt: 若指定（如 "milp_expert_data.pt"），会同时生成与 ga_expert_data.pt 同格式的
      专家数据，供 Supervised_train 训练 diffusion；可与 parallel_workers 同时使用。
    """
    solve_all_in_dir_and_save(
        folder="TestSet/Generalization_Temp",
        output_csv="result_milp_generalization_temp.csv",
        time_limit=time_limit,
        gap_rel=gap_rel,
        parallel_workers=parallel_workers,
        save_expert_pt=save_expert_pt,
        # gantt_dir=gantt_dir,
    )


if __name__ == "__main__":
    # 对 Trainset 求解 MILP，使用 ga_expert_batches 的 makespan 作为 Big-M 加速，每 1000 个保存为 milp_expert_data_batch_*.pt，支持断点续跑
    solve_trainset_and_save_batches(
        folder="Trainset",
        ga_expert_batches_dir="ga_expert_batches",
        milp_batch_dir="milp_expert_batches",
        BATCH_SIZE=1000,
        time_limit=None,
        gap_rel=None,
        parallel_workers=8,
    )
