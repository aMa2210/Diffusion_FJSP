import json
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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


def load_ga_makespan_lookup_from_csv(csv_path: str) -> dict:
    """
    从 GA 结果 CSV（如 results_GA_for_training_Diffusion.csv）读取 makespan，
    返回 文件名 -> makespan 的映射。使用第三列（Stochastic_MK）作为上界。
    若文件不存在或读取出错则返回 {}。
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    lookup = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                fname = row[0].strip()
                if not fname:
                    continue
                try:
                    # 第三列（索引 2）作为 makespan 上界
                    lookup[fname] = float(row[2])
                except (ValueError, IndexError):
                    continue
        return lookup
    except Exception as e:
        print(f"Warning: could not load GA makespan from {csv_path}: {e}")
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


def _build_milp_model(machines, workpieces, cand_machines, proc_time, ops_on_machine, ops_per_job, num_jobs, big_m):
    """用给定 big_m 构建 FJSP 的 PuLP 模型，返回 (model, S, Y, Z, C_max)。用于两阶段：先求可行解，再用 incumbent 收紧 big_m 重建并重解。"""
    model = pulp.LpProblem("FJSP_Makespan_Minimization", pulp.LpMinimize)
    S = {
        (j, o): pulp.LpVariable(f"S_{j}_{o}", lowBound=0, cat="Continuous")
        for j in range(num_jobs)
        for o in range(ops_per_job[j])
    }
    Y = {
        (j, o, m): pulp.LpVariable(f"Y_{j}_{o}_{m}", lowBound=0, upBound=1, cat="Binary")
        for (j, o), mach_list in cand_machines.items()
        for m in mach_list
    }
    C_max = pulp.LpVariable("C_max", lowBound=0, cat="Continuous")
    Z = {}
    for m in machines:
        ops_m = ops_on_machine[m]
        for (j1, o1), (j2, o2) in combinations(ops_m, 2):
            if j1 == j2:
                continue
            Z[(m, j1, o1, j2, o2)] = pulp.LpVariable(
                f"Z_{m}_{j1}_{o1}_{j2}_{o2}", lowBound=0, upBound=1, cat="Binary"
            )
    model += C_max, "Minimize_makespan"
    for (j, o), mach_list in cand_machines.items():
        model += (
            pulp.lpSum(Y[(j, o, m)] for m in mach_list) == 1,
            f"Select_one_machine_job{j}_op{o}",
        )
    for j in range(num_jobs):
        for o in range(ops_per_job[j] - 1):
            mach_list = cand_machines[(j, o)]
            processing_expr = pulp.lpSum(
                proc_time[(j, o, m)] * Y[(j, o, m)] for m in mach_list
            )
            model += (
                S[(j, o + 1)] >= S[(j, o)] + processing_expr,
                f"Job_precedence_j{j}_o{o}",
            )
    for m in machines:
        ops_m = ops_on_machine[m]
        for (j1, o1), (j2, o2) in combinations(ops_m, 2):
            if j1 == j2:
                continue
            z_var = Z[(m, j1, o1, j2, o2)]
            p_a = pulp.lpSum(
                proc_time[(j1, o1, m2)] * Y[(j1, o1, m2)]
                for m2 in cand_machines[(j1, o1)]
            )
            p_b = pulp.lpSum(
                proc_time[(j2, o2, m2)] * Y[(j2, o2, m2)]
                for m2 in cand_machines[(j2, o2)]
            )
            relax = big_m * (1 - Y[(j1, o1, m)]) + big_m * (1 - Y[(j2, o2, m)])
            model += (
                S[(j1, o1)] + p_a <= S[(j2, o2)] + big_m * (1 - z_var) + relax,
                f"Mach_{m}_pair_{j1}_{o1}_before_{j2}_{o2}",
            )
            model += (
                S[(j2, o2)] + p_b <= S[(j1, o1)] + big_m * z_var + relax,
                f"Mach_{m}_pair_{j2}_{o2}_before_{j1}_{o1}",
            )
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
    return model, S, Y, Z, C_max


def build_and_solve_milp(
    instance_path: str,
    time_limit: int = None,
    msg: bool = True,
    gap_rel: float = None,
    threads: int = None,
    ga_makespan_ub: float = None,
    first_solve_time_limit: int = 10,
    # gantt_path: str = None,  # 甘特图已关闭
):
    """
    针对一个 FJSP 实例构建 MILP 模型，求解最小完工时间（makespan）的调度。

    建模假设（标准柔性 Job-Shop / FJSP 形式）:
    - 每个工件内部工序按给定顺序依次加工。
    - 每道工序必须选择其候选机器集合里的恰好一台机器。
    - 同一时间每台机器最多加工一道工序。
    - 目标：最小化系统完工时间 C_max。

    - ga_makespan_ub: 若提供，用作 Big-M 初始上界，并添加约束 C_max <= ga_makespan_ub，保证解不差于 GA（否则可能因 CSV 缺该文件而得不到上界，出现 MILP 报“最优”大于 GA 的情况）。
    - first_solve_time_limit: 默认 10。先以此秒数求一可行解，若得到则用该解的 makespan 收紧 Big-M 后重建模型再解，往往能更快证最优；设为 0 或 None 则关闭两阶段、只解一次。
    """
    machines, workpieces = load_instance(instance_path)
    num_jobs = len(workpieces)
    ops_per_job = [len(wp["optional_machines"]) for wp in workpieces]
    cand_machines = {}
    proc_time = {}
    for j, wp in enumerate(workpieces):
        opt_machs_list = wp["optional_machines"]
        proc_times_list = wp["processing_time"]
        for o, (mach_list, time_list) in enumerate(zip(opt_machs_list, proc_times_list)):
            cand_machines[(j, o)] = list(mach_list)
            for m, p in zip(mach_list, time_list):
                proc_time[(j, o, m)] = p
    ops_on_machine = {m: [] for m in machines}
    for (j, o), mach_list in cand_machines.items():
        for m in mach_list:
            ops_on_machine[m].append((j, o))

    total_max = 0
    for j, wp in enumerate(workpieces):
        for p_list in wp["processing_time"]:
            total_max += max(p_list) if p_list else 0
    if ga_makespan_ub is not None and ga_makespan_ub > 0:
        big_m = min(total_max, ga_makespan_ub)
    else:
        big_m = total_max

    def _add_ga_ub_constraint(model, C_max_var, ga_ub):
        if ga_ub is not None and ga_ub > 0:
            model += C_max_var <= ga_ub, "C_max_ub_from_GA"

    use_two_phase = first_solve_time_limit and first_solve_time_limit > 0
    if use_two_phase:
        model, S, Y, Z, C_max = _build_milp_model(
            machines, workpieces, cand_machines, proc_time, ops_on_machine,
            ops_per_job, num_jobs, big_m,
        )
        _add_ga_ub_constraint(model, C_max, ga_makespan_ub)
        solver_kw = {"timeLimit": first_solve_time_limit, "msg": msg}
        if gap_rel is not None:
            solver_kw["gapRel"] = gap_rel
        if threads is not None:
            solver_kw["threads"] = threads
        solver = pulp.HiGHS(**solver_kw)
        result_status = model.solve(solver)
        status_str = pulp.LpStatus[result_status]
        raw_cmax = pulp.value(C_max) if C_max is not None else None
        if status_str in ("Optimal", "Feasible") and raw_cmax is not None and raw_cmax < big_m:
            incumbent = int(np.ceil(raw_cmax))
            big_m_new = min(big_m, incumbent)
            if big_m_new < big_m and big_m_new >= 1:
                big_m = big_m_new
                if msg:
                    print(f"Two-phase: tightening big_m to incumbent {big_m}, re-solving.")
                model, S, Y, Z, C_max = _build_milp_model(
                    machines, workpieces, cand_machines, proc_time, ops_on_machine,
                    ops_per_job, num_jobs, big_m,
                )
                _add_ga_ub_constraint(model, C_max, ga_makespan_ub)
                solver_kw2 = {"timeLimit": time_limit, "msg": msg}
                if gap_rel is not None:
                    solver_kw2["gapRel"] = gap_rel
                if threads is not None:
                    solver_kw2["threads"] = threads
                solver2 = pulp.HiGHS(**solver_kw2)
                result_status = model.solve(solver2)
                status_str = pulp.LpStatus[result_status]
        else:
            # 第一阶段未得可行解或未证最优时：用同一模型再解一次，使用完整 time_limit（可为 None 表示不限制）
            if status_str not in ("Optimal", "Feasible"):
                solver_kw = {"timeLimit": time_limit, "msg": msg}
                if gap_rel is not None:
                    solver_kw["gapRel"] = gap_rel
                if threads is not None:
                    solver_kw["threads"] = threads
                solver = pulp.HiGHS(**solver_kw)
                result_status = model.solve(solver)
                status_str = pulp.LpStatus[result_status]
    else:
        model, S, Y, Z, C_max = _build_milp_model(
            machines, workpieces, cand_machines, proc_time, ops_on_machine,
            ops_per_job, num_jobs, big_m,
        )
        _add_ga_ub_constraint(model, C_max, ga_makespan_ub)
        solver_kw = {"timeLimit": time_limit, "msg": msg}
        if gap_rel is not None:
            solver_kw["gapRel"] = gap_rel
        if threads is not None:
            solver_kw["threads"] = threads
        solver = pulp.HiGHS(**solver_kw)
        result_status = model.solve(solver)
        status_str = pulp.LpStatus[result_status]

    # -------------------- 求解后处理与解提取 --------------------

    # 问题数据均为整数时，最优 C_max 及开始时间理论为整数；求解器浮点误差会产生 64.999998... 等
    # 容差 1e-4：与整数的差小于 0.0001 则视为整数并取整返回
    def _round_if_integer(x):
        if x is None:
            return None
        r = round(x)
        return r if abs(r - x) < 1e-4 else x

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
    first_solve_time_limit: int = 10,
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
        first_solve_time_limit=first_solve_time_limit,
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


def _solve_one_instance_unpack(args):
    """供 ProcessPoolExecutor.map 使用：可 pickle 的包装，避免传 lambda."""
    return _solve_one_instance(*args)


def solve_all_in_dir_and_save(
    folder: str = "TestSet/Generalization_Temp",
    output_csv: str = "result_milp_generalization_temp.csv",
    time_limit: int = None,
    gap_rel: float = 0.01,
    threads: int = None,
    parallel_workers: int = 0,
    save_expert_pt: str = None,
    first_solve_time_limit: int = 10,
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
    - first_solve_time_limit: 默认 10，先短时求可行解再收紧 Big-M 重解；0 或 None 关闭。
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
                    None,
                    first_solve_time_limit,
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
                first_solve_time_limit=first_solve_time_limit,
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
    ga_makespan_csv: str = "results_GA_for_training_Diffusion.csv",
    milp_batch_dir: str = "milp_expert_batches",
    BATCH_SIZE: int = 1000,
    time_limit: int = None,
    gap_rel: float = None,
    threads: int = None,
    parallel_workers: int = 0,
    chunk_size: int = None,
    first_solve_time_limit: int = 10,
):
    """
    对 Trainset 中每个任务求解 MILP，每 1000 个结果保存为一个 .pt 文件，支持断点续跑。
    GA makespan 上界仅从 ga_makespan_csv（按文件名、第三列）读取。

    - folder: Trainset 目录。
    - ga_makespan_csv: GA 结果 CSV 路径（如 results_GA_for_training_Diffusion.csv），按文件名匹配、第三列作为 makespan 上界。
    - milp_batch_dir: 输出目录，保存 milp_expert_data_batch_0.pt, milp_expert_data_batch_1.pt, ...
    - BATCH_SIZE: 每批实例数（默认 1000）。
    - time_limit / gap_rel / threads: 传给 HiGHS。
    - parallel_workers: 并行进程数，0 表示串行。
    - chunk_size: 并行时每批提交的任务数，默认 max(parallel_workers*4, 32)。
    - first_solve_time_limit: 默认 10，先短时求可行解再收紧 Big-M 重解；设为 0 或 None 关闭。
    若出现 "process pool was terminated abruptly" / BrokenProcessPool，多为某 worker 被系统杀进程（如内存不足 OOM）：
    可减小 parallel_workers（如 8）或设置 time_limit 限制单实例时间；当前实现会在检测到后把该 chunk 剩余任务改为串行重跑并继续。
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
            print(f"Error: {pt_file} is not a valid batch file")
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

    # 从 CSV 按文件名读取第三列作为 GA makespan 上界
    ga_lookup = load_ga_makespan_lookup_from_csv(ga_makespan_csv) if ga_makespan_csv else {}
    if ga_lookup:
        print(f"已从 CSV 按文件名读取 GA makespan 上界：{ga_makespan_csv}，共 {len(ga_lookup)} 条.")

    use_parallel = parallel_workers and parallel_workers > 1
    if use_parallel:
        chunk_size = chunk_size or max(parallel_workers * 4, 32)
        n_workers = min(parallel_workers, os.cpu_count() or 2)

    processed = 0
    for batch_id in batch_ids:
        todo_list = todo_by_batch[batch_id]
        n_ga = sum(1 for (_, fname) in todo_list if fname in ga_lookup)
        print(f"  [批次 {batch_id}] GA makespan 可用 {n_ga}/{len(todo_list)} 条，待求解 {len(todo_list)} 个实例.")

        current_batch = []
        if use_parallel:
            n_workers_batch = min(n_workers, len(todo_list))
            for start in range(0, len(todo_list), chunk_size):
                chunk = todo_list[start : start + chunk_size]
                future_to_item = None
                # 每个 chunk 使用独立进程池，避免单个 worker 被系统杀死（如 OOM）时拖垮整个批处理
                with ProcessPoolExecutor(max_workers=n_workers_batch) as executor:
                    future_to_item = {
                        executor.submit(
                            _solve_one_instance_unpack,
                            (str(jpath), time_limit, gap_rel, threads, ga_lookup.get(fname), first_solve_time_limit),
                        ): (jpath, fname)
                        for (jpath, fname) in chunk
                    }
                    done_fnames = set()
                    pool_broken = False
                    for future in as_completed(future_to_item):
                        jpath, fname = future_to_item[future]
                        try:
                            name, bm, status, cmax, res = future.result()
                        except BrokenProcessPool as e:
                            pool_broken = True
                            print(f"[chunk] Worker died (often OOM); re-running remaining in chunk serially.", flush=True)
                            remaining = [(jp, fn) for (jp, fn) in future_to_item.values() if fn not in done_fnames]
                            for jp, fn in remaining:
                                processed += 1
                                try:
                                    res = build_and_solve_milp(
                                        instance_path=str(jp),
                                        time_limit=time_limit,
                                        msg=False,
                                        gap_rel=gap_rel,
                                        threads=threads,
                                        ga_makespan_ub=ga_lookup.get(fn),
                                        first_solve_time_limit=first_solve_time_limit,
                                    )
                                    if res and res.get("status") in ("Optimal", "Feasible"):
                                        entry = milp_result_to_expert_entry(str(jp), res)
                                        if entry is not None:
                                            current_batch.append(entry)
                                    print(f"[{processed}/{total_todo}] {fn} -> status={res['status']}, C_max={res.get('C_max')}", flush=True)
                                except Exception as e2:
                                    print(f"[{processed}/{total_todo}] {fn} -> Error: {e2}", flush=True)
                            break
                        except Exception as e:
                            print(f"[{processed + 1}/{total_todo}] {fname} -> Error: {e}", flush=True)
                            processed += 1
                            continue
                        done_fnames.add(fname)
                        processed += 1
                        if res and res.get("status") in ("Optimal", "Feasible"):
                            entry = milp_result_to_expert_entry(str(jpath), res)
                            if entry is not None:
                                current_batch.append(entry)
                        print(f"[{processed}/{total_todo}] {fname} -> status={status}, C_max={cmax}", flush=True)
                    if pool_broken:
                        pass  # 已在本 chunk 内串行补跑完剩余任务，继续下一 chunk
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
                    first_solve_time_limit=first_solve_time_limit,
                )
                if res and res.get("status") in ("Optimal", "Feasible"):
                    entry = milp_result_to_expert_entry(str(jpath), res)
                    if entry is not None:
                        current_batch.append(entry)
                print(f"[{processed}/{total_todo}] {fname} -> status={res['status']}, C_max={res.get('C_max')}", flush=True)

        if current_batch:
            batch_path = Path(milp_batch_dir) / f"milp_expert_data_batch_{batch_id}.pt"
            torch.save(current_batch, batch_path)
            print(f"   -> 已保存批次 {batch_id} ({len(current_batch)} 条)")

    print(f"✅ MILP 专家数据已保存到 '{milp_batch_dir}'.")




if __name__ == "__main__":
    # 对 Trainset 求解 MILP，从 results_GA_for_training_Diffusion.csv 按文件名读取第三列作为 GA makespan 上界，每 1000 个保存为 milp_expert_data_batch_*.pt，支持断点续跑
    solve_trainset_and_save_batches(
        folder="Trainset",
        ga_makespan_csv="results_GA_for_training_Diffusion.csv",
        milp_batch_dir="milp_expert_batches",
        BATCH_SIZE=1000,
        time_limit=300,
        gap_rel=None,
        parallel_workers=32,
    )
