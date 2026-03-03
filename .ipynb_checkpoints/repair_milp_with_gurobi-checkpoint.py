"""
针对 milp_expert_batches 中两类问题使用 Gurobi 重新求解并填充：

1. MILP 解劣于 GA 解（因代码 bug 导致）—— 重新求解
2. 实例在指定时间内未求得可行解（MILP 中缺省）—— 重新求解

用法: python repair_milp_with_gurobi.py
      或修改下方配置后运行
"""

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch

# ---------- 配置 ----------
TRAINSET_DIR = "Trainset"
MILP_BATCH_DIR = "milp_expert_batches"
GA_EXPERT_BATCH_DIR = "ga_expert_batches"
GA_CSV = "results_GA_for_training_Diffusion.csv"
BATCH_SIZE = 1000
TIME_LIMIT = 3000  # Gurobi 求解时间上限（秒）
FIRST_SOLVE_TIME_LIMIT = 30  # 两阶段：先短时求可行解
GAP_REL = None  # 相对间隙，None 表示求到最优
THREADS = 1  # 线程数，None 为默认
PARALLEL_WORKERS = 32  # 0=串行，>0 多进程（注意 Gurobi 许可证可能限制并行）


def load_instance(path):
    """与 MILP_solve_ipps 相同：读取 JSON 实例。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["machines"], data["workpieces"]


def load_ga_makespan_from_csv(csv_path: str) -> dict:
    """文件名 -> GA makespan (Stochastic_MK, 第三列)。"""
    path = Path(csv_path)
    if not path.exists():
        return {}
    lookup = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    print(f"Warning: could not load GA makespan from {csv_path}: {row}")
                    continue
                fname = row[0].strip()
                if not fname:
                    print(f"Warning: could not load GA makespan from {csv_path}: {row}")
                    continue
                try:
                    lookup[fname] = float(row[2])
                except (ValueError, IndexError):
                    continue
        return lookup
    except Exception as e:
        print(f"Warning: could not load GA makespan from {csv_path}: {e}")
        return {}


def load_ga_makespan_from_batches(batch_dir: str) -> dict:
    """从 ga_expert_batches 加载 文件名 -> makespan。"""
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return {}
    lookup = {}
    for pt_file in sorted(batch_path.glob("ga_expert_data_batch_*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(pt_file, map_location="cpu")
        if not isinstance(data, list):
            continue
        for entry in data:
            fname = entry.get("problem_file")
            mk = entry.get("makespan")
            if fname and mk is not None:
                lookup[fname] = float(mk)
    return lookup


def get_instance_index(fname: str) -> int:
    """job5_m3_6271.json -> 6271"""
    base = os.path.splitext(fname)[0]
    try:
        return int(base.split("_")[-1])
    except (ValueError, IndexError):
        return -1


def _build_gurobi_model(machines, workpieces, cand_machines, proc_time, ops_on_machine, ops_per_job, num_jobs, big_m):
    """构建 FJSP 的 Gurobi 模型，返回 (model, S, Y, Z, C_max)。"""
    m = gp.Model("FJSP_Makespan")
    m.setParam("OutputFlag", 0)

    S = {}
    for j in range(num_jobs):
        for o in range(ops_per_job[j]):
            S[(j, o)] = m.addVar(lb=0, name=f"S_{j}_{o}", vtype=GRB.CONTINUOUS)

    Y = {}
    for (j, o), mach_list in cand_machines.items():
        for mach in mach_list:
            Y[(j, o, mach)] = m.addVar(lb=0, ub=1, name=f"Y_{j}_{o}_{mach}", vtype=GRB.BINARY)

    C_max = m.addVar(lb=0, name="C_max", vtype=GRB.CONTINUOUS)

    Z = {}
    for mach in machines:
        ops_m = ops_on_machine[mach]
        for (j1, o1), (j2, o2) in combinations(ops_m, 2):
            if j1 == j2:
                continue
            Z[(mach, j1, o1, j2, o2)] = m.addVar(lb=0, ub=1, name=f"Z_{mach}_{j1}_{o1}_{j2}_{o2}", vtype=GRB.BINARY)

    m.setObjective(C_max, GRB.MINIMIZE)

    for (j, o), mach_list in cand_machines.items():
        m.addConstr(gp.quicksum(Y[(j, o, mach)] for mach in mach_list) == 1, name=f"Select_one_machine_{j}_{o}")

    for j in range(num_jobs):
        for o in range(ops_per_job[j] - 1):
            mach_list = cand_machines[(j, o)]
            m.addConstr(
                S[(j, o + 1)] >= S[(j, o)] + gp.quicksum(proc_time[(j, o, mach)] * Y[(j, o, mach)] for mach in mach_list),
                name=f"Job_precedence_{j}_{o}",
            )

    for mach in machines:
        ops_m = ops_on_machine[mach]
        for (j1, o1), (j2, o2) in combinations(ops_m, 2):
            if j1 == j2:
                continue
            z_var = Z[(mach, j1, o1, j2, o2)]
            p_a = gp.quicksum(proc_time[(j1, o1, m2)] * Y[(j1, o1, m2)] for m2 in cand_machines[(j1, o1)])
            p_b = gp.quicksum(proc_time[(j2, o2, m2)] * Y[(j2, o2, m2)] for m2 in cand_machines[(j2, o2)])
            relax = big_m * (1 - Y[(j1, o1, mach)]) + big_m * (1 - Y[(j2, o2, mach)])
            m.addConstr(S[(j1, o1)] + p_a <= S[(j2, o2)] + big_m * (1 - z_var) + relax, name=f"Mach_{mach}_pair_1")
            m.addConstr(S[(j2, o2)] + p_b <= S[(j1, o1)] + big_m * z_var + relax, name=f"Mach_{mach}_pair_2")

    for j in range(num_jobs):
        last_op = ops_per_job[j] - 1
        mach_list = cand_machines[(j, last_op)]
        p_last = gp.quicksum(proc_time[(j, last_op, mach)] * Y[(j, last_op, mach)] for mach in mach_list)
        m.addConstr(C_max >= S[(j, last_op)] + p_last, name=f"Define_Cmax_{j}")

    m.update()
    return m, S, Y, Z, C_max


def build_and_solve_milp_gurobi(
    instance_path: str,
    time_limit: int = None,
    msg: bool = False,
    gap_rel: float = None,
    threads: int = None,
    ga_makespan_ub: float = None,
    first_solve_time_limit: int = 30,
) -> dict:
    """
    使用 Gurobi 求解 FJSP，返回与 MILP_solve_ipps.build_and_solve_milp 相同格式的 dict。
    """
    machines, workpieces = load_instance(instance_path)
    num_jobs = len(workpieces)
    ops_per_job = [len(wp["optional_machines"]) for wp in workpieces]
    cand_machines = {}
    proc_time = {}
    for j, wp in enumerate(workpieces):
        for o, (mach_list, time_list) in enumerate(zip(wp["optional_machines"], wp["processing_time"])):
            cand_machines[(j, o)] = list(mach_list)
            for m, p in zip(mach_list, time_list):
                proc_time[(j, o, m)] = p
    ops_on_machine = {m: [] for m in machines}
    for (j, o), mach_list in cand_machines.items():
        for m in mach_list:
            ops_on_machine[m].append((j, o))

    total_max = sum(max(p_list) if p_list else 0 for wp in workpieces for p_list in wp["processing_time"])
    if ga_makespan_ub is not None and ga_makespan_ub > 0:
        big_m = min(total_max, ga_makespan_ub)
    else:
        print(f'ga results not exist for {instance_path}')
        big_m = total_max

    def _round_if_integer(x):
        if x is None:
            return None
        r = round(x)
        return r if abs(r - x) < 1e-4 else x

    use_two_phase = first_solve_time_limit and first_solve_time_limit > 0
    m, S, Y, Z, C_max = _build_gurobi_model(
        machines, workpieces, cand_machines, proc_time, ops_on_machine, ops_per_job, num_jobs, big_m,
    )
    if ga_makespan_ub is not None and ga_makespan_ub > 0:
        m.addConstr(C_max <= ga_makespan_ub, name="C_max_ub_from_GA")
    m.update()

    m.setParam("TimeLimit", first_solve_time_limit if use_two_phase else (time_limit or 1e9))
    if gap_rel is not None:
        m.setParam("MIPGap", gap_rel)
    if threads is not None:
        m.setParam("Threads", threads)

    m.optimize()
    status = m.status
    status_str = "Optimal" if status == GRB.OPTIMAL else "Feasible" if status == GRB.TIME_LIMIT and m.SolCount > 0 else "Infeasible" if status == GRB.INFEASIBLE else "Unknown"

    raw_cmax = C_max.X if hasattr(C_max, "X") and C_max.X is not None else None
    if use_two_phase and status_str in ("Optimal", "Feasible") and raw_cmax is not None and raw_cmax < big_m:
        incumbent = int(np.ceil(raw_cmax))
        big_m_new = min(big_m, incumbent)
        if big_m_new < big_m and big_m_new >= 1:
            big_m = big_m_new
            m.dispose()
            m, S, Y, Z, C_max = _build_gurobi_model(
                machines, workpieces, cand_machines, proc_time, ops_on_machine, ops_per_job, num_jobs, big_m,
            )
            if ga_makespan_ub is not None and ga_makespan_ub > 0:
                m.addConstr(C_max <= ga_makespan_ub, name="C_max_ub_from_GA")
            m.update()
            m.setParam("TimeLimit", time_limit or 1e9)
            if gap_rel is not None:
                m.setParam("MIPGap", gap_rel)
            if threads is not None:
                m.setParam("Threads", threads)
            m.optimize()
            status = m.status
            status_str = "Optimal" if status == GRB.OPTIMAL else "Feasible" if status == GRB.TIME_LIMIT and m.SolCount > 0 else "Infeasible" if status == GRB.INFEASIBLE else "Unknown"
    elif use_two_phase and status_str not in ("Optimal", "Feasible"):
        m.setParam("TimeLimit", time_limit or 1e9)
        m.optimize()
        status = m.status
        status_str = "Optimal" if status == GRB.OPTIMAL else "Feasible" if status == GRB.TIME_LIMIT and m.SolCount > 0 else "Infeasible" if status == GRB.INFEASIBLE else "Unknown"

    if status_str not in ("Optimal", "Feasible"):
        return {"status": status_str, "C_max": None, "operations": []}

    raw_cmax = C_max.X
    cmax_rounded = _round_if_integer(raw_cmax)
    schedule = []
    for j, wp in enumerate(workpieces):
        job_name = wp["name"]
        for o in range(ops_per_job[j]):
            start_time = S[(j, o)].X
            start_rounded = _round_if_integer(start_time)
            assigned_m = None
            proc = None
            for mach in cand_machines[(j, o)]:
                if Y[(j, o, mach)].X > 0.5:
                    assigned_m = mach
                    proc = proc_time[(j, o, mach)]
                    break
            end_time = start_rounded + (proc if proc is not None else 0)
            schedule.append({
                "job_index": j, "job_name": job_name, "op_index": o, "machine": assigned_m,
                "start": start_rounded, "proc_time": proc, "end": end_time,
            })
    schedule.sort(key=lambda x: x["start"])
    try:
        m.dispose()
    except Exception:
        pass
    return {"status": status_str, "C_max": cmax_rounded, "operations": schedule}


def _solve_one_instance_gurobi(
    json_path_str: str,
    time_limit: int,
    gap_rel: float,
    threads: int,
    ga_makespan_ub: float = None,
    first_solve_time_limit: int = 30,
) -> tuple:
    """
    供多进程调用的单实例求解，返回 (filename, status, cmax, full_result)。
    """
    res = build_and_solve_milp_gurobi(
        instance_path=json_path_str,
        time_limit=time_limit,
        msg=False,
        gap_rel=gap_rel,
        threads=threads,
        ga_makespan_ub=ga_makespan_ub,
        first_solve_time_limit=first_solve_time_limit,
    )
    cmax = res["C_max"]
    return (Path(json_path_str).name, res["status"], cmax, res)


def milp_result_to_expert_entry(instance_path: str, milp_result: dict, problem_filename: str = None) -> dict:
    """与 MILP_solve_ipps 相同：将 MILP 解转为 expert 条目。"""
    if milp_result["status"] not in ("Optimal", "Feasible") or not milp_result.get("operations"):
        return None
    machines, workpieces = load_instance(instance_path)
    ops_per_job = [len(wp["optional_machines"]) for wp in workpieces]
    num_ops = sum(ops_per_job)
    C_max = milp_result["C_max"]
    if C_max is None:
        return None
    ops_sorted = sorted(milp_result["operations"], key=lambda x: (x["job_index"], x["op_index"]))
    op_node_indices = []
    machine_node_indices = []
    for op in ops_sorted:
        j, o, m = op["job_index"], op["op_index"], op["machine"]
        op_node_i = sum(ops_per_job[:j]) + o
        machine_node_i = num_ops + machines.index(m)
        op_node_indices.append(op_node_i)
        machine_node_indices.append(machine_node_i)
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


def load_milp_batches(batch_dir: str) -> tuple:
    """返回 (fname_to_entry, batch_id_to_entries)。"""
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return {}, {}
    fname_to_entry = {}
    batch_id_to_entries = {}
    for pt_file in sorted(batch_path.glob("milp_expert_data_batch_*.pt")):
        try:
            bid = int(pt_file.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(pt_file, map_location="cpu")
        if not isinstance(data, list):
            continue
        batch_id_to_entries[bid] = list(data)
        for entry in data:
            fname = entry.get("problem_file")
            if fname:
                fname_to_entry[fname] = entry
    return fname_to_entry, batch_id_to_entries


def _batch_id(fname: str) -> int:
    idx = get_instance_index(fname)
    return idx // BATCH_SIZE if idx >= 0 else -1


def _apply_repair_to_batches(fname, entry, batch_id_to_entries, milp_fname_to_entry):
    """将单个修复结果合并到 batch_id_to_entries。"""
    bid = _batch_id(fname)
    if bid < 0:
        return
    if bid not in batch_id_to_entries:
        batch_id_to_entries[bid] = []
    entries = batch_id_to_entries[bid]
    found = False
    for k, e in enumerate(entries):
        if e.get("problem_file") == fname:
            entries[k] = entry
            found = True
            break
    if not found:
        entries.append(entry)
        entries.sort(key=lambda e: get_instance_index(e.get("problem_file", "")))
    milp_fname_to_entry[fname] = entry


def _save_one_batch(milp_dir, bid, entries):
    """只保存指定 batch 到磁盘。"""
    milp_dir.mkdir(parents=True, exist_ok=True)
    out_path = milp_dir / f"milp_expert_data_batch_{bid}.pt"
    torch.save(entries, out_path)
    print(f"  [检查点] 已保存 batch_{bid}.pt ({len(entries)} 条)")


def main():
    trainset = Path(TRAINSET_DIR)
    milp_dir = Path(MILP_BATCH_DIR)
    if not trainset.exists():
        print(f"错误: Trainset 目录不存在: {trainset}")
        return
    if not milp_dir.exists():
        print(f"错误: MILP 批次目录不存在: {milp_dir}")
        return

    json_files = sorted(trainset.glob("*.json"), key=lambda p: get_instance_index(p.name))
    all_fnames = {p.name for p in json_files}
    print(f"Trainset 中共 {len(all_fnames)} 个实例")

    ga_csv = load_ga_makespan_from_csv(GA_CSV)
    ga_batches = load_ga_makespan_from_batches(GA_EXPERT_BATCH_DIR)
    ga_makespan = {**ga_batches, **ga_csv}
    print(f"GA makespan 来源: CSV {len(ga_csv)} 条, ga_expert_batches {len(ga_batches)} 条, 合并后 {len(ga_makespan)} 条")

    milp_fname_to_entry, batch_id_to_entries = load_milp_batches(MILP_BATCH_DIR)
    existing_batch_ids = set(batch_id_to_entries.keys())
    print(f"MILP 批次: {len(batch_id_to_entries)} 个批次 (batch_id: {sorted(existing_batch_ids)}), 共 {len(milp_fname_to_entry)} 条")

    to_repair_by_batch = {}
    skipped_not_yet_processed = 0
    for fname in all_fnames:
        bid = _batch_id(fname)
        if bid < 0 or bid not in existing_batch_ids:
            skipped_not_yet_processed += 1
            continue
        json_path = trainset / fname
        milp_entry = milp_fname_to_entry.get(fname)
        ga_mk = ga_makespan.get(fname)

        if milp_entry is None:
            to_repair_by_batch.setdefault(bid, []).append((fname, json_path, None, ga_mk, "missing"))
        elif ga_mk is not None and milp_entry.get("makespan", float("inf")) > ga_mk:
            to_repair_by_batch.setdefault(bid, []).append((fname, json_path, milp_entry, ga_mk, "worse"))
        else:
            pass

    if skipped_not_yet_processed > 0:
        print(f"已跳过 {skipped_not_yet_processed} 个实例（所属 batch 尚未生成，MILP 未处理）")

    total_to_repair = sum(len(v) for v in to_repair_by_batch.values())
    if total_to_repair == 0:
        print("无需修复，退出。")
        return

    n_missing = sum(1 for items in to_repair_by_batch.values() for item in items if item[4] == "missing")
    n_worse = total_to_repair - n_missing
    print(f"\n待修复: 缺省 {n_missing} 个, 劣于 GA {n_worse} 个, 共 {total_to_repair} 个（按 batch 逐个处理）")

    use_parallel = PARALLEL_WORKERS and PARALLEL_WORKERS > 1
    if use_parallel:
        print(f"使用 {PARALLEL_WORKERS} 个进程并行求解（每个 batch 内并行）")

    total_repaired = 0
    for bid in sorted(to_repair_by_batch.keys()):
        batch_to_repair = to_repair_by_batch[bid]
        entries = batch_id_to_entries[bid]
        print(f"\n--- Batch {bid}: 待修复 {len(batch_to_repair)} 个 ---")

        repaired_in_batch = 0
        if use_parallel:
            n_workers = min(PARALLEL_WORKERS, len(batch_to_repair), os.cpu_count() or 2)
            batch_repair_by_fname = {item[0]: item for item in batch_to_repair}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        _solve_one_instance_gurobi,
                        str(item[1]),
                        TIME_LIMIT,
                        GAP_REL,
                        THREADS,
                        item[3] if item[3] is not None and item[3] > 0 else None,
                        FIRST_SOLVE_TIME_LIMIT,
                    ): item[0]
                    for item in batch_to_repair
                }
                done_count = 0
                for fut in as_completed(futures):
                    fname = futures[fut]
                    fname, json_path, old_entry, ga_mk, reason = batch_repair_by_fname[fname]
                    try:
                        name, status, cmax, res = fut.result()
                        entry = milp_result_to_expert_entry(str(json_path), res, fname)
                        if entry is not None:
                            _apply_repair_to_batches(fname, entry, batch_id_to_entries, milp_fname_to_entry)
                            repaired_in_batch += 1
                            total_repaired += 1
                            if repaired_in_batch % 5 == 0:
                                _save_one_batch(milp_dir, bid, entries)
                            print(f"  [{done_count + 1}/{len(batch_to_repair)}] {fname} -> 成功, C_max={res['C_max']}")
                        else:
                            print(f"  [{done_count + 1}/{len(batch_to_repair)}] {fname} -> 失败, status={res['status']}")
                    except Exception as e:
                        print(f"  [{done_count + 1}/{len(batch_to_repair)}] {fname} -> 错误: {e}")
                    done_count += 1
        else:
            for i, (fname, json_path, old_entry, ga_mk, reason) in enumerate(batch_to_repair):
                ga_ub = ga_mk if ga_mk is not None and ga_mk > 0 else None
                print(f"[{i+1}/{len(batch_to_repair)}] {fname} ({reason}) -> Gurobi 求解中...")
                res = build_and_solve_milp_gurobi(
                    str(json_path),
                    time_limit=TIME_LIMIT,
                    msg=False,
                    gap_rel=GAP_REL,
                    threads=THREADS,
                    ga_makespan_ub=ga_ub,
                    first_solve_time_limit=FIRST_SOLVE_TIME_LIMIT,
                )
                entry = milp_result_to_expert_entry(str(json_path), res, fname)
                if entry is not None:
                    _apply_repair_to_batches(fname, entry, batch_id_to_entries, milp_fname_to_entry)
                    repaired_in_batch += 1
                    total_repaired += 1
                    if repaired_in_batch % 5 == 0:
                        _save_one_batch(milp_dir, bid, entries)
                    print(f"  -> 成功, C_max={res['C_max']}")
                else:
                    print(f"  -> 失败, status={res['status']}")

        if repaired_in_batch > 0:
            _save_one_batch(milp_dir, bid, entries)

    print(f"\n完成: 共修复 {total_repaired} 个实例，已写回 {MILP_BATCH_DIR}")


if __name__ == "__main__":
    main()
