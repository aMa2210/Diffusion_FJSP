import json
import random
import copy
import os
import csv
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import re
from pathlib import Path
import torch

class GA_to_Diffusion_Converter:
    def __init__(self, jobs_data, machine_ids, device='cpu'):
        self.jobs_data = jobs_data
        self.machine_ids = machine_ids
        self.device = device
        
        self.num_machines = len(machine_ids)
        self.total_ops = sum(len(ops) for ops in jobs_data.values())
        self.num_nodes = self.total_ops + self.num_machines
        
        # 建立 Machine ID -> Graph Node Index 的映射
        self.machine_id_to_node_idx = {
            m_id: i + self.total_ops 
            for i, m_id in enumerate(machine_ids)
        }
        
        # 预计算：Job ID -> 全局工序起始索引 的映射
        # 例如: Job1从0开始, Job2从5开始...
        self.job_start_indices = {}
        curr = 0
        sorted_job_ids = sorted(self.jobs_data.keys())
        for j in sorted_job_ids:
            self.job_start_indices[j] = curr
            curr += len(self.jobs_data[j])

    def convert(self, individual, completed_operations):
        """
        修改版: 
        1. Routing (边) 依然来自 individual.machine_gene (这是决策源头)
        2. Priority (值) 改为来自 completed_operations (这是仿真结果，包含真实顺序)
        """
        
        # --- 1. 转换 Edges (Routing) ---
        # (这部分逻辑不变，因为机器分配是由基因决定的)
        source_nodes = []
        target_nodes = []
        global_op_idx = 0
        gene_idx = 0
        sorted_job_ids = sorted(self.jobs_data.keys())
        
        for jid in sorted_job_ids:
            ops = self.jobs_data[jid]
            for op_data in ops:
                possible_machines = list(op_data['machines'].keys())
                choice_idx = individual.machine_gene[gene_idx] % len(possible_machines)
                selected_m_id = possible_machines[choice_idx]
                
                u = global_op_idx 
                v = self.machine_id_to_node_idx[selected_m_id]
                
                source_nodes.append(u)
                target_nodes.append(v)
                
                global_op_idx += 1
                gene_idx += 1
                
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # --- 2. 转换 Priorities (Machine-Level Sequencing) ---
        # 🔥🔥🔥 核心修改 🔥🔥🔥
        
        priorities = torch.zeros(self.total_ops)
        
        # A. 按机器分组收集工序
        # 结构: machine_queues = { m_id: [ (start_time, global_op_idx), ... ] }
        machine_queues = {m: [] for m in self.machine_ids}
        
        for op_record in completed_operations:
            # 解析记录: {'workpiece': 'Workpiece1', 'machine': 1, 'start_time': 10, 'feature': 1}
            wp_str = op_record['workpiece']
            # 从 "Workpiece1" 提取 ID 1
            # 注意：你的代码里 extract_number 逻辑可能有变，这里写稳健一点
            job_id = int(re.search(r'\d+', str(wp_str)).group())
            
            # feature 是工序在工件内的序号 (1-based)，转为 0-based
            op_internal_idx = op_record['feature'] - 1
            
            # 计算全局节点索引
            g_idx = self.job_start_indices[job_id] + op_internal_idx
            
            m_id = op_record['machine']
            start_t = op_record['start_time']
            
            machine_queues[m_id].append((start_t, g_idx))
            
        # B. 对每台机器的队列按时间排序，并赋值优先级
        for m_id, queue in machine_queues.items():
            if not queue:
                continue
                
            # 按开始时间从小到大排序
            # 如果开始时间相同，保持原序 (虽然在甘特图中不太可能完全相同)
            queue.sort(key=lambda x: x[0])
            
            # 赋值逻辑:
            # 越早开始 -> 优先级越高 (1.0)
            # 越晚开始 -> 优先级越低 (0.0)
            n = len(queue)
            for rank, (st, g_idx) in enumerate(queue):
                if n > 1:
                    # 线性插值: Rank 0 -> 1.0, Rank N-1 -> 0.0
                    val = 1.0 - (rank / (n - 1))
                else:
                    val = 1.0 # 只有一个工序，优先级拉满
                
                priorities[g_idx] = val

        return edge_index, priorities
class Individual:
    def __init__(self):
        self.process_gene = []  # 工序编码
        self.machine_gene = []  # 机器编码
        self.makespan = float('inf')  # 适应度值 (Makespan)


class SingleObjectiveGA:
    def __init__(self, jobs_data, machine_ids, pop_size=100, max_gen=100, pc=0.8, pm=0.1):
        self.jobs_data = jobs_data
        self.machine_ids = machine_ids
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pc = pc
        self.pm = pm
        self.population = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = Individual()

            # 1. 生成工序编码
            p_gene = []
            for jid, ops in self.jobs_data.items():
                p_gene.extend([jid] * len(ops))
            random.shuffle(p_gene)
            ind.process_gene = p_gene

            # 2. 生成机器编码
            m_gene = []
            for jid in sorted(self.jobs_data.keys()):
                for op in self.jobs_data[jid]:
                    num_options = len(op['machines'])
                    m_gene.append(random.randint(0, num_options - 1))
            ind.machine_gene = m_gene

            self.calculate_fitness(ind)
            self.population.append(ind)

    def calculate_fitness(self, ind):
        machine_timeline = {m: [] for m in self.machine_ids}
        job_next_available = {j: 0 for j in self.jobs_data.keys()}
        job_op_counter = {j: 0 for j in self.jobs_data.keys()}

        for job_id in ind.process_gene:
            op_idx = job_op_counter[job_id]
            op_data = self.jobs_data[job_id][op_idx]

            # 计算机器基因索引偏移
            gene_offset = 0
            for j in sorted(self.jobs_data.keys()):
                if j == job_id: break
                gene_offset += len(self.jobs_data[j])
            machine_gene_idx = gene_offset + op_idx

            possible_machines = list(op_data['machines'].keys())
            choice_idx = ind.machine_gene[machine_gene_idx] % len(possible_machines)
            machine_id = possible_machines[choice_idx]
            proc_time = op_data['machines'][machine_id]

            # 计算开始时间
            start_time_job = job_next_available[job_id]
            m_log = machine_timeline[machine_id]
            start_time_machine = m_log[-1][1] if m_log else 0

            real_start = max(start_time_job, start_time_machine)
            real_end = real_start + proc_time

            machine_timeline[machine_id].append((real_start, real_end))
            job_next_available[job_id] = real_end
            job_op_counter[job_id] += 1

        ind.makespan = max(job_next_available.values())

    def selection(self):
        p1 = random.choice(self.population)
        p2 = random.choice(self.population)
        return p1 if p1.makespan < p2.makespan else p2

    def crossover(self, p1, p2):
        # POX Crossover
        all_jobs = list(self.jobs_data.keys())
        job_set1 = set(random.sample(all_jobs, random.randint(1, max(1, len(all_jobs) - 1))))

        c1_proc = [-1] * len(p1.process_gene)
        c2_proc = [-1] * len(p2.process_gene)

        for i, gene in enumerate(p1.process_gene):
            if gene in job_set1: c1_proc[i] = gene
        for i, gene in enumerate(p2.process_gene):
            if gene in job_set1: c2_proc[i] = gene

        p2_idx = 0
        for i in range(len(c1_proc)):
            if c1_proc[i] == -1:
                while p2.process_gene[p2_idx] in job_set1: p2_idx += 1
                c1_proc[i] = p2.process_gene[p2_idx]
                p2_idx += 1

        p1_idx = 0
        for i in range(len(c2_proc)):
            if c2_proc[i] == -1:
                while p1.process_gene[p1_idx] in job_set1: p1_idx += 1
                c2_proc[i] = p1.process_gene[p1_idx]
                p1_idx += 1

        # Uniform Crossover (Machines)
        c1_mach, c2_mach = [], []
        for i in range(len(p1.machine_gene)):
            if random.random() < 0.5:
                c1_mach.append(p1.machine_gene[i])
                c2_mach.append(p2.machine_gene[i])
            else:
                c1_mach.append(p2.machine_gene[i])
                c2_mach.append(p1.machine_gene[i])

        off1, off2 = Individual(), Individual()
        off1.process_gene, off1.machine_gene = c1_proc, c1_mach
        off2.process_gene, off2.machine_gene = c2_proc, c2_mach
        return off1, off2

    def mutation(self, ind):
        if random.random() < self.pm and len(ind.process_gene) > 1:
            idx1, idx2 = sorted(random.sample(range(len(ind.process_gene)), 2))
            ind.process_gene[idx1:idx2 + 1] = ind.process_gene[idx1:idx2 + 1][::-1]

        if random.random() < self.pm:
            idx = random.randint(0, len(ind.machine_gene) - 1)
            ind.machine_gene[idx] = random.randint(0, 50)

    def run(self):
        self.initialize_population()
        # 初始最优
        best_so_far = min(self.population, key=lambda x: x.makespan).makespan

        for gen in range(self.max_gen):
            offspring_pop = []
            while len(offspring_pop) < self.pop_size:
                p1 = self.selection()
                p2 = self.selection()
                if random.random() < self.pc:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                self.mutation(c1)
                self.mutation(c2)
                self.calculate_fitness(c1)
                self.calculate_fitness(c2)
                offspring_pop.append(c1)
                offspring_pop.append(c2)

            combined = self.population + offspring_pop
            combined.sort(key=lambda x: x.makespan)
            self.population = combined[:self.pop_size]

            # 可选：如果收敛太快可以提前退出，这里为了稳定性跑满
            current_best = self.population[0].makespan

        # return self.population[0].makespan
        return self.population[0]


def evaluate_stochastic_with_log(ind: Individual, jobs_data, machine_ids, uncertainty=0.1, seed=None):
    """
    修改后的评估函数：除了返回 makespan，还返回详细的工序列表供绘图使用。
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    machine_timeline = {m: [] for m in machine_ids}
    job_next_available = {j: 0 for j in jobs_data.keys()}
    job_op_counter = {j: 0 for j in jobs_data.keys()}
    
    # 🔥 新增：用于存储详细操作记录的列表
    completed_operations = []

    for job_id in ind.process_gene:
        op_idx = job_op_counter[job_id]
        op_data = jobs_data[job_id][op_idx]

        gene_offset = 0
        for j in sorted(jobs_data.keys()):
            if j == job_id: break
            gene_offset += len(jobs_data[j])
        machine_gene_idx = gene_offset + op_idx

        possible_machines = list(op_data['machines'].keys())
        choice_idx = ind.machine_gene[machine_gene_idx] % len(possible_machines)
        machine_id = possible_machines[choice_idx]
        
        base_time = op_data['machines'][machine_id]
        
        if uncertainty > 0:
            fluctuation = rng.uniform(-uncertainty, uncertainty)
            real_proc_time = base_time * (1 + fluctuation)
            real_proc_time = max(0.1, real_proc_time)
        else:
            real_proc_time = base_time

        start_time_job = job_next_available[job_id]
        m_log = machine_timeline[machine_id]
        start_time_machine = m_log[-1][1] if m_log else 0

        real_start = max(start_time_job, start_time_machine)
        real_end = real_start + real_proc_time

        machine_timeline[machine_id].append((real_start, real_end))
        job_next_available[job_id] = real_end
        job_op_counter[job_id] += 1
        
        # 🔥 记录操作详情 (格式需匹配 create_gantt_chart)
        # 注意：这里构造的 workpiece 名字要符合 "Workpiece{ID}" 的格式
        completed_operations.append({
            'workpiece': f"Workpiece{job_id}", 
            'machine': machine_id,
            'start_time': real_start,
            'processing_time': real_proc_time,
            'feature': op_idx + 1
        })

    return max(job_next_available.values()), completed_operations


def create_gantt_chart(completed_operations, title="Gantt Chart", filename=None):
    """
    绘制甘特图
    """
    if not completed_operations:
        print("⚠️ No operations to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. 数据解析与排序
    raw_workpieces = list(set(op['workpiece'] for op in completed_operations))
    
    def extract_number(text):
        match = re.search(r'\d+', str(text))
        return int(match.group()) if match else 0

    workpieces = sorted(raw_workpieces, key=extract_number)
    colors = plt.cm.tab20(np.linspace(0, 1, len(workpieces)))
    color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}

    # 2. 绘制条形图
    for operation in completed_operations:
        m_id = operation['machine']      
        wp = operation['workpiece']      
        start = operation['start_time']
        dur = operation['processing_time']
        
        wp_num = extract_number(wp)
        label_text = f"J{wp_num-1}" # 显示为 J0, J1... 建议改为 J{wp_num} 看个人习惯
        
        ax.barh(y=m_id, width=dur, left=start, 
                height=0.6, align='center', 
                color=color_map[wp], edgecolor='black', alpha=0.9)
        
        ax.text(start + dur / 2, m_id, label_text, 
                ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # 3. 设置坐标轴
    machines = sorted(list(set(op['machine'] for op in completed_operations)))
    ax.set_yticks(machines)
    ax.set_yticklabels([f"M-{m}" for m in machines])
    
    ax.set_ylabel("Machines")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"📊 Gantt chart saved to {filename}")
        plt.close(fig)
    else:
        plt.show()

# ==========================================
# 2. 数据加载函数
# ==========================================

def load_data_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    machines_list = data.get("machines", [])
    jobs_data = {}

    for idx, wp in enumerate(data["workpieces"]):
        job_id = idx + 1
        ops_list = []
        opt_machines = wp["optional_machines"]
        proc_times = wp["processing_time"]

        for m_list, t_list in zip(opt_machines, proc_times):
            machine_dict = {}
            for m_id, t_val in zip(m_list, t_list):
                machine_dict[m_id] = t_val
            ops_list.append({'machines': machine_dict})

        jobs_data[job_id] = ops_list
    return machines_list, jobs_data


def _process_one_file(args):
    """
    供多进程调用的单文件求解（需在模块顶层定义以便 pickle）。
    args: (file_path, filename, uncertainty, eval_seed, n_classifier_solutions)
    返回: ("ok", filename, det_makespan, stochastic_mk, data_sample, classifier_sample) 或 ("error", filename, None, None, error_msg, None)
    """
    file_path, filename, uncertainty, eval_seed, n_classifier_solutions = args
    try:
        machine_ids, jobs_data = load_data_from_json(file_path)
        ga = SingleObjectiveGA(jobs_data, machine_ids, pop_size=100, max_gen=500)
        best_ind = ga.run()
        det_makespan = best_ind.makespan
        stochastic_mk, ops_list = evaluate_stochastic_with_log(
            best_ind, jobs_data, machine_ids, uncertainty=uncertainty, seed=eval_seed
        )
        converter = GA_to_Diffusion_Converter(jobs_data, machine_ids)
        edge_index, priorities = converter.convert(best_ind, ops_list)
        data_sample = {
            "problem_file": filename,
            "machine_ids": machine_ids,
            "expert_edges": edge_index,
            "expert_priorities": priorities,
            "makespan": best_ind.makespan,
        }
        # 从「优秀解 → 随机差解」连续谱中均匀取 n 个，供 classifier 训练（非仅最终种群）
        # 最终种群都是进化后的较优解；再生成一批随机个体（未进化）作为差解
        ga_random = SingleObjectiveGA(jobs_data, machine_ids, pop_size=100, max_gen=0)
        ga_random.initialize_population()
        combined = list(ga.population) + list(ga_random.population)  # 优 + 随机
        sorted_combined = sorted(combined, key=lambda x: x.makespan)
        n = min(n_classifier_solutions, len(sorted_combined))
        if n == 0:
            indices = []
        else:
            indices = [int(i * (len(sorted_combined) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        solutions = []
        for idx in indices:
            ind = sorted_combined[idx]
            _, ops_list_i = evaluate_stochastic_with_log(
                ind, jobs_data, machine_ids, uncertainty=uncertainty, seed=eval_seed
            )
            ei, pr = converter.convert(ind, ops_list_i)
            solutions.append({
                "expert_edges": ei,
                "expert_priorities": pr,
                "makespan": ind.makespan,
            })
        classifier_sample = {
            "problem_file": filename,
            "machine_ids": machine_ids,
            "solutions": solutions,  # 共 n 个，性能从优到劣
        }
        return ("ok", filename, det_makespan, stochastic_mk, data_sample, classifier_sample)
    except Exception as e:
        return ("error", filename, None, None, str(e), None)


# ==========================================
# 3. 批量处理主程序
# ==========================================

if __name__ == "__main__":

    # folder_path = os.path.join("TestSet", "Generalization_Temp")
    # folder_path = 'Problem_TrainSet_GA'
    
    folder_path = 'Trainset'  # 求解 Trainset 中所有 instances
    output_csv = "results_GA_for_training_Diffusion.csv"
    # 分批保存：按文件名末尾的 index 划分，每 BATCH_SIZE 个 instance 存一个 .pt
    dataset_batch_dir = Path("ga_expert_batches")
    dataset_batch_dir.mkdir(parents=True, exist_ok=True)
    classifier_batch_dir = Path("classifier_batches")  # 每问题 9 个解（1 最优 + 8 覆盖优→劣），供 classifier 训优劣
    classifier_batch_dir.mkdir(parents=True, exist_ok=True)
    BATCH_SIZE = 1000  # 每批 instance 数量，可按需修改
    N_CLASSIFIER_SOLUTIONS = 9  # 每问题保存的解数：1 最优 + 8 个覆盖不同性能

    # 甘特图已关闭，不再创建目录与图片
    # gantt_dir = Path("Gantt_Charts_GA_for_training_DM")
    # gantt_dir.mkdir(parents=True, exist_ok=True)
    UNCERTAINTY_LEVEL = 0  # 10% 的时间波动
    EVAL_SEED = 42
    N_WORKERS = min(8, os.cpu_count() or 8)  # 并行进程数，可按 CPU 核心数调整
    CHUNK_SIZE = max(N_WORKERS * 4, 32)  # 每批提交的任务数，保证顺序与批次一致

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 找不到文件夹路径: {folder_path}")
        print("请确认代码所在的目录下是否存在 TestSet/Generalization_Temp 文件夹。")
        exit()

    def get_instance_index(fname):
        """从 Trainset 文件名提取 index，如 job5_m4_12345.json -> 12345"""
        base = os.path.splitext(fname)[0]
        try:
            return int(base.split('_')[-1])
        except (ValueError, IndexError):
            return -1

    # 获取所有 json 文件，按文件名中的 index 排序，保证分批顺序一致
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    files.sort(key=lambda f: get_instance_index(f))

    # 断点续跑：已存在对应 .pt 的批次不再处理
    existing_batches = set()
    for pt_file in dataset_batch_dir.glob("ga_expert_data_batch_*.pt"):
        try:
            bid = int(pt_file.stem.split("_")[-1])
            existing_batches.add(bid)
        except (ValueError, IndexError):
            pass
    if existing_batches:
        print(f"断点续跑: 以下批次结果已存在，将跳过: {sorted(existing_batches)}")

    # 待处理列表：排除已存在批次的文件，按 instance_index 排序以保持批次顺序
    todo_list = []
    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        instance_index = get_instance_index(filename)
        batch_id = instance_index // BATCH_SIZE if instance_index >= 0 else (i // BATCH_SIZE)
        if batch_id not in existing_batches:
            todo_list.append((file_path, filename))
    todo_list.sort(key=lambda x: get_instance_index(x[1]))

    total_todo = len(todo_list)
    print(f"检测到 {len(files)} 个任务文件，待处理 {total_todo} 个（多进程 workers={N_WORKERS}，chunk={CHUNK_SIZE}）")
    print("-" * 50)

    results = []
    results_written_count = 0  # 已写入 CSV 的条数，用于每 1000 条同步一次
    csv_header_written = not os.path.exists(output_csv)  # 断点续跑时文件可能已存在

    start_time_total = time.time()
    current_batch = []
    current_classifier_batch = []  # 与 current_batch 一一对应，存 classifier 用 9 解
    current_batch_id = None
    processed_count = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for start in range(0, len(todo_list), CHUNK_SIZE):
            chunk = todo_list[start : start + CHUNK_SIZE]
            args = [(path, fname, UNCERTAINTY_LEVEL, EVAL_SEED, N_CLASSIFIER_SOLUTIONS) for path, fname in chunk]
            for ret in executor.map(_process_one_file, args):
                status, filename, det_mk, stoch_mk, data_or_err, classifier_sample = ret
                instance_index = get_instance_index(filename)
                batch_id = instance_index // BATCH_SIZE if instance_index >= 0 else -1
                if status == "ok":
                    results.append([filename, f"{det_mk:.2f}", f"{stoch_mk:.2f}"])
                    if current_batch_id is not None and batch_id != current_batch_id:
                        batch_path = dataset_batch_dir / f"ga_expert_data_batch_{current_batch_id}.pt"
                        torch.save(current_batch, batch_path)
                        classifier_path = classifier_batch_dir / f"classifier_data_batch_{current_batch_id}.pt"
                        torch.save(current_classifier_batch, classifier_path)
                        n = len(current_batch)
                        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
                            w = csv.writer(f)
                            if not csv_header_written:
                                w.writerow(["Filename", "Deterministic_MK", "Stochastic_MK"])
                                csv_header_written = True
                            w.writerows(results[results_written_count : results_written_count + n])
                        results_written_count += n
                        print(f"   -> 已保存批次 {current_batch_id}: expert + classifier ({n} 条)，CSV 已同步")
                        current_batch = []
                        current_classifier_batch = []
                    current_batch_id = batch_id
                    current_batch.append(data_or_err)
                    current_classifier_batch.append(classifier_sample)
                    processed_count += 1
                    print(f"[{processed_count}/{total_todo}] {filename} -> Det: {det_mk:.1f} | Stoch: {stoch_mk:.1f}")
                else:
                    results.append([filename, "Error"])
                    processed_count += 1
                    print(f"[{processed_count}/{total_todo}] 出错: {filename} -> {data_or_err}")

    # 保存最后一批 .pt（expert + classifier），并写入该批对应的 CSV 行
    if current_batch:
        batch_path = dataset_batch_dir / f"ga_expert_data_batch_{current_batch_id}.pt"
        torch.save(current_batch, batch_path)
        classifier_path = classifier_batch_dir / f"classifier_data_batch_{current_batch_id}.pt"
        torch.save(current_classifier_batch, classifier_path)
        n = len(current_batch)
        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not csv_header_written:
                w.writerow(["Filename", "Deterministic_MK", "Stochastic_MK"])
                csv_header_written = True
            w.writerows(results[results_written_count : results_written_count + n])
        results_written_count += n
        print(f"   -> 已保存批次 {current_batch_id}: expert + classifier ({n} 条)，CSV 已同步")

    # 写入尚未同步的 CSV 行（通常已在上方每批同步完，此处多为空；保留以防逻辑差异）
    if results_written_count < len(results):
        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not csv_header_written:
                w.writerow(["Filename", "Deterministic_MK", "Stochastic_MK"])
            w.writerows(results[results_written_count:])

    total_time = time.time() - start_time_total
    print("-" * 50)
    print(f"所有任务已完成。总耗时: {total_time:.2f} 秒")
    print(f"CSV 结果: {os.path.abspath(output_csv)}")
    print(f"专家数据批次目录: {dataset_batch_dir.absolute()}")
    print(f"Classifier 数据批次目录: {classifier_batch_dir.absolute()}")