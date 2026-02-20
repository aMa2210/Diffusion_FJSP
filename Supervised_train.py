import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import glob
import os
from tqdm import tqdm
from torch_geometric.data import Dataset, Batch, Data
from torch_geometric.utils import to_dense_batch
from torch.utils.data import DataLoader
import json
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from Industrial_Pipeline_Functions import (
    LightweightIndustrialDiffusion,
    load_ipps_problem_from_json,
    get_ipps_problem_data,
    get_ipps_allowed_mask
)

# ==========================================
# 1. 定义监督学习的数据集 (返回字典模式)
# ==========================================
class SupervisedDataset(Dataset):
    def __init__(self, problem_dir, expert_data_path, device='cpu'):
        super().__init__()
        self.problem_dir = problem_dir
        self.device = device 
        
        # 加载 GA 生成的 .pt 文件
        print(f"Loading expert data from {expert_data_path}...")
        self.expert_data = torch.load(expert_data_path)
        self.expert_map = {item['problem_file']: item for item in self.expert_data}
        
        self.file_list = []
        all_files = glob.glob(os.path.join(problem_dir, "*.json"))
        for f in all_files:
            fname = os.path.basename(f)
            if fname in self.expert_map:
                self.file_list.append(f)
            else:
                print(f'expert result not found for {fname}')
                pass # 忽略没有专家解的文件

        print(f"Found {len(self.file_list)} aligned samples.")

    def len(self):
        return len(self.file_list)
    
    def __len__(self):
        return self.len()

    def get(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        filename = os.path.basename(filepath)
        
        # 1. 加载原始问题图 (Input)
        # 强制使用 CPU 加载数据，防止 DataLoader 多进程时的 CUDA 错误
        raw_wp_dicts, raw_machines = load_ipps_problem_from_json(filepath)
        data = get_ipps_problem_data(raw_wp_dicts, raw_machines, device='cpu')
        
        # 2. 获取专家解 (Label)
        expert_sample = self.expert_map[filename]
        
        # --- 准备 GT 数据 ---
        num_nodes = data.x.size(0)
        
        # A. GT Edges [N, N] (Long Tensor)
        gt_edges = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        
        # 填充 Expert Routing (Op -> Machine)
        exp_edges = expert_sample['expert_edges']
        gt_edges[exp_edges[0].long(), exp_edges[1].long()] = 1
        
        # # 填充 Pinned Edges (Op -> Op) - 这些也算作 Ground Truth 的一部分
        # seq_src, seq_dst = data.edge_index
        # gt_edges[seq_src, seq_dst] = 1
        
        # B. GT Priorities [N] (Float Tensor)
        gt_prio = torch.zeros(num_nodes, dtype=torch.float)
        exp_prio = expert_sample['expert_priorities']
        num_ops = exp_prio.size(0)
        gt_prio[:num_ops] = exp_prio
        
        # 3. 返回字典 (关键！不要直接返回 Data 对象)
        return {
            'data': data,
            'gt_edges': gt_edges,
            'gt_priorities': gt_prio
        }

# ==========================================
# 2. 关键：Custom Collate Function (处理 Padding)
# ==========================================
def custom_collate_fn(batch_list):
    """
    处理 SupervisedDataset 返回的字典列表:
    1. 提取并 Padding 变长的 GT 和 属性矩阵 (Time, Advantage)
    2. 生成干净的 PyG Batch
    """
    # 1. 获取当前 Batch 中最大的节点数
    max_nodes = max([item['data'].x.size(0) for item in batch_list])
    batch_size = len(batch_list)

    # 2. 初始化 Padding 后的 Tensor [B, Max_N, Max_N]
    padded_gt_edges = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.long)
    padded_gt_prio = torch.zeros((batch_size, max_nodes), dtype=torch.float)
    padded_time_mat = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float)
    padded_adv_mat = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float)
    padded_allowed_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)

    clean_data_list = []

    for i, item in enumerate(batch_list):
        data = item['data']
        num_nodes = data.x.size(0)
        
        # --- A. 填充 GT 数据 ---
        padded_gt_edges[i, :num_nodes, :num_nodes] = item['gt_edges']
        padded_gt_prio[i, :num_nodes] = item['gt_priorities']
        node_labels = data.x.argmax(dim=1)
        # 获取该图的合法连接掩码 [N, N]
        # 注意：get_ipps_allowed_mask 需要 device 参数，这里暂时给 'cpu'
        mask = get_ipps_allowed_mask(node_labels, data, device='cpu')
        machine_indices = (node_labels == 1)
        mask[:, ~machine_indices] = False

        padded_allowed_mask[i, :num_nodes, :num_nodes] = mask

        # --- B. 提取并填充 Data 中的属性矩阵 ---
        # 必须在这里提取，因为 PyG Batch 无法自动处理 [N, N] 的堆叠
        if hasattr(data, 'time_matrix'):
            padded_time_mat[i, :num_nodes, :num_nodes] = data.time_matrix
        else:
            print('error code 21412')
        
        if hasattr(data, 'advantage_matrix') and data.advantage_matrix is not None:
            padded_adv_mat[i, :num_nodes, :num_nodes] = data.advantage_matrix
        else:
            print('error code 9872')

        # --- C. 构建干净的 Data 对象 ---
        # "白名单策略"：只保留 x 和 edge_index，彻底避免 PyG 报错
        clean_data = Data(x=data.x, edge_index=data.edge_index)
        clean_data.num_nodes = num_nodes
        clean_data_list.append(clean_data)

    # 3. 生成 PyG Batch
    batch = Batch.from_data_list(clean_data_list)

    return batch, padded_gt_edges, padded_gt_prio, padded_time_mat, padded_adv_mat, padded_allowed_mask


def apply_constrained_edge_noise(gt_edges, allowed_mask, alpha_bar, device):
    """
    Args:
        gt_edges: [B, N, N] 专家解 (0/1)
        allowed_mask: [B, N, N] 合法连接掩码 (True/False)
        alpha_bar: [B] 当前时间步的信号保留率
    Returns:
        noisy_edges: [B, N, N] 加噪后的边 (0/1)
    """
    B, N, _ = gt_edges.shape

    # 1. 扩展 alpha_bar 以匹配形状 [B, 1, 1]
    alpha_reshape = alpha_bar.view(B, 1, 1)

    # 2. 生成随机选择矩阵 (Random Valid Selection)
    # 我们需要在 allowed_mask 为 True 的地方均匀采样
    # 技巧：给 allowed_mask 加随机噪声，然后取 argmax

    # 生成随机噪声，只在 allowed 的位置有值，其他位置为负无穷
    rand_logits = torch.rand_like(allowed_mask.float())
    rand_logits[~allowed_mask] = -1e9

    # 对于每个 Op (行)，选一个随机的合法 Machine (列)
    # [B, N] -> 每一行选中的 machine index
    # 注意：只有 Op->Machine 的行需要随机选，Machine->Op 或 Machine->Machine 应该是空的
    # 这里我们假设 allowed_mask 已经处理好了 Op 行的约束
    random_target_idx = rand_logits.argmax(dim=2)

    # 转回 One-Hot 形式 [B, N, N]
    random_valid_edges = torch.zeros_like(gt_edges).scatter_(2, random_target_idx.unsqueeze(2), 1)

    # 修正：如果某一行全是 False (比如机器节点行)，argmax 会出错或归零，需要 mask 掉
    # 只有那些至少有一个合法连接的行，才应该有边
    has_valid_connection = allowed_mask.sum(dim=2) > 0
    random_valid_edges[~has_valid_connection] = 0

    # 3. 混合策略 (Mix)
    # 生成概率 mask: P < alpha_bar 则保留 GT，否则使用随机合法解
    prob_mask = torch.rand(B, N, N, device=device) < alpha_reshape

    # 组合：Mask 处用 GT，非 Mask 处用 Random Valid
    noisy_edges = torch.where(prob_mask, gt_edges, random_valid_edges)

    # 确保结构完整性：非 Allowed 的地方强制为 0 (双重保险)
    noisy_edges = noisy_edges * allowed_mask.long()

    return noisy_edges


# ==========================================
# 3. 完整的训练脚本
# ==========================================

if __name__ == "__main__":
    
    RUN_NAME = "SL_Run_Fix_Batching_121"
    SPT_checkpoint_dir = 'SPT_checkpoints'
    log_dir = Path(SPT_checkpoint_dir) / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    
    SEED = 42
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_DIR = "Problem_TrainSet_GA"      # JSON 文件夹
    EXPERT_DATA = "ga_expert_data.pt"      # GA 生成的 PT 文件
    
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 2000
    HIDDEN_DIMENSION = 128
    NUM_LAYERS = 6
    N_HEADS = 4
    T_STEPS = 8
    
    
    # 1. 加载数据
    dataset = SupervisedDataset(TRAIN_DIR, EXPERT_DATA, device='cpu')
    # 【关键】使用 custom_collate_fn
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    # 2. 模型
    model = LightweightIndustrialDiffusion(
        T=T_STEPS, 
        hidden_dim=HIDDEN_DIMENSION, 
        device=DEVICE, 
        num_layers=NUM_LAYERS, 
        nhead=N_HEADS
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    # 保存配置
    config = {
        "RUN_NAME": RUN_NAME,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "EPOCHS": EPOCHS,
        "HIDDEN_DIMENSION": HIDDEN_DIMENSION,
        "NUM_LAYERS": NUM_LAYERS,
        "Description": "Supervised Learning with Correct Padding and Batching",
        "T_STEPS": T_STEPS,
        "NUM_LAYERS": NUM_LAYERS,
        "N_HEADS": N_HEADS
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    log_path = log_dir / "training_log.txt"
    with open(log_path, "w") as f:
        f.write("Epoch,Total_Loss,Link_Loss,Prio_Loss\n")
    
    print(f"Start Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss_sum = 0
        link_loss_sum = 0
        prio_loss_sum = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_data, gt_edges, gt_prio_padded, time_matrix, adv_matrix, padded_allowed_mask in pbar:
            
            batch_data = batch_data.to(DEVICE)
            gt_edges = gt_edges.to(DEVICE)          # [B, Max, Max]
            gt_prio_padded = gt_prio_padded.to(DEVICE) # [B, Max]
            time_matrix = time_matrix.to(DEVICE)    # [B, Max, Max]
            adv_matrix = adv_matrix.to(DEVICE)      # [B, Max, Max]
            
            bs = len(gt_edges)
            
            # --- 构造加噪 Priority 输入 (Flattened) ---
            # 1. 采样时间 t
            t = torch.randint(0, model.T, (bs,), device=DEVICE).long()
            
            # 2. 获取每个节点的 alpha_bar
            # node_batch_idx: [Total_Nodes] -> 0,0,0, 1,1, ...
            node_batch_idx = batch_data.batch
            a_bar = model.alpha_bar[t].to(DEVICE) # [B]

            noisy_edges_dense = apply_constrained_edge_noise(
                gt_edges,
                padded_allowed_mask.to(DEVICE),
                a_bar,
                DEVICE
            )
            batch_noisy_edge_index = []
            for i in range(bs):
                num_n = (batch_data.batch == i).sum().item()
                curr_adj = noisy_edges_dense[i, :num_n, :num_n]
                curr_indices = curr_adj.nonzero().t()
                offset = (batch_data.batch < i).sum().item()
                batch_noisy_edge_index.append(curr_indices + offset)
            noisy_routing_edge_index = torch.cat(batch_noisy_edge_index, dim=1)
            final_input_edge_index = torch.cat([
                noisy_routing_edge_index,
                batch_data.edge_index
            ], dim=1)

            node_a_bar = a_bar[node_batch_idx]    # [Total_Nodes]
            # print(f'model alpha bar min {model.alpha_bar[-1]}')
            # 3. 将 Padded Priority 展平为 [Total_Nodes] 以匹配 PyG 的 batch 结构
            flat_prio_list = []
            for i in range(bs):
                # 获取第 i 个图的真实节点数
                num_n = (node_batch_idx == i).sum().item()
                # 只取有效部分，忽略 Padding
                flat_prio_list.append(gt_prio_padded[i, :num_n])
            
            flat_gt_prio = torch.cat(flat_prio_list) # [Total_Nodes]
            
            # 4. 加噪
            noise = torch.randn_like(flat_gt_prio)
            noisy_input_prio = torch.sqrt(node_a_bar) * flat_gt_prio + torch.sqrt(1 - node_a_bar) * noise
            
            # -------------------------------------------------------
            # C. 模型前向
            # -------------------------------------------------------
            # 我们刚才修复了 forward，现在它可以接受 3D 的 time_matrix 和 adv_matrix
            edge_outputs_list = model(
                batch_data.x, 
                final_input_edge_index,
                batch_data.batch, 
                t, 
                time_matrix=time_matrix,     # [B, Max, Max] (3D)
                priorities=noisy_input_prio, # [Total_Nodes] (1D)
                advantage_matrix=adv_matrix  # [B, Max, Max] (3D)
            )
            
            batch_size = gt_edges.size(0)
            max_nodes = gt_edges.size(1)
            output_dim = edge_outputs_list[0].size(-1) # 通常是 4
            
            # 2. 初始化全 0 的 Tensor [B, Max, Max, 4]
            pred_output = torch.zeros((batch_size, max_nodes, max_nodes, output_dim), device=DEVICE)
            
            # 3. 将每个图的预测填入对应的位置
            for i, pred_single in enumerate(edge_outputs_list):
                n = pred_single.size(0) # 当前图的节点数
                pred_output[i, :n, :n, :] = pred_single
            
            # -------------------------------------------------------
            # D. 计算 Loss
            # -------------------------------------------------------
            # 利用 to_dense_batch 获取 mask，只在有效节点对上计算 Loss
            # _, mask = to_dense_batch(batch_data.x, batch_data.batch) # [B, Max]
            # # 边 Mask [B, Max, Max] -> 有效区域为 True
            # edge_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            loss_calc_mask = padded_allowed_mask.to(DEVICE)

            # 1. Routing Loss (Cross Entropy)
            pred_logits = pred_output[..., :2] # [B, Max, Max, 2]

            
            
            if loss_calc_mask.sum() > 0:
                valid_pred = pred_logits[loss_calc_mask]  # [Total_Valid_Options, 2]
                valid_gt = gt_edges[loss_calc_mask]  # [Total_Valid_Options]
                num_pos = valid_gt.sum().float()
                num_total = valid_gt.numel()
                num_neg = num_total - num_pos
                pos_weight = num_neg / (num_pos + 1e-6)
                curr_weight = torch.tensor([1.0, pos_weight], device=DEVICE)
                
                
                loss_link = F.cross_entropy(
                    valid_pred,
                    valid_gt,
                    # weight 可以保留，用于处理正负样本不平衡
                    weight=curr_weight
                )
            else:
                print('error code 94562')
                loss_link = torch.tensor(0.0, device=DEVICE)
            
            # 2. Priority Loss (MSE)
            pred_prio_map = pred_output[..., 2] # [B, Max, Max]
            # GT Prio Map 需要构造: 如果 Op->Machine 有边，则该边的 Prio = Op Prio
            # 我们已经在 gt_prio_padded 里有了 Node Prio，现在需要广播到 Edge 上
            
            # 扩展 gt_prio_padded [B, Max] -> [B, Max, 1]
            gt_prio_map = gt_prio_padded.unsqueeze(-1).expand(-1, -1, gt_edges.size(2))
            
            # # 只在 (GT Edge 存在) AND (有效区域) 的地方计算 Prio Loss
            # prio_calc_mask = (gt_edges == 1)
            prio_calc_mask = loss_calc_mask
            if prio_calc_mask.sum() > 0:
                loss_prio = F.mse_loss(
                    pred_prio_map[prio_calc_mask], 
                    gt_prio_map[prio_calc_mask]
                )
            else:
                loss_prio = torch.tensor(0.0, device=DEVICE)
                
            loss = loss_link + 1.0 * loss_prio
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪是个好习惯
            optimizer.step()
            
            total_loss_sum += loss.item()
            link_loss_sum += loss_link.item()
            prio_loss_sum += loss_prio.item()
            
            pbar.set_postfix({'Link': loss_link.item(), 'Prio': loss_prio.item()})
            
        avg_loss = total_loss_sum / len(loader)
        avg_link = link_loss_sum / len(loader)
        avg_prio = prio_loss_sum / len(loader)
        
        log_msg = f"Epoch {epoch} | Loss: {avg_loss:.4f} | Link: {avg_link:.4f} | Prio: {avg_prio:.4f}"
        print(log_msg)
        scheduler.step()
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.5f},{avg_link:.5f},{avg_prio:.5f}\n")
            
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), f"{SPT_checkpoint_dir}/{RUN_NAME}/sl_model_{epoch}.pth")