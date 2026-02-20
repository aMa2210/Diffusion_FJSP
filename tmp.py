import torch

# 加载文件
results = torch.load('ga_expert_data.pt', weights_only=True)


target_file = 'job5_m3_574.json'  # 你想要查找的文件名

# 遍历列表查找匹配的文件名
found = False
for entry in results:
    if entry.get('problem_file') == target_file:
        print(f"文件名: {entry['problem_file']}")
        print(f"Makespan: {entry['makespan']}")
        found = True
        break

if not found:
    print(f"未找到文件: {target_file}")