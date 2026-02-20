import os
import random
from pathlib import Path
from tqdm import tqdm

# --- 导入你的生成函数 ---
# 确保 Generate_random_problem_instances.py 在同一目录下
from Generate_random_problem_instances import generate_random_fjsp_problem

# ================= 配置区域 =================

# 输出目录 (你的训练脚本读取这个目录)
OUTPUT_DIR = Path("Trainset")

# 总共想生成多少个训练样本？
# 建议至少几百个，RL通常需要大量数据
TOTAL_SAMPLES = 100000

# 配置问题的难度分布
# 我们可以混合生成不同规模的问题，防止模型只学会解特定大小的问题
# 格式: (工件数量范围, 机器数量范围, 权重占比)
PROBLEM_CONFIGS = [
    # [Small]  工件 5-10, 机器 3-5 (占比 20%) -> 适合初期快速学习逻辑
    {"job_range": (4, 7), "mach_range": (3,4), "weight": 1},
]

# 工序参数 (可以根据实际工厂情况微调)
MIN_OPS = 3  # 每个工件最少几道工序
MAX_OPS = 7  # 每个工件最多几道工序
MIN_OPTS = 1  # 每个工序最少有几个可选机器 (柔性)
MAX_OPTS = 3  # 每个工序最多有几个可选机器


# ===========================================

def generate_dataset():
    # 1. 创建目录
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        print(f"📁 Created directory: {OUTPUT_DIR}")
    else:
        print(f"📂 Output directory: {OUTPUT_DIR}")
        print("⚠️  Warning: New files will be added to existing ones.")

    print(f"🚀 Starting generation of {TOTAL_SAMPLES} samples...")
    print(f"   Distribution configs: {len(PROBLEM_CONFIGS)} types")

    # 根据权重计算每种类型生成的数量
    generated_count = 0

    # 创建进度条
    pbar = tqdm(total=TOTAL_SAMPLES)

    while generated_count < TOTAL_SAMPLES:
        # A. 随机选择一种配置模式 (根据权重)
        configs = random.choices(
            PROBLEM_CONFIGS,
            weights=[c['weight'] for c in PROBLEM_CONFIGS],
            k=1
        )[0]

        # B. 在该配置范围内随机采样具体数值
        n_jobs = random.randint(*configs['job_range'])
        n_machs = random.randint(*configs['mach_range'])

        # C. 生成文件名 (包含关键信息方便查看)
        # e.g., "job25_m8_idx102.json"
        filename = f"job{n_jobs}_m{n_machs}_{generated_count}.json"
        filepath = OUTPUT_DIR / filename

        # D. 调用生成器
        try:
            generate_random_fjsp_problem(
                filename=str(filepath),
                num_machines=n_machs,
                num_workpieces=n_jobs,
                min_ops=MIN_OPS,
                max_ops=MAX_OPS,
                min_opts=MIN_OPTS,
                max_opts=MAX_OPTS,
                seed=None  # 随机种子设为None，保证每次不一样
            )
            generated_count += 1
            pbar.update(1)

        except Exception as e:
            print(f"\n❌ Error generating {filename}: {e}")

    pbar.close()
    print(f"\n✅ Done! Generated {generated_count} files in '{OUTPUT_DIR}'.")

    # 简单的统计
    print("\n📊 Dataset Summary:")
    files = list(OUTPUT_DIR.glob("*.json"))
    print(f"   Total Files: {len(files)}")
    print("   You can now run your training script.")


if __name__ == "__main__":
    generate_dataset()