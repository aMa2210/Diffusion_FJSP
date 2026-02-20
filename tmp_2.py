import os
import subprocess

# --- 配置区域 ---
TARGET_DIR = "Trainset"  # 包含那10万个文件的文件夹
BATCH_SIZE = 500  # 每批次处理的文件数量
REMOTE = "origin"  # 远程仓库名
BRANCH = "main"  # 分支名


# ----------------

def run_git(args):
    try:
        subprocess.run(["git"] + args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    return True


def batch_upload():
    # 1. 获取所有未追踪或已修改的文件列表
    # 注意：在超大数量下，git ls-files 也会有一定开销
    print("正在扫描文件列表...")
    files = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", TARGET_DIR],
        text=True
    ).splitlines()

    total_files = len(files)
    print(f"共发现 {total_files} 个文件待处理。")

    for i in range(0, total_files, BATCH_SIZE):
        batch = files[i: