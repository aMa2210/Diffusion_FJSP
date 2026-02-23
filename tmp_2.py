import os
import subprocess

# --- 配置区域 ---
TARGET_DIR = "Trainset"  # 包含那10万个文件的文件夹
BATCH_SIZE = 1000        # 每批次处理的文件数量
REMOTE = "origin"        # 远程仓库名
BRANCH = "main"          # 分支名

# ----------------

def run_git(args, capture=True):
    try:
        result = subprocess.run(
            ["git"] + args,
            check=True,
            capture_output=capture,
            text=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        print(f"Git 错误: {err}")
        return False, err
    except FileNotFoundError:
        print("错误: 未找到 git 命令，请确保已安装 Git 并加入 PATH。")
        return False, "No Git"

def batch_upload():
    # 1. 基础配置优化
    print("正在优化 Git 配置...")
    run_git(["config", "--global", "user.email", "tkm199888@gmail.com"])
    run_git(["config", "--global", "user.name", "aMa2210"])
    run_git(["config", "http.postBuffer", "524288000"])  # 增加缓冲区到 500MB
    run_git(["config", "core.fscache", "true"])         # Windows 扫描加速

    # 2. 清理旧索引（防止之前的卡死残留锁定）
    print("清理潜在的索引锁定...")
    if os.path.exists(".git/index.lock"):
        try: os.remove(".git/index.lock")
        except: pass
    run_git(["reset"])

    # 3. 使用 Python 原生扫描文件，避开 git ls-files 的性能瓶颈
    print(f"正在扫描目录 '{TARGET_DIR}' 下的文件...")
    files = []
    if not os.path.exists(TARGET_DIR):
        print(f"错误: 找不到目录 {TARGET_DIR}")
        return

    for root, dirs, filenames in os.walk(TARGET_DIR):
        for f in filenames:
            full_path = os.path.join(root, f)
            # 转换为 Git 兼容的正斜杠路径
            rel_path = os.path.relpath(full_path, ".").replace("\\", "/")
            files.append(rel_path)

    total_files = len(files)
    print(f"扫描完成！共计 {total_files} 个文件。")

    if total_files == 0:
        print("没有发现待上传的文件。")
        return

    # 4. 分批处理
    first_push = True 
    
    for i in range(0, total_files, BATCH_SIZE):
        batch = files[i : i + BATCH_SIZE]
        current_batch_num = i // BATCH_SIZE + 1
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n进度: [{current_batch_num}/{total_batches}] 正在处理第 {i+1} 到 {i+len(batch)} 个文件...")
        
        # Add
        add_res, _ = run_git(["add"] + batch)
        if not add_res:
            print("Git add 失败，停止运行。")
            break

        # Commit
        commit_msg = f"batch upload {current_batch_num} ({len(batch)} files)"
        commit_res, _ = run_git(["commit", "-m", commit_msg])
        if not commit_res:
            print("提交可能为空或失败，尝试继续...")

        # Push
        print(f"正在推送到远程仓库 {REMOTE}/{BRANCH}...")
        push_args = ["push", REMOTE, BRANCH]
        
        # 核心逻辑：第一次推送必须强制，以覆盖远程错误的 10万个旧文件
        if first_push:
            print("检测到初次推送，正在使用强制推送模式同步远程历史...")
            push_args.insert(1, "-f") 
            push_args.insert(1, "-u")

        success, _ = run_git(push_args, capture=False)
        if success:
            print(f"成功推送第 {current_batch_num} 批次。")
            first_push = False
        else:
            print("推送失败！请检查网络或远程权限。")
            # 如果强制推送都失败了，说明远程可能有保护分支或者网络彻底断了
            break

    print("\n任务结束。")

if __name__ == "__main__":
    batch_upload()