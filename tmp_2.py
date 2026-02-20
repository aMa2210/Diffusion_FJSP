import os
import subprocess

# --- 配置区域 ---
TARGET_DIR = "Trainset"  # 包含那10万个文件的文件夹
BATCH_SIZE = 100  # 每批次处理的文件数量（Windows 下不宜过大，避免命令行超长）
REMOTE = "origin"  # 远程仓库名
BRANCH = "main"  # 分支名


# ----------------

def run_git(args, capture=True):
    try:
        subprocess.run(
            ["git"] + args,
            check=True,
            capture_output=capture,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        print(f"Git 错误: {err}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 git 命令，请确保已安装 Git 并加入 PATH。")
        return False
    return True


def batch_upload():
    first_push = True  # 首次推送时使用 -u 设置上游分支

    try:
        # 0. 先检查是否有「已暂存但未提交」的文件（例如之前运行过 git add 但没提交）
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
        )
        # Git 输出统一用正斜杠，兼容可能的反斜杠
        prefix_slash = TARGET_DIR + "/"
        prefix_back = TARGET_DIR + "\\"
        def under_target(p):
            return p == TARGET_DIR or p.startswith(prefix_slash) or p.startswith(prefix_back)

        if result.returncode == 0:
            staged = [
                f for f in result.stdout.splitlines()
                if f.strip() and under_target(f.replace("\\", "/"))
            ]
            if staged:
                print(f"发现 {len(staged)} 个已暂存文件，先从索引移除，将按批次提交推送...")
                # 无提交时 reset 可能无效，用 rm --cached 从索引移除
                if not run_git(["rm", "-r", "--cached", TARGET_DIR]):
                    if not run_git(["reset", "HEAD", "--", TARGET_DIR + "/"]):
                        print("取消暂存失败，已停止。")
                        return

        # 1. 获取待添加文件：先列「未追踪」，再筛出 TARGET_DIR 下（避免路径参数在部分环境不生效）
        print("正在扫描待添加文件...")
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        if result.returncode != 0:
            print(f"扫描失败: {result.stderr or result.stdout}")
            return
        all_others = [f.replace("\\", "/") for f in result.stdout.splitlines() if f.strip()]
        files = [f for f in all_others if under_target(f)]
    except FileNotFoundError:
        print("错误: 未找到 git 命令，请确保已安装 Git 并加入 PATH。")
        return
    except Exception as e:
        print(f"扫描文件时出错: {e}")
        return

    total_files = len(files)
    print(f"共 {total_files} 个文件将按每批 {BATCH_SIZE} 个提交并推送。")
    if total_files == 0:
        print("没有需要添加的文件（Trainset 已全部提交）。")
        # 若本地有提交未推送（例如 upstream is gone），仍执行一次推送
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("正在推送当前分支到远程（设置上游）...")
            if run_git(["push", "-u", REMOTE, BRANCH], capture=False):
                print("推送完成。")
        else:
            r2 = subprocess.run(
                ["git", "rev-list", "--count", "@{u}..HEAD"],
                capture_output=True,
                text=True,
            )
            if r2.returncode == 0 and int(r2.stdout.strip() or "0") > 0:
                print("存在未推送的提交，正在推送...")
                if run_git(["push", REMOTE, BRANCH], capture=False):
                    print("推送完成。")
        return

    for i in range(0, total_files, BATCH_SIZE):
        batch = files[i : i + BATCH_SIZE]

        # 批量添加文件（每批单独 add，避免命令行过长）
        print(f"正在处理第 {i + 1} 到 {i + len(batch)} 个文件...")
        add_ok = subprocess.run(["git", "add"] + batch, capture_output=True, text=True)
        if add_ok.returncode != 0:
            print(f"git add 失败: {add_ok.stderr or add_ok.stdout}")
            break

        # 提交
        commit_msg = f"batch upload {i // BATCH_SIZE + 1} ({len(batch)} files)"
        if not run_git(["commit", "-m", commit_msg]):
            print("提交失败，跳过该批次。")
            continue

        # 推送（首次使用 -u 设置 upstream）
        print("正在推送到远程仓库...")
        push_args = ["push", REMOTE, BRANCH]
        if first_push:
            push_args.insert(1, "-u")  # git push -u origin main
        if run_git(push_args, capture=False):  # 推送时不捕获输出，便于看进度/认证
            print(f"成功推送第 {i // BATCH_SIZE + 1} 批次。")
            first_push = False
        else:
            print("推送失败，停止运行。请检查网络、远程地址或缓冲区设置。")
            break

    print("全部完成。")


if __name__ == "__main__":
    batch_upload()