import time
import psutil
import os
import sys
import multiprocessing
import hashlib

# 配置参数
CHUNK_SIZE_MB = 1000          # 每个进程每次分配的大块大小
NUM_PROCESSES = 256           # 并发进程数，用来拉高总压力速度

def memory_worker(proc_id):
    pid = os.getpid()
    sys.stdout.write(f"[进程 {proc_id} / PID {pid}] 启动内存抢占与 CPU 计算...\n")
    sys.stdout.flush()
    data = []
    
    while True:
        try:
            # 申请物理内存
            chunk = bytearray(CHUNK_SIZE_MB * 1024 * 1024)
            # 强行刷新每页 (4KB)，占用物理内存 (RSS)
            slice_len = len(chunk) // 4096
            chunk[::4096] = b'\xFF' * slice_len
                
            data.append(chunk)
            
            if len(data) % 3 == 0:
                current_rss = psutil.Process(pid).memory_info().rss / (1024**3)
                sys.stdout.write(f"\r[进程 {proc_id}] 物理内存 {current_rss:.2f} GB | 正使用积攒的 {len(data)} 个数据块猛烈计算中...")
                sys.stdout.flush()
            
            # --- 消耗 CPU 和内存总线的核心计算逻辑 ---
            # 开始遍历已经霸占的内存，在 C 层面进行大规模密集运算
            for c in data:
                # 疯狂读取：MD5 哈希底层用的是 C 语言且可以释放 GIL，它能完美榨干单核 CPU 并吞噬极大的内存读取带宽
                _ = hashlib.md5(c).hexdigest()
                
                # 疯狂写入：更新内存位的值，让操作系统以为它正在被活跃使用，禁止系统将其转移至 Swap (强制变脏页)
                c[0] = (c[0] + 1) % 256
                c[-1] = (c[-1] + 1) % 256
                
        except Exception:
            time.sleep(1)

def stress_memory():
    # 获取系统内存状态
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    
    print(f"=== 系统极限内存并发压测工具 ===")
    print(f"总物理内存: {total_gb:.2f} GB")
    print(f"并发进程数: {NUM_PROCESSES}")
    print(f"单次分配块: {CHUNK_SIZE_MB} MB (带真实物理数据写入)")
    print("-" * 50)
    print("警告：此模式将并发地耗尽所有可用的物理内存。")
    print("系统可能会变得极其卡顿，触发 OOM Killer。")
    print("随时按 Ctrl+C 可停止所有子进程释放内存。")
    print("-" * 50)
    
    processes = []
    
    try:
        # 启动多个独立的内存吞噬进程
        for i in range(NUM_PROCESSES):
            p = multiprocessing.Process(target=memory_worker, args=(i,))
            p.daemon = True
            p.start()
            processes.append(p)
            
        # 主进程：监控整体压力
        last_time = time.time()
        while True:
            time.sleep(2)
            # 如果有进程被 OOM Killer 杀死，自动重启
            for i, p in enumerate(processes):
                if not p.is_alive():
                    sys.stdout.write(f"\n[警告] 进程 {i} (PID {p.pid}) 被 OOM 杀除。正在重新启动以维持压力...\n")
                    sys.stdout.flush()
                    new_p = multiprocessing.Process(target=memory_worker, args=(i,))
                    new_p.daemon = True
                    new_p.start()
                    processes[i] = new_p
            
            # 打印系统整体情况
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            sys.stdout.write(f"\n---> [系统监控] 全局剩余 RAM: {mem.available/(1024**3):.2f} GB | 剩余 Swap: {swap.free/(1024**3):.2f} GB <---")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n\n[停止] 收到终止信号。正在清理所有子进程...")
        for p in processes:
            p.terminate()
        print("所有子进程已终止，内存已归还给系统。")

if __name__ == "__main__":
    stress_memory()
