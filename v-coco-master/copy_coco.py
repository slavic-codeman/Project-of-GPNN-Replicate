import os
import shutil
import torch
from pathlib import Path
from tqdm import tqdm

##TODO: 
"""YOU change path of sources and destination"""
# 目标文件夹路径
destination_dir = '/data1/tangjq/COCO_data/all_data'

# 源文件夹路径列表
source_dirs = ['/data1/tangjq/COCO_data/test2014', '/data1/tangjq/COCO_data/val2014', '/data1/tangjq/COCO_data/train2014']

num_processes=6
def prepare_file_list(source_dirs):
    """从所有源文件夹收集文件路径并返回列表"""
    file_list = []
    for source_dir in source_dirs:
        for file_name in tqdm(os.listdir(source_dir)):
            file_path = os.path.join(source_dir, file_name)
            if os.path.isfile(file_path):  # 只添加文件
                file_list.append(file_path)
    return file_list

def copy_files(rank, files, destination_dir):
    """
    复制文件列表到目标文件夹，每个进程处理文件列表的一个子集。
    
    参数:
    - rank: 当前进程的索引
    - files: 要复制的文件列表
    - destination_dir: 目标文件夹路径
    """
    # 计算每个进程负责的文件范围
    total_files = len(files)
    files_per_process = (total_files + num_processes - 1) // num_processes  # 每个进程处理的文件数量
    start = rank * files_per_process
    end = min(start + files_per_process, total_files)
    
    # 确保目标目录存在
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 使用 tqdm 显示进度条
    with tqdm(total=end - start, desc=f"Process {rank}", position=rank) as pbar:
        for i in range(start, end):
            file = files[i]
            shutil.copy(file, dest_path / Path(file).name)
            pbar.update(1)  # 更新进度条

def main():
    # 准备所有文件路径的列表
    all_files = prepare_file_list(source_dirs)
    
    # 使用 torch.multiprocessing.spawn 启动进程并复制文件
    torch.multiprocessing.spawn(
        copy_files,
        args=(all_files, destination_dir),
        nprocs=num_processes,
        join=True
    )

if __name__ == "__main__":
    main()
