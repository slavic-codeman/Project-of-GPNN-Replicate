#!/bin/bash

zip_folder='/data1/tangjq/COCO'  # 存放 zip 文件的目录
destination_folder='/data1/tangjq/COCO_data/all_data'  # 解压到的目标目录

# 获取所有 zip 文件
zip_files=$(find "$zip_folder" -type f -name "*.zip" -and -name "*images*")

# 循环处理每个 zip 文件
for zip_file in $zip_files; do
    # 获取当前 zip 文件中的文件数量
    total_files=$(unzip -l "$zip_file" | awk 'NR>3 {print $1}' | tail -n +2 | wc -l)
    
    # 显示当前 zip 文件的解压状态
    echo "正在解压: $zip_file"

    # 初始化进度
    current_file=0

    # 解压每个文件，并显示进度
    unzip -l "$zip_file" | awk 'NR>3 {print $4}' | tail -n +2 | while read file; do
        # 解压文件
        unzip -o "$zip_file" "$file" -d "$destination_folder"
        
        # 更新进度条（百分比）
        current_file=$((current_file + 1))
        progress=$((current_file * 100 / total_files))
        
        # 打印进度
        printf "\r解压进度: [%3d%%] (%d/%d)" "$progress" "$current_file" "$total_files"
    done

    # 换行显示解压完成
    echo -e "\n$zip_file 解压完成！"
done
