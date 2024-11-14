#!/bin/bash

# 切换到上一层目录
cd ..

# 定义要运行的 Python 脚本及其参数列表
# scripts=(
#     "main.py --model internvl2_8b --experiment_type full --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_full.png"
#     "main.py --model internvl2_8b --experiment_type top_left --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_top_left.png --patch_size 112"
#     "main.py --model internvl2_8b --experiment_type center --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_center.png --patch_size 112"
# )

scripts=(
    "main.py --model qwen2_vl_7b --experiment_type full --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_full.png"
    "main.py --model qwen2_vl_7b --experiment_type top_left --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_top_left.png --patch_size 112"
    "main.py --model qwen2_vl_7b --experiment_type center --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_center.png --patch_size 112"
)

# 定义要使用的 GPU 卡列表
gpus=(0 1 2 3)

# 定义输出目录
output_dir="./process_logs"
mkdir -p $output_dir

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 并行运行每个 Python 脚本
for i in "${!scripts[@]}"; do
    script_and_params="${scripts[$i]}"
    gpu="${gpus[$i]}"
    script_name=$(echo $script_and_params | awk '{print $1}')
    output_file="$output_dir/output_${script_name%.py}_${timestamp}_${i}.log"
    
    echo "Running $script_and_params on GPU $gpu, output to $output_file"
    
    # 使用 CUDA_VISIBLE_DEVICES 指定 GPU 并运行 Python 脚本，将输出重定向到文件并实时更新
    CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python -u $script_and_params > $output_file 2>&1 &
done

# 等待所有后台进程完成
wait

echo "All scripts have finished running."

# 切换回 scripts 目录
cd scripts