#!/bin/bash

# 切换到上一层目录
cd ..

# 定义要运行的 Python 脚本及其参数列表
scripts=(
    "/workshop/crm/project/patch_attack/hades_main.py --model internvl2_8b --experiment_type right --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/hades_internvl2_patch_jb_adv2_img4_cross1_right.png --annotation _jb_adv2_img4_cross1"
    "/workshop/crm/project/patch_attack/hades_main.py --model internvl2_8b --experiment_type right --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/hades_internvl2_patch_jb_adv4_img2_cross0_right.png --annotation _jb_adv4_img2_cross0"
    "/workshop/crm/project/patch_attack/hades_main.py --model internvl2_8b --experiment_type right --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/hades_internvl2_patch_jb_adv4_img2_cross1_right.png --annotation _jb_adv4_img2_cross1"
    "/workshop/crm/project/patch_attack/hades_main.py --model internvl2_8b --experiment_type right --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/hades_internvl2_patch_jb_adv4_img2_cross2_right.png --annotation _jb_adv4_img2_cross2"
   
)

# 定义要使用的 GPU 卡列表
# gpus=(0 1 2 3 4 5 6 7 8)
gpus=(0 1 2 3 4 5 6 7)

# 定义输出目录
output_dir="$(pwd)/process_logs"
mkdir -p "$output_dir"

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 并行运行每个 Python 脚本
for i in "${!scripts[@]}"; do
    script_and_params="${scripts[$i]}"
    
    # 直接从 GPU 列表中获取对应的 GPU
    gpu="${gpus[$i]}"
    
    # 提取目录和脚本名
    script_part=$(echo "$script_and_params" | awk '{print $1}')
    params=$(echo "$script_and_params" | cut -d' ' -f2-)

    script_dir=$(dirname "$script_part")
    script_name=$(basename "$script_part")
    script_command="$script_name $params"

    output_name=$(echo "$script_command" | sed 's/\//|/g')
    output_file="$output_dir/output_${timestamp}_${i}_${output_name%.py}.log"

    
    echo "Running $script_command on GPU $gpu, output to $output_file"
    
    # 切换到相应目录并运行 Python 脚本，将输出重定向到文件并实时更新
    (
        cd "$script_dir" || exit 1
        CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python -u $script_command > "$output_file" 2>&1
    ) &  # 在后台运行脚本并立即返回（不等待脚本运行完成）
    # )
done

# 等待所有后台进程完成
wait

echo "All scripts have finished running."

# 定义要运行的 Python 脚本及其参数列表
scripts=(
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/right/internvl2_8b_jb_adv2_img4_cross1.json"
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/right/internvl2_8b_jb_adv4_img2_cross0.json"
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/right/internvl2_8b_jb_adv4_img2_cross1.json"
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/right/internvl2_8b_jb_adv4_img2_cross2.json"
)

# 定义要使用的 GPU 卡列表
# gpus=(0 1 2 3 4 5 6 7 8)
gpus=(0 1 2 3 4 5 6 7)

# 定义输出目录
output_dir="$(pwd)/process_logs"
mkdir -p "$output_dir"

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 并行运行每个 Python 脚本
for i in "${!scripts[@]}"; do
    script_and_params="${scripts[$i]}"
    
    # 直接从 GPU 列表中获取对应的 GPU
    gpu="${gpus[$i]}"
    
    # 提取目录和脚本名
    script_part=$(echo "$script_and_params" | awk '{print $1}')
    params=$(echo "$script_and_params" | cut -d' ' -f2-)

    script_dir=$(dirname "$script_part")
    script_name=$(basename "$script_part")
    script_command="$script_name $params"

    output_name=$(echo "$script_command" | sed 's/\//|/g')
    output_file="$output_dir/output_${timestamp}_${i}_${output_name%.py}.log"

    
    echo "Running $script_command on GPU $gpu, output to $output_file"
    
    # 切换到相应目录并运行 Python 脚本，将输出重定向到文件并实时更新
    (
        cd "$script_dir" || exit 1
        CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python -u $script_command > "$output_file" 2>&1
    ) &  # 在后台运行脚本并立即返回（不等待脚本运行完成）
done

# 等待所有后台进程完成
wait

echo "All scripts have finished running."