#!/bin/bash

# 切换到上一层目录
cd ..

# 定义要运行的 Python 脚本及其参数列表
scripts=(
    "/workshop/crm/project/patch_attack/model/InternVL2_8B/patch_universal.py"
    "/workshop/crm/project/patch_attack/model/InternVL2_8B/unlimit_full_universal.py"
    "/workshop/crm/project/patch_attack/model/Qwen2_VL_7B_Instruct/patch_universal.py"
    "/workshop/crm/project/patch_attack/model/Qwen2_VL_7B_Instruct/unlimit_full_universal.py"
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


# 定义要运行的 Python 脚本及其参数列表
scripts=(
    "/workshop/crm/project/patch_attack/main.py --model internvl2_8b --experiment_type unlimit_full --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_unlimit_full.png"
    "/workshop/crm/project/patch_attack/main.py --model qwen2_vl_7b --experiment_type unlimit_full --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_unlimit_full.png"
    "/workshop/crm/project/patch_attack/main.py --model internvl2_8b --experiment_type full --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_full.png"
    "/workshop/crm/project/patch_attack/main.py --model qwen2_vl_7b --experiment_type full --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_full.png"
    "/workshop/crm/project/patch_attack/main.py --model internvl2_8b --experiment_type top_left --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_top_left.png"
    "/workshop/crm/project/patch_attack/main.py --model qwen2_vl_7b --experiment_type top_left --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_top_left.png"
    "/workshop/crm/project/patch_attack/main.py --model internvl2_8b --experiment_type center --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/internvl2_patch_center.png"
    "/workshop/crm/project/patch_attack/main.py --model qwen2_vl_7b --experiment_type center --patch_path /workshop/crm/project/patch_attack/images/patch_noise_image/qwen2vl_patch_center.png"
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
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/center"
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/full"
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/top_left"
    "/workshop/crm/project/patch_attack/judge.py --input_path /workshop/crm/project/patch_attack/results/generate/unlimit_full"
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