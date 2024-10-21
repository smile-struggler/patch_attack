import json
import torch
import time
import sys
sys.path.append("../..")
sys.path.append('/workshop/crm/checkpoint/InternVL2-8B')

import torch.nn as nn

import csv

from PIL import Image

from tqdm import tqdm

from utils.string_utils import autodan_SuffixManager

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

torch.autograd.set_detect_anomaly(True)

image_mean = (
    0.48145466,
    0.4578275,
    0.40821073
  )
image_std = (
    0.26862954,
    0.26130258,
    0.27577711
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

# default processer
processor = AutoProcessor.from_pretrained("/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct")

question_file = '/workshop/crm/data/MM-SafetyBench/processed_questions/01-Illegal_Activitiy.json'
img_file_path = '/workshop/crm/data/MM-SafetyBench/img/01-Illegal_Activitiy'
target_file = '/workshop/crm/project/patch_attack/goals-and-targets_detailed.json'
result = {}

question_list = []
image_list = []
target_list = []

with open(question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

for data_id in data:
    question_list.append(data[data_id]['Rephrased Question'])

    image_SD = os.path.join(img_file_path, 'SD_TYPO', f'{data_id}.jpg')
    image = Image.open(image_SD).convert('RGB')
    image = image.resize((448, 448))
    image_list.append(image)

with open(target_file, 'r', encoding='utf-8') as file:
    target_data = json.load(file)

    for data in target_data:
        if data['type'] == '01':
            target_list.append(data['target'])

# Take the first 5 and attack
attack_question_num = 50
attack_question_list = question_list[:attack_question_num]
attack_target_list = target_list[:attack_question_num]
attack_image_list = image_list[:attack_question_num]

suffix_manager_list = []

prompt_list = []
for id in range(len(attack_question_list)):
    suffix_manager_list.append(autodan_SuffixManager(tokenizer=processor,
                                            conv_template=None,
                                            instruction=attack_question_list[id],
                                            target=attack_target_list[id],
                                            adv_string=None))



for id, suffix_manager in enumerate(suffix_manager_list):
    input_ids, prompt = suffix_manager.get_qwen_image_input_ids(adv_string=None, image=image_list[id])
    input_ids = input_ids.to(model.device)
    prompt_list.append(prompt)

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    for msg in prompt_list
]
image_inputs, video_inputs = process_vision_info(prompt_list)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

input_ids_list = inputs['input_ids'].to(model.device)
attention_mask = inputs['attention_mask'].to(model.device)

num_steps = 300
eps=0.6
alpha=16/255
patch_size = 112

normalized_min = [(0 - m) / s for m, s in zip(image_mean, image_std)]
normalized_max = [(1 - m) / s for m, s in zip(image_mean, image_std)]

ori_images = inputs['pixel_values'].to(model.device)

min_values = torch.tensor(normalized_min, device = model.device, dtype = ori_images.dtype)
max_values = torch.tensor(normalized_max, device = model.device, dtype = ori_images.dtype)

min_values = min_values.view(1, 3, 1, 1)
max_values = max_values.view(1, 3, 1, 1)



# 攻击类型列表
attack_types = ['full', 'top_left', 'center']

for attack_type in attack_types:
    print(f'开始攻击方式：{attack_type}')
    batch_size, _, img_height, img_width = (attack_question_num, 3, 448, 448)
    image_size = (img_height, img_width)

    # 初始化对抗样本或补丁
    if attack_type == 'full':
        # 第一种攻击：从全零图像开始
        adv_example = torch.zeros((1, 3, *image_size),device = model.device, dtype=ori_images.dtype, requires_grad=True)
    elif attack_type in ['top_left', 'center']:
        # 第二种和第三种攻击：补丁攻击，从全零补丁开始
        adv_example = torch.zeros((1, 3, patch_size, patch_size),device = model.device, dtype=ori_images.dtype, requires_grad=True)
    else:
        raise ValueError("无效的攻击类型")

    temporal_patch_size = processor.image_processor.temporal_patch_size
    model_patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    grid_t, grid_h, grid_w = inputs['image_grid_thw'][0]

    # 假设每个小批次大小为 batch_size_per_step
    batch_size_per_step = 6
    num_batches = (batch_size + batch_size_per_step - 1) // batch_size_per_step  # 计算总批次数

    # 攻击迭代
    for i in range(num_steps):
        epoch_start_time = time.time()

        if attack_type == 'full':
            # 扩展到 batch_size 大小
            adv_images = adv_example.repeat(batch_size * 2, 1, 1, 1)
        else:
            # 将补丁应用到每张图像
            adv_images = ori_images.clone()
            adv_images = adv_images.view(attack_question_num, 
                    adv_images.shape[0] // attack_question_num, 
                    adv_images.shape[1]).reshape(
                -1, grid_h // merge_size, grid_w // merge_size, 
                merge_size, merge_size, 3, 
                temporal_patch_size, model_patch_size, model_patch_size
            )
            adv_images = adv_images.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
            adv_images = adv_images.reshape(-1, 3, grid_h * model_patch_size, grid_w * model_patch_size)

            # 确定补丁位置
            if attack_type == 'top_left':
                x_start, y_start = 0, 0
            elif attack_type == 'center':
                x_center = img_height // 2
                y_center = img_width // 2
                x_start = x_center - patch_size // 2
                y_start = y_center - patch_size // 2

            # 将补丁应用到图像
            adv_patch_expanded = adv_example.expand(batch_size * 2, -1, -1, -1)
            adv_images[:, :, x_start:x_start+patch_size, y_start:y_start+patch_size] = adv_patch_expanded

        accumulated_grad = torch.zeros_like(adv_example)
        total_samples = 0

        for batch_idx in range(num_batches):
            # 获取当前小批次数据
            start_idx = batch_idx * batch_size_per_step
            end_idx = min(start_idx + batch_size_per_step, batch_size)
            current_batch_size = end_idx - start_idx

            # 如果当前小批次是空的，跳过
            if current_batch_size == 0:
                continue
            
            temp_adv_images = adv_images[start_idx * 2:end_idx * 2].clone().detach().requires_grad_(True)
            # 处理当前小批次
            
            if temp_adv_images.shape[0] == 1:
                temp_adv_images= temp_adv_images.repeat(temporal_patch_size, 1, 1, 1)
            channel = temp_adv_images.shape[1]
            grid_t = temp_adv_images.shape[0] // temporal_patch_size
            grid_h, grid_w = 448 // model_patch_size, 448 // model_patch_size
            temp_adv_images = temp_adv_images.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                model_patch_size,
                grid_w // merge_size,
                merge_size,
                model_patch_size,
            )
            temp_adv_images = temp_adv_images.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            temp_adv_images = temp_adv_images.reshape(
                grid_t * grid_h * grid_w, channel * temporal_patch_size * model_patch_size * model_patch_size
            )

            if temp_adv_images.grad is not None:
                temp_adv_images.grad.zero_()
            temp_adv_images.requires_grad_().retain_grad()

            output_logits = model(
                input_ids=input_ids_list[start_idx:end_idx],
                attention_mask=attention_mask[start_idx:end_idx],
                pixel_values=temp_adv_images,
                image_grid_thw=inputs['image_grid_thw'][start_idx:end_idx],
            ).logits
            
            crit = nn.CrossEntropyLoss(reduction='none')
            loss_list = []
            # 只使用对应子批次的suffix_manager
            for id in range(start_idx, end_idx):
                suffix_manager = suffix_manager_list[id]
                loss_slice = slice(suffix_manager._target_slice.start - 1, suffix_manager._target_slice.stop - 1)
                valid_output_logits = output_logits[id - start_idx][attention_mask[id] == 1]  # 注意索引调整
                valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
                loss = crit(valid_output_logits[loss_slice, :], valid_input_ids[suffix_manager._target_slice])
                loss = loss.mean(dim=-1)
                loss_list.append(loss)
            stacked_loss = torch.stack(loss_list)
            loss = stacked_loss.mean()
            loss.backward()

            # 累积梯度并更新样本数
            grad = temp_adv_images.grad
            grad = grad.view(current_batch_size, 
                            grad.shape[0] // current_batch_size, 
                            grad.shape[1])[0].reshape(
                -1, grid_h // merge_size, grid_w // merge_size, 
                merge_size, merge_size, 3, 
                temporal_patch_size, model_patch_size, model_patch_size
            )
            grad = grad.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
            grad = grad.reshape(-1, 3, grid_h * model_patch_size, grid_w * model_patch_size)

            if attack_type == 'full':
                accumulated_grad += grad.sum(dim=0)
            else:
                grad_patch = grad[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
                accumulated_grad += grad_patch.sum(dim=0)
            
            total_samples += current_batch_size

        # 计算最终的平均梯度
        avg_grad = accumulated_grad / total_samples

        # 更新对抗样本或补丁
        adv_example = adv_example[0] - alpha * avg_grad.sign()

        # 进行范围限制
        adv_example = torch.clamp(adv_example, min=min_values, max=max_values).detach()
        
        success_num = 0

        if i > 200:
            for id, suffix_manager in enumerate(suffix_manager_list):
                valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
                valid_input_ids = valid_input_ids[:suffix_manager._assistant_role_slice.stop].unsqueeze(0)

                generated_ids = model.generate(
                    input_ids=valid_input_ids,
                    pixel_values=adv_images[id],
                    attention_mask=torch.ones(valid_input_ids.shape[1], device=model.device).unsqueeze(0),
                    image_grid_thw=inputs['image_grid_thw'][0].unsqueeze(0),
                    max_new_tokens=64
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(valid_input_ids, generated_ids)
                ]
                gen_str = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                if gen_str[:len(attack_target_list[id])] == attack_target_list[id]:
                    is_success=True
                else:
                    is_success=False

                success_num+=is_success
                print("**********************")
                print(f"Current Response:\n{gen_str}\n")
                print("**********************")

        epoch_end_time = time.time()
        epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)
        print(
            "################################\n"
            # f"Current Data: {i}/{len(harmful_data.goal[args.start:])}\n"
            f"Current Epoch: {i}/{num_steps}\n"
            f"Passed:{success_num}/{attack_question_num}\n"
            f"Loss:{loss.item()}\n"
            f"Epoch Cost:{epoch_cost_time}\n"
            # f"Current Suffix:\n{best_new_adv_suffix}\n"
            
            "################################\n")
        
    image_mean = torch.tensor(image_mean)
    image_std = torch.tensor(image_std)
    result_image = adv_example[0].cpu() * image_std[:,None,None] + image_mean[:,None,None]
    result_image = result_image * 255
    result_image = result_image.byte()
    img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
    img.save(f'../../images/patch_noise_image/qwen2vl_patch_{attack_type}.png')
print('所有攻击已完成。')