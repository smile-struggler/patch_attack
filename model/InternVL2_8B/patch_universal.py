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

import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os

from conversation import get_conv_template

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, resize_shape = None):
    image = Image.open(image_file).convert('RGB')

    if resize_shape is not None:  
        image = image.resize(resize_shape)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/workshop/crm/checkpoint/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
# pixel_values = load_image('/workshop/crm/checkpoint/InternVL2-8B/examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=64, do_sample=True)

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
    image_list.append(load_image(image_SD, resize_shape=(448, 448)))

with open(target_file, 'r', encoding='utf-8') as file:
    target_data = json.load(file)

    for data in target_data:
        if data['type'] == '01':
            target_list.append(data['target'])

# Take the first 5 and attack
attack_question_num = 20
attack_question_list = question_list[:attack_question_num]
attack_target_list = target_list[:attack_question_num]
attack_image_list = image_list[:attack_question_num]

# attack_question_num = 3
# attack_question_list = question_list[:attack_question_num]
# attack_target_list = target_list[:attack_question_num]
images_tensor = torch.cat(attack_image_list, dim=0)

template = get_conv_template(model.template)
template.system_message = model.system_message

suffix_manager_list = []

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * 1 + IMG_END_TOKEN

prompt_list = []
for id in range(len(attack_question_list)):
    instruction = '<image>\n' + attack_question_list[id]

    instruction = instruction.replace('<image>', image_tokens, 1)
    suffix_manager_list.append(autodan_SuffixManager(tokenizer=tokenizer,
                                            conv_template=template,
                                            instruction=instruction,
                                            target=attack_target_list[id],
                                            adv_string=None))

ori_images = images_tensor.clone().detach().to(model.device)

for suffix_manager in suffix_manager_list:
    input_ids = suffix_manager.get_image_input_ids(adv_string=None).to(model.device)
    prompt = suffix_manager.get_prompt(adv_string=None)
    prompt_list.append(prompt)

tokenizer.padding_side = 'left'
model_inputs = tokenizer(prompt_list, return_tensors='pt', padding=True)
input_ids_list = model_inputs['input_ids'].to(model.device)
attention_mask = model_inputs['attention_mask'].to(model.device)

num_steps = 300
eps=0.6
alpha=16/255
patch_size = 84

normalized_min = [(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
normalized_max = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

min_values = torch.tensor(normalized_min, device = model.device, dtype = ori_images.dtype)
max_values = torch.tensor(normalized_max, device = model.device, dtype = ori_images.dtype)

min_values = min_values.view(1, 3, 1, 1)
max_values = max_values.view(1, 3, 1, 1)

# 攻击类型列表
attack_types = ['full', 'top_left', 'center']

for attack_type in attack_types:
    print(f'开始攻击方式：{attack_type}')
    batch_size, _, img_height, img_width = ori_images.shape
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

    # 攻击迭代
    for i in range(num_steps):
        epoch_start_time = time.time()

        if attack_type == 'full':
            # 扩展到 batch_size 大小
            adv_images = adv_example.repeat(batch_size, 1, 1, 1)
        else:
            # 将补丁应用到每张图像
            adv_images = ori_images.clone()
            # 确定补丁位置
            if attack_type == 'top_left':
                x_start, y_start = 0, 0
            elif attack_type == 'center':
                x_center = img_height // 2
                y_center = img_width // 2
                x_start = x_center - patch_size // 2
                y_start = y_center - patch_size // 2

            # 将补丁应用到图像
            adv_patch_expanded = adv_example.expand(batch_size, -1, -1, -1)
            adv_images[:, :, x_start:x_start+patch_size, y_start:y_start+patch_size] = adv_patch_expanded

        if adv_images.grad is not None:
            adv_images.grad.zero_()
        adv_images.requires_grad_().retain_grad()
        
        image_flags = torch.ones(images_tensor.shape[0])
        output_logits = model(
            pixel_values=adv_images,
            input_ids=input_ids_list,
            attention_mask=attention_mask,
            image_flags=image_flags,
        ).logits
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_list = []
        for id, suffix_manager in enumerate(suffix_manager_list):
            loss_slice = slice(suffix_manager._target_slice.start - 1,  suffix_manager._target_slice.stop - 1)
            valid_output_logits = output_logits[id][attention_mask[id] == 1]
            valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
            loss = crit(valid_output_logits[loss_slice,:], valid_input_ids[suffix_manager._target_slice])
            loss = loss.mean(dim=-1)
            loss_list.append(loss)
        stacked_loss = torch.stack(loss_list)
        loss = stacked_loss.mean()
        loss.backward()

        if attack_type == 'full':
            # 更新 adv_example
            adv_example = adv_example[0] - alpha * adv_images.grad.mean(dim=0).sign()
        else:
            # 只更新补丁区域
            grad = adv_images.grad  # 获取梯度
            # 提取补丁区域的梯度
            grad_patch = grad[:, :, x_start:x_start+patch_size, y_start:y_start+patch_size]
            # 计算平均梯度
            adv_example = adv_example[0] - alpha * grad_patch.mean(dim=0).sign()

        adv_example = torch.clamp(adv_example, min=min_values, max=max_values).detach()

        success_num = 0
        for id, suffix_manager in enumerate(suffix_manager_list):
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
            generation_config['eos_token_id'] = eos_token_id
            valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
            generation_output = model.generate(
                pixel_values=adv_example,
                input_ids=valid_input_ids[:suffix_manager._assistant_role_slice.stop].unsqueeze(0),
                **generation_config
            )
            gen_str  = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            if gen_str[:len(attack_target_list[id])] == attack_target_list[id]:
                is_success=True
            else:
                is_success=False
            success_num+=is_success
            print("**********************")
            print(f"Current Response:\n{gen_str}\n")
            print("**********************")

        images_tensor = images_tensor.repeat(attack_question_num, 1, 1, 1)
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
        
    norm_mean = torch.tensor(IMAGENET_MEAN)
    norm_std = torch.tensor(IMAGENET_STD)
    result_image = images_tensor[0].cpu() * norm_std[:,None,None] + norm_mean[:,None,None]
    result_image = result_image * 255
    result_image = result_image.byte()
    img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
    img.save(f'../../images/patch_noise_image/internvl2_patch_{attack_type}.png')
print('所有攻击已完成。')