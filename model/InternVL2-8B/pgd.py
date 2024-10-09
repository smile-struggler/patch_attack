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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
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

question_file = '/workshop/crm/data/advbench/harmful_behaviors.csv'
result_file = "/data/chenrenmiao/project/Llama-guard2/advbench/advbench_withsuffix_result.json"

result = {}

question_list = []
target_list = []

with open(question_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    next(reader)
    
    for row in reader:
        question_list.append(row[0])
        target_list.append(row[1])

# Take the first 5 and attack
attack_list = [0,1,2,3,4]
attack_question_num = len(attack_list)
attack_question_list = [question_list[i] for i in attack_list]
attack_target_list = [target_list[i] for i in attack_list]

# attack_question_num = 3
# attack_question_list = question_list[:attack_question_num]
# attack_target_list = target_list[:attack_question_num]
images_tensor = torch.zeros((1, 3, 448, 448), dtype=model.dtype).to(model.device)

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

num_steps = 2000
eps=0.6
alpha=16/255

normalized_min = [(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
normalized_max = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

min_values = torch.tensor(normalized_min, device = model.device, dtype = ori_images.dtype)
max_values = torch.tensor(normalized_max, device = model.device, dtype = ori_images.dtype)

min_values = min_values.view(1, 3, 1, 1)
max_values = max_values.view(1, 3, 1, 1)

images_tensor = ori_images.repeat(attack_question_num, 1, 1, 1)

for i in range(num_steps):
    epoch_start_time = time.time()
    if images_tensor.grad is not None:
        images_tensor.grad.zero_()
    images_tensor.requires_grad_().retain_grad()


    # with torch.inference_mode():
        # output_ids = model.generate(
        #     input_ids,
        #     images=images_tensor,
        #     image_sizes=image_sizes,
        #     do_sample=True if temperature > 0 else False,
        #     temperature=temperature,
        #     top_p=top_p,
        #     num_beams=num_beams,
        #     max_new_tokens=max_new_tokens,
        #     use_cache=True,
        # )
    image_flags = torch.ones(images_tensor.shape[0])
    output_logits = model(
        pixel_values=images_tensor,
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
    
    # adv_images = images_tensor[0] - alpha * images_tensor.grad.mean(dim=0).sign()
    # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
    # images_tensor = torch.clamp(ori_images + eta, min=min_values, max=max_values).detach_()

    adv_images = images_tensor[0] - alpha * images_tensor.grad.mean(dim=0).sign()
    images_tensor = torch.clamp(adv_images, min=min_values, max=max_values).detach_()
    
    success_num = 0
    for id, suffix_manager in enumerate(suffix_manager_list):
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
        generation_output = model.generate(
            pixel_values=images_tensor,
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

    if success_num == attack_question_num:
        norm_mean = torch.tensor(IMAGENET_MEAN)
        norm_std = torch.tensor(IMAGENET_STD)
        result_image = images_tensor[0].cpu() * norm_std[:,None,None] + norm_mean[:,None,None]
        result_image = result_image * 255
        result_image = result_image.byte()
        img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
        img.save('../../images/uniform_noise_image/internvl2_pgd.png')
        import pdb;pdb.set_trace()
        break
noise = images_tensor - ori_images

# test