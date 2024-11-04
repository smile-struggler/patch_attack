import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

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

class InternVL2_8b_inference:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def load_image(self, image_file, input_size=448, max_num=12, resize_shape = None):
        image = Image.open(image_file).convert('RGB')

        if resize_shape is not None:  
            image = image.resize(resize_shape)

        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        return pixel_values

    def inference(self, texts, images, max_new_tokens=1024, batch_size=5):
        assert isinstance(texts, list), "prompts should be a list"

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True)

        if images is None:
            responses = []
            for i in tqdm(range(0, len(texts), batch_size)):  # Add tqdm here
                batch_texts = texts[i:i+batch_size]
                batch_responses = self.model.batch_chat(self.tokenizer, None,
                    num_patches_list=[1] * len(batch_texts),
                    questions=batch_texts,
                    generation_config=generation_config)
                responses.extend(batch_responses)

            return responses
        
        assert isinstance(images, list), "prompts should be a list"
        assert (len(texts) == len(images))

        responses = []
        for i in tqdm(range(0, len(texts), batch_size)):  # Add tqdm here
            batch_texts = texts[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            pixel_values = torch.cat(batch_images, dim=0)
            num_patches_list = [x.size(0) for x in batch_images]
            batch_responses = self.model.batch_chat(self.tokenizer, pixel_values,
                    num_patches_list=num_patches_list,
                    questions=batch_texts,
                    generation_config=generation_config)
            responses.extend(batch_responses)

        return responses
