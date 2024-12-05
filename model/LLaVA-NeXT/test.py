from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

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

pretrained = "/workshop/crm/checkpoint/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None) # Add any other thing you want to pass in llava_model_args
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

model.eval()
model.tie_weights()
a = 1

url = "/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg"
image = Image.open(url)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# image_mean = torch.tensor(image_mean)
# image_std = torch.tensor(image_std)
# result_image = image_tensor[0].cpu() * image_std[:,None,None] + image_mean[:,None,None]
# result_image = result_image * 255
# result_image = result_image.byte()
# import pdb;pdb.set_trace()
# img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
# img.save(f'./test.png')

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.tokenizer = tokenizer
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
input_ids=input_ids.repeat(3,1)
images_tensor = [image_tensor[0].clone() for _ in range(3)]
image_sizes = [image.size] * 3
# import pdb;pdb.set_trace()
# test = model(input_ids=input_ids,images=images_tensor,image_sizes=image_sizes,output_attentions=True)
# import pdb;pdb.set_trace()

image_sizes = [image.size] * 3
cont = model.generate(
    input_ids,
    images=images_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=256,
)
# import pdb;pdb.set_trace()
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
for i in text_outputs:
    print(i)
    print("***********")

import pdb;pdb.set_trace()
# The image shows a radar chart, also known as a spider chart or a web chart, which is a type of graph used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along each axis and connected to form a polygon.\n\nIn this particular radar chart, there are several axes labeled with different variables, such as "MM-Vet," "LLaVA-Bench," "SEED-Bench," "MMBench-CN," "MMBench," "TextVQA," "VizWiz," "GQA," "BLIP-2," "InstructBLIP," "Owen-VL-Chat," and "LLaVA-1.5." These labels suggest that the chart is comparing the performance of different models or systems across various benchmarks or tasks, such as machine translation, visual question answering, and text-based question answering.\n\nThe chart is color-coded, with each color representing a different model or system. The points on the chart are connected to form a polygon, which shows the relative performance of each model across the different benchmarks. The closer the point is to the outer edge of thep
# The image shows a radar chart, also known as a spider chart or a web chart, which is a type of graph used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along each axis and connected to form a polygon.\n\nIn this particular radar chart, there are several axes labeled with different variables, such as "MM-Vet," "LLaVA-Bench," "SEED-Bench," "MMBench-CN," "MMBench," "TextVQA," "VizWiz," "GQA," "BLIP-2," "InstructBLIP," "Owen-VL-Chat," and "LLaVA-1.5." These labels suggest that the chart is comparing the performance of different models or systems across various benchmarks or tasks, such as machine translation, visual question answering, and text-based question answering.\n\nThe chart is color-coded, with each color representing a different model or system. The points on the chart are connected to form a polygon, which shows the relative performance of each model across the different benchmarks. The closer the point is to the outer edge of the
# The image shows a radar chart, also known as a spider chart or a web chart, which is a type of graph used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along each axis and connected to form a polygon.\n\nIn this particular radar chart, there are several axes labeled with different variables, such as "MM-Vet," "LLaVA-Bench," "SEED-Bench," "MMBench-CN," "MMBench," "TextVQA," "VizWiz," "GQA," "BLIP-2," "InstructBLIP," "Owen-VL-Chat," and "LLaVA-1.5." These labels suggest that the chart is comparing the performance of different models or systems across various benchmarks or tasks, such as machine translation, visual question answering, and text-based question answering.\n\nThe chart is color-coded, with each color representing a different model or system. The points on the chart are connected to form a polygon, which shows the relative performance of each model across the different benchmarks. The closer the point is to the outer edge of the
# model(input_ids=input_ids,images=image_tensor,image_sizes=image_sizes[0],output_attentions=True)