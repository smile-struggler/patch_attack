import os
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tokenizer, model, image_processor, context_len = load_pretrained_model('/workshop/crm/checkpoint/llama3-llava-next-8b', None, 'llava-v1.5-7b')

input_text = ['<image>\nThis is the first sentence<image>.', '<image>\nThis is the second sentence.\n<image>']
image_files = [['/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg',
                '/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg'],
               ['/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg',
                '/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg']]

input_text = ['<image>\nThis is the first sentence.', '<image>\nThis is the second sentence.']
image_files = [['/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg'],
               ['/workshop/crm/project/patch_attack/model/LLaVA-NeXT/llava_v1_5_radar.jpg']]

input_ids = [tokenizer_image_token(s, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for s in input_text]
tokenizer.pad_token_id = tokenizer.eos_token_id
input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id).cuda()
print('input_ids.shape:', input_ids.shape)


image_lists = [torch.cat([image_processor.preprocess(Image.open(f).resize((336,336)), return_tensors='pt')['pixel_values'] for f in files]).half().cuda() for files in image_files]
image_lists = torch.stack(image_lists)
print('image_lists.shape:', image_lists.shape)
print('image_lists[1].shape:', image_lists[1].shape)
image_sizes = [(336,336)] * 2
import pdb;pdb.set_trace()
model(input_ids=input_ids, images = image_lists,image_sizes=image_sizes)