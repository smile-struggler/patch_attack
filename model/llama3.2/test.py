import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "/workshop/crm/checkpoint/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "/workshop/crm/data/MM-SafetyBench/img/01-Illegal_Activitiy/SD_TYPO/0.jpg"
image = Image.open(url)
image = image.resize((448, 448))

messages = [
    {
        "role": "user", 
        "content": [
            {
                "type": "image"
            },
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, size = {"height": 448, "width": 448}, return_tensors="pt").to(model.device)

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

image_mean = torch.tensor(image_mean)
image_std = torch.tensor(image_std)
result_image = inputs['pixel_values'][0][0][0].cpu() * image_std[:,None,None] + image_mean[:,None,None]
result_image = result_image * 255
result_image = result_image.byte()
img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
img.save(f'test.png')

import pdb;pdb.set_trace()

output = model.generate(**inputs, max_new_tokens=30)
prompt_len = inputs.input_ids.shape[-1]
generated_ids = output[:, prompt_len:]
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(generated_text)
