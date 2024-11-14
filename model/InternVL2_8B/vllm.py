from vllm import LLM, SamplingParams
from PIL import Image
import torch
import torchvision.transforms as transforms

llm = LLM(model="/workshop/crm/checkpoint/InternVL2-8B",trust_remote_code=True)

images_tensor = torch.zeros((3, 448, 448))
to_pil = transforms.ToPILImage()
image_obj = to_pil(images_tensor)
image_obj.save('./test.jpg')

image_obj = Image.open('./test.jpg')
sampling_params = SamplingParams(max_tokens=1024, temperature=0)
outputs = llm.generate(
    [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_obj},
        },
        # {
        #     "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
        #     "multi_modal_data": {"image": image_obj},
        # }
    ], sampling_params
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)