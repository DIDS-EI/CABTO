from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
import os



def draw_points_on_image(image, points, output_path):
    # 创建图片副本以免修改原图
    img_with_points = image.copy()
    
    # 转换为可绘制格式
    draw = ImageDraw.Draw(img_with_points)
    
    radius = 20
    # 为每个点画一个红色圆圈和序号
    for i, point in enumerate(points):
        x, y = point
        # 画一个半径为20像素的红色圆圈
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='green')
        # 在圆圈上方添加序号，使用更大的字体
        draw.text((x-10, y-radius-30), str(i+1), fill='white', font=None)
    # 保存并显示结果
    img_with_points.save(output_path)

class MolmoModel:   
    def __init__(self):
        self.repo_name = f"cyan2k/molmo-7B-D-bnb-4bit"
        self.arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
        self.processor = None
        self.model = None
        self.device = "cuda"
        self.load_model()

    def load_model(self):
        # load the processor
        self.processor = AutoProcessor.from_pretrained(self.repo_name, **self.arguments)

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(self.repo_name, **self.arguments)

    def point_qa(self, query, image_path):
        # load image and prompt
        image = Image.open(image_path)
        inputs = self.processor.process(
            images=[image],
            text=f"point_qa: {query}",
        )

        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text
        # print the generated text
        # print(generated_text)


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.abspath(__file__))
    molmo = MolmoModel()
    response = molmo.point_qa(
                   query='Point out the important parts for doing the task. The task is "reorient the white pen and drop it upright into the black pen holder".',
                   image_path=os.path.join(DIR, "camera_0_rgb.png"),
                )
    print(response)