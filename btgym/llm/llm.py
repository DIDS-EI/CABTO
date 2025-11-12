import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime
from btgym.dataclass.cfg import cfg


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class LLM:
    def __init__(self):
        # check necessary environment variables
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set the environment variable OPENAI_API_KEY (and OPENAI_BASE_URL)")
        if not os.getenv("OPENAI_BASE_URL"): 
            raise ValueError("Please set the environment variable OPENAI_BASE_URL (and OPENAI_API_KEY)")
        self.client = OpenAI()

        # self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')

    def custom_request_no_stream(self, messages):
        response = self.client.chat.completions.create(
            model=cfg.llm_model, # gpt-4o
            messages=messages,
            temperature=cfg.llm_temperature #0
            # max_tokens=cfg.llm_max_tokens*5 #2048
        )
        if isinstance(response, str):  # check if it is a string
            response = json.loads(response)  # convert to Python object
        output = response.choices[0].message.content
        print(f"custom_request_no_stream: {output}")
        return output

    def custom_request(self, messages):
        # build prompt 
        stream = self.client.chat.completions.create(model=cfg.llm_model,
                                                        messages=messages,
                                                        temperature=cfg.llm_temperature,
                                                        stream=True)
        output = ""
        print("", end="", flush=True)  # clear current line
        print("Generating answer...\n")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                output += content
        print("\n")
        return output


    def _build_prompt_with_img(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _build_prompt(self, instruction):
        return [
            {
                "role": "user",
                "content": instruction
            }
        ]

    def request(self, message):
        stream = self.client.chat.completions.create(model=cfg.llm_model,
                                                            messages=message,
                                                            temperature=cfg.llm_temperature,
                                                            stream=True)
        output = ""
        print("", end="", flush=True)  # clear current line
        print("Generating answer...\n")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                output += content
        print("\n")
        return output

    def request_instruction(self, instruction):
        """
        Args:
            instruction (str): instruction for the query 
        Returns:
            save_dir (str): directory where the constraints 
        """
        # build prompt 
        messages = self._build_prompt(instruction)
        output = self.request(messages)
        return output

    def get_model_list(self):
        models = self.client.models.list()
        return [model.id for model in models]

    def embedding(self, instruction):
        response = self.client.embeddings.create(
            model=cfg.llm_embedding_model,
            input=instruction
        )
        return response.data[0].embedding

if __name__ == "__main__":
    llm = LLM()
    # print(llm.request("generate a python code to print 'hello world'"))
    # print(llm.get_model_list())

    embedding = llm.embedding("reorient the white pen and drop it upright into the black pen holder")
    print(embedding)
    print(len(embedding))