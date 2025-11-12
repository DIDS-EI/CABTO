from openai import OpenAI
from dexrl import global_config
from dexrl.utils import configclass

@configclass
class LMCfg:
    model_name = "gpt-4o-mini" # 模型倍率 0.5
    # model_name = "gpt-4o" # 模型倍率 7.5
    # model_name = "gpt-4" # 模型倍率 15

    embedding_model_name = "text-embedding-ada-002"


class LM:
    def __init__(self, cfg: LMCfg=None):
        self.cfg = LMCfg() if cfg is None else cfg
        self.client = OpenAI(api_key=global_config.api_key, base_url=global_config.base_url)

    def request(self, messages: list[dict]):
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages
        )
        return response.choices[0].message.content

    def embed(self, text: str):
        response = self.client.embeddings.create(
            model=self.cfg.model_name,
            input=text
        )
        return response.data[0].embedding

if __name__ == "__main__":
    lm = LM()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ]
    print(lm.request(messages))