import os

# from openai import OpenAI
import http.client
import json



class LLM():
    def __init__(self):
        self.base_url = DEFAULT_BASE_URL
        self.api_key = DEFAULT_API_KEY
        self.llm_model = "claude-3-5-sonnet"
        # self.llm_model = "gpt-4o-mini"

    def request(self,message):
        conn = http.client.HTTPSConnection(self.base_url)
        payload = json.dumps({
            "model": self.llm_model,
            "messages": [
                {"role": "user", "content": message}
            ]
        })
        headers = {
            'Authorization': f' {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        data = conn.getresponse().read()
        response = json.loads(data.decode("utf-8"))
        print(response)
        answer = response['choices'][0]['message']['content']
        return answer


    def get_model_list(self):
        conn = http.client.HTTPSConnection(self.base_url)
        conn.request("GET", "/v1/models", headers=headers)
        data = conn.getresponse().read()
        response = json.loads(data.decode("utf-8"))
        print(response)

if __name__ == '__main__':
    llm = LLM()
    llm.get_model_list()

