from openai import OpenAI
from openai import AuthenticationError, APIError, RateLimitError


class LLM():
    def __init__(self,request_model="gpt-4o-mini"):
        self.client = OpenAI(
            # base_url="https://api.xty.app/v1",
            # api_key="sk-Wtr2qAJy2GlsQCjhm6GbSuXBAlfYt6g53Z3NeeNzv5wTGFR3"
            base_url="https://api.dwyu.top/v1",
            api_key="sk-Gtk0rmTrrRjEOj8Kru5exXuOKwpwSqiR3intYCMvtIBzLqzN"
        )
        self.request_model = request_model
        # self.request_model = "gpt-4o-mini"
        # self.request_model = "o4-mini-2025-04-16"

    def request(self,message):
        try:
            # 如果 message 是字符串，转换为消息列表格式
            if isinstance(message, str):
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": message}
                ]
            else:
                # 如果已经是列表格式，直接使用
                messages = message

            completion = self.client.chat.completions.create(
              # model="gpt-4-turbo-preview", 	# 模型倍率 15.00
            #   model="gpt-3.5-turbo",
            #   model="gpt-4o-2024-08-06",
              # model="gpt-4o-2024-08-06-preview",
            #   model="gpt-4o",   #  模型倍率 7.50	
              model=self.request_model,   #  模型倍率 0.50
              # model="o4-mini-2025-04-16", # 模型倍率 1.65
              # messages=[
              #   {"role": "system", "content": ""},#You are a helpful assistant.
              #   {"role": "user", "content": question}
              # ]
                messages=messages
            )

            return completion.choices[0].message.content
        except AuthenticationError as e:
            error_msg = str(e)
            if "quota exhausted" in error_msg.lower() or "quota" in error_msg.lower():
                print("❌ API 密钥配额已耗尽。请检查您的 API 密钥配额或更换新的 API 密钥。")
            else:
                print("❌ API 认证失败。请检查您的 API 密钥是否正确。")
            raise
        except RateLimitError as e:
            print("⚠️ API 请求频率超限。请稍后重试。")
            raise
        except APIError as e:
            print(f"❌ API 请求失败: {str(e)}")
            raise

    def stream_request(self,messages):
        try:
            print("开始流式请求...")
            response = self.client.chat.completions.create(
                model=self.request_model,  # 或者其他模型
                messages=messages,
                stream=True  # 启用流式输出
            )

            # 初始化一个变量来存储最终的返回值
            full_response = ""
            chunk_count = 0

            # 遍历流式返回的内容
            for chunk in response:
                chunk_count += 1
                # 检查 choices 是否存在且不为空
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                
                # 检查是否有 finish_reason，如果有且为 "stop"，说明流式响应结束
                if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason == "stop":
                    print(f"\n流式请求完成，共接收 {chunk_count} 个数据块")
                    break
                
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                    full_response += content
                    print(content, end="", flush=True)  # 实时打印内容，确保立即输出

            print("\n\n最终返回值：")
            return full_response

        except (AuthenticationError, RateLimitError) as e:
            # 认证错误或配额错误不应该回退，直接抛出
            error_msg = str(e)
            if isinstance(e, AuthenticationError) and ("quota exhausted" in error_msg.lower() or "quota" in error_msg.lower()):
                print("❌ API 密钥配额已耗尽。请检查您的 API 密钥配额或更换新的 API 密钥。")
            elif isinstance(e, AuthenticationError):
                print("❌ API 认证失败。请检查您的 API 密钥是否正确。")
            elif isinstance(e, RateLimitError):
                print("⚠️ API 请求频率超限。请稍后重试。")
            raise
        except Exception as e:
            # 其他错误（如网络错误）可以尝试回退到普通请求
            print(f"流式请求出错: {e}")
            print("回退到普通请求模式...")
            try:
                return self.request(messages)
            except Exception as fallback_error:
                print(f"普通请求也失败: {fallback_error}")
                raise


    def embedding(self,question):
        try:
            embeddings = self.client.embeddings.create(
              model="text-embedding-ada-002", # 模型倍率 5.00
              input=question
            )

            return embeddings
        except AuthenticationError as e:
            error_msg = str(e)
            if "quota exhausted" in error_msg.lower() or "quota" in error_msg.lower():
                print("❌ API 密钥配额已耗尽。请检查您的 API 密钥配额或更换新的 API 密钥。")
            else:
                print("❌ API 认证失败。请检查您的 API 密钥是否正确。")
            raise
        except RateLimitError as e:
            print("⚠️ API 请求频率超限。请稍后重试。")
            raise
        except APIError as e:
            print(f"❌ API 请求失败: {str(e)}")
            raise


if __name__ == '__main__':
    lm = LLM(request_model="gemini-1.5-pro-exp-0827")
    # for model in lm.client.models.list().data:
    #     print(model.id)
    # answer = llm.embedding(question="who are you,gpt?")
    # print(answe r)

    # test request
    # messages = [{"role": "system", "content": ""}]
    # messages.append({"role": "user", "content": "你是谁？"})
    # res_msg = lm.request(messages)
    # messages.append({"role": "assistant", "content": res_msg})
    # print(res_msg)

    # test stream_request
    messages = [{"role": "system", "content": ""}]
    messages.append({"role": "user", "content": "你是基于gpt多少？你是 GPT-4o 的模型吗？"})
    try:
        res_msg = lm.stream_request(messages)
        if res_msg:
            messages.append({"role": "assistant", "content": res_msg})
        print(res_msg)
    except (AuthenticationError, RateLimitError, APIError) as e:
        print(f"请求失败: {e}")