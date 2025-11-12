import grpc
from concurrent import futures
from transformers import LlamaForCausalLM, LlamaTokenizer
import btgym.llm.large_model_pb2 as large_model_pb2
import btgym.llm.large_model_pb2_grpc as large_model_pb2_grpc

class LargeModelServicer(large_model_pb2_grpc.LargeModelServiceServicer):
    def __init__(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path)

    def GenerateText(self, request, context):
        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=2048, temperature=0.5)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return large_model_pb2.TextResponse(text=text)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    large_model_pb2_grpc.add_LargeModelServiceServicer_to_server(LargeModelServicer('path/to/llama3'), server)
    port = 50054
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Large Model server started on port {port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()