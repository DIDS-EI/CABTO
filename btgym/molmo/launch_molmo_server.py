import grpc
from concurrent import futures
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, Type

import btgym.molmo.molmo_pb2 as molmo_pb2
import btgym.molmo.molmo_pb2_grpc as molmo_pb2_grpc
from btgym.molmo.molmo import MolmoModel
import numpy as np
import time

class RPCMethod:
    """RPC方法注册器"""
    registry = {}

    def __init__(self, response_type: Type):
        self.response_type = response_type

    def __call__(self, func):
        method_name = func.__name__
        RPCMethod.registry[method_name] = {
            'handler': func,
            'response_type': self.response_type
        }

        def servicer_method(servicer, request, context):
            try:
                servicer._track_connection(context)
                servicer.command_queue.put((method_name, request))
                result = servicer.result_queue.get(timeout=10)
                return self.response_type(**result) if result else self.response_type()
            except (Empty, Exception):
                return self.response_type()

        RPCMethod.registry[method_name]['servicer_method'] = servicer_method
        return func

class MolmoServicer(molmo_pb2_grpc.MolmoServiceServicer):
    def __init__(self, command_queue: mp.Queue, result_queue: mp.Queue):
        self.command_queue = command_queue
        self.result_queue = result_queue
        self._active_connections = set()
        
    def _track_connection(self, context):
        peer = context.peer()
        self._active_connections.add(peer)
        context.add_callback(lambda: self._handle_disconnect(peer))
        return peer
    
    def _handle_disconnect(self, peer):
        self._active_connections.remove(peer)
        self.command_queue.put(('client_disconnected', peer))

class MolmoCommandHandler:
    def __init__(self, molmo: MolmoModel):
        self.molmo = molmo

    def handle_command(self, command: str, request: Any) -> Dict:
        if command not in RPCMethod.registry:
            return {}
        
        try:
            handler = RPCMethod.registry[command]['handler']
            return handler(self, request) or {}
        except Exception as e:
            print(f"Error handling command {command}: {str(e)}")
            return {}

    @RPCMethod(molmo_pb2.PointQAResponse)
    def PointQA(self, request):
        return {'text': self.molmo.point_qa(request.query, request.image_path)}

# 动态添加方法到ServicerClass
for method_name, method_info in RPCMethod.registry.items():
    setattr(MolmoServicer, method_name, method_info['servicer_method'])

def main_process_loop(command_queue: mp.Queue, result_queue: mp.Queue):
    # 添加这些行来处理SSL问题
    import os
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    molmo = MolmoModel()
    handler = MolmoCommandHandler(molmo)
    
    while True:
        try:
            if command_queue.empty():
                time.sleep(0.01)
                continue
                
            command, request = command_queue.get()
            if command == 'client_disconnected':
                print(f"客户端断开连接: {request}")
                continue
                
            result = handler.handle_command(command, request)
            result_queue.put(result)
        except Exception as e:
            print(f"Error handling command {command}: {str(e)}")

def serve():
    command_queue = mp.Queue()
    result_queue = mp.Queue()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = MolmoServicer(command_queue, result_queue)
    molmo_pb2_grpc.add_MolmoServiceServicer_to_server(servicer, server)
    
    port = 50053
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"Molmo server started on port {port}.")
    try:
        main_process_loop(command_queue, result_queue)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    serve()