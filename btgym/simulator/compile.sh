#!/bin/bash
# cd simulator
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. simulator.proto 

