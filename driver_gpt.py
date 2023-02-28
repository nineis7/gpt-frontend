import tvm.relay
import torch
import numpy as np
import os

# Import required libraries
import torch
from transformers import AutoTokenizer, OpenAIGPTModel

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt", return_dict=False)
model.eval() # Set the model in evaluation mode to deactivate the DropOut modules
print(model)

# 生成输入tensor，负责输入进trace中flow一遍得到trace后的计算图
# Encode a text inputs
text = "What is the fastest car in the"
indexed_tokens = tokenizer.encode(text)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor], strict=False)
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
print(traced_model)

import tvm.relay
# parse pytorch model to tvm relay ir
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
#print(mod)

from tvm.relay.build_module import BuildModule
opt_level = 3
target = "llvm"
with tvm.transform.PassContext(opt_level=opt_level):
    module = BuildModule()
    # optimize() is where we will do operator fusion and quatization
    mod, params = module.optimize(mod, target=target, params=params)
    print(mod)

# add some changes