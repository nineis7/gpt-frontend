import tvm.relay
import torch
import numpy as np
import os
import time
from tvm.contrib import graph_executor

# Import required libraries
import torch
#print(torch.__version__)
from transformers import AutoTokenizer, OpenAIGPTModel

# ------------------------------- 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt", return_dict=False)
model.eval() # Set the model in evaluation mode to deactivate the DropOut modules
for p in model.parameters():
    p.requires_grad_(False)
# print(model)

# ------------------------------- 生成输入tensor，负责输入进trace中flow一遍得到trace后的计算图
# Encode a text inputs
text = "What is the fastest car in the"
indexed_tokens = tokenizer.encode(text)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# ------------------------ Creating the trace_model
traced_model = torch.jit.trace(model, [tokens_tensor], strict=False)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

# ------------------------------- GPU中计算预训练模型的inference耗时
model.cuda()
tt_c = tokens_tensor.cuda()
res_pt = model(tt_c)
torch.cuda.synchronize()
#print(res_pt) # tuple
#print(res_pt[0].shape) # torch.Size([1, 7, 768]) 

start_time = time.time()
for i in range(1000):
    model(tt_c)
torch.cuda.synchronize()

end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

# ------------------------------- 依据trace_model生成relay ir
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
#print(shape_list) # [('input_ids', [1, 7])]

import tvm.relay
# parse pytorch model to tvm relay ir
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
mod = tvm.relay.transform.InferType()(mod) #注释中输出type信息

#输出mod信息
import sys
sys.stdout = open('info32.txt', mode = 'w',encoding='utf-8')
print(mod)

# ------------------------------- 利用GPU进行Relay ir转换后的mod inference耗时
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target, params=params)
    
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input_ids", tt_c)
module.set_input(**params)
module.run()

start_time = time.time()
for i in range(1000):
    module.run()
torch.cuda.synchronize()

end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

# # ------------------------ 对relay ir进行优化，fp32->fp16
# mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
# BindPass = tvm.relay.transform.function_pass(
#     lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
#         fn, params
#     ),
#     opt_level=1,
# )

# mod = BindPass(mod)
# mod = tvm.relay.transform.FoldConstant()(mod)
# mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
# mod = tvm.relay.transform.FoldConstant()(mod)

# #mod = tvm.relay.transform.InferType()(mod) #注释中输出type信息
# mod = tvm.relay.transform.ToMixedPrecision()(mod)

# mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
# mod = tvm.relay.transform.FoldConstant()(mod)

# import sys
# sys.stdout = open('infor16.txt', mode = 'w',encoding='utf-8')
# print(mod)
# # from tvm.relay.build_module import BuildModule
# # opt_level = 3
# # target = "llvm"
# # with tvm.transform.PassContext(opt_level=opt_level):
# #     module = BuildModule()
# #     # optimize() is where we will do operator fusion and quatization
# #     mod, params = module.optimize(mod, target=target, params=params)
# #     #print(mod)

