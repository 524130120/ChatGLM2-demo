from transformers import AutoConfig, AutoModel, AutoTokenizer
import os, torch

CHECKPOINT_PATH = "D:\\ChatGLM\\ChatGLM2-6B-main\\ptuning\\output\\adgen-chatglm2-6b-int4-pt-128-2e-2\\checkpoint-600"

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("d:\\ChatGLM\\ChatGLM2-6B-main\\chatglm2-6b-int4", trust_remote_code=True)

config = AutoConfig.from_pretrained("d:\\ChatGLM\\ChatGLM2-6B-main\\chatglm2-6b-int4", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("d:\\ChatGLM\\ChatGLM2-6B-main\\chatglm2-6b-int4", config=config, trust_remote_code=True)
print('-------------------BASE LOAD SUCCESS------------------')
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
print('-------------------PTUNING LOAD SUCCESS------------------')

model = model.cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好, 请问你是谁？", history=[])
print("你好, 请问你是谁？")
print(response)

response, history = model.chat(tokenizer, "喵01是谁呢？", history=[])
print(response)

response, history = model.chat(tokenizer, "你会叫吗？", history=[])
print(response)
