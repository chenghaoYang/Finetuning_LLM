# infer_and_eval.py
import json
from infer_model import load_my_model    # 你自己的包装函数：返回 (model, tokenizer)
from evaluate import evaluate            # 刚才那段评测代码

# ① 加载训练好的 ckpt
model, tokenizer = load_my_model("best_ckpt/last-v1")

# ② 定义要测试的 prompt 以及每个 prompt 对应的参考答案
prompts_and_refs = {
    "写一个诗歌，关于冬天":   ["冬夜寂静冷…把快乐和温暖带回家。"],
    "晚上睡不着应该怎么办": ["保持规律作息…寻求睡眠专家帮助。"],
}

# ③ 逐条推理
records = []
for prompt, ref_list in prompts_and_refs.items():
    gen, _ = model.chat(tokenizer, prompt, history=[])
    records.append({"text": gen.strip(), "ref": ref_list})
    print(f"[Prompt] {prompt}\n[Gen] {gen}\n")

# ④ 直接评测
print("\n===  automatic metrics ===")
print(evaluate(records))
