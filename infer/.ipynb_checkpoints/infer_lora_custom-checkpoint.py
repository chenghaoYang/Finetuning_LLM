# -*- coding: utf-8 -*-
"""
Steps
 1) Load the fine-tuned LoRA model from best_ckpt/last-v11
 2) Generate for the first <SAMPLE_SIZE> prompts of the test set
 3) Save all results to infer_results.json
 4) Compute BLEU / ROUGE-L / BERT-F1 with the updated evaluate()
"""
import os, sys, json, torch
from tqdm import tqdm
from evaluate import evaluate                     # <- your newer helper

# -------------------------------------------------------------------------
# Project imports – make sure repo root is on PYTHONPATH
# -------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

from data_utils import config_args, NN_DataHelper
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig, GenerationConfig
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer, PetlArguments
from data_processer import make_context

# ----------------------------- hyper-params --------------------------------
WEIGHT_ROOT  = "../scripts/best_ckpt"
CKPT_DIR     = os.path.join(WEIGHT_ROOT, "last-v10")
TEST_FILE    = "finetune_test_data.json"
SAMPLE_SIZE  = 500               # -- change here only
MAX_NEW_TOK  = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.float16 if torch.cuda.is_available() else torch.float32
# ---------------------------------------------------------------------------

# 1) build model & tokenizer ------------------------------------------------
config = AutoConfig.from_pretrained(WEIGHT_ROOT)
parser = HfArgumentParser((ModelArguments,))
(model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)

# 1) 载入基座 + LoRA
pl_model = MyTransformer(
    config      = config,
    model_args  = model_args,
    lora_args   = PetlArguments.from_pretrained(CKPT_DIR),
    torch_dtype = DTYPE,               # 半精度加载基座
)
pl_model.load_sft_weight(CKPT_DIR)      # 注入 LoRA

# 2) *立刻*统一精度 & device
pl_model = pl_model.to(device=DEVICE, dtype=DTYPE)   # <- 关键行
pl_model.eval()

tokenizer, *_ = NN_DataHelper(model_args).load_tokenizer_and_config()

pl_model.get_llm_model().generation_config = GenerationConfig(
    eos_token_id          = tokenizer.eos_token_id,
    pad_token_id          = tokenizer.eos_token_id,
    chat_format           = "chatml",
    max_new_tokens        = MAX_NEW_TOK,
    do_sample             = False,          # deterministic beam search
    num_beams             = 4,
    length_penalty        = 1.0,
    no_repeat_ngram_size  = 3,
    early_stopping        = True,
)

# 2) load test-prompts ------------------------------------------------------
with open(TEST_FILE, encoding="utf-8") as fr:
    test_data = json.load(fr)

eval_data, save_list = [], []

# 3) inference --------------------------------------------------------------
for sample in tqdm(test_data[:SAMPLE_SIZE], desc=f"Running inference (first {SAMPLE_SIZE})"):
    prompt, reference = sample["text"], sample["ref"]

    # build input
    _, input_ids = make_context(tokenizer, prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # generate
    with torch.no_grad():
        output_ids = pl_model.get_llm_model().generate(inputs=input_ids)
    gen_ids = output_ids[0, input_ids.shape[-1]:]

    # decode – keep full answer but trim to safeguard run-away generations
    prediction = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    prediction = prediction[:1000]    # optional hard cut

    eval_data.append({"text": prediction, "ref": reference})
    save_list.append({"prompt": prompt, "text": prediction, "ref": reference})

# 4) save & echo samples ----------------------------------------------------
with open("infer_results.json", "w", encoding="utf-8") as fw:
    json.dump(save_list, fw, ensure_ascii=False, indent=2)

print("✅ Results saved to infer_results.json\n")
print("=== Sample Outputs ===")
for item in save_list[:5]:
    print("Prompt    :", item["prompt"][:60] + "…" if len(item["prompt"]) > 60 else item["prompt"])
    print("Prediction:", item["text"])
    print("Reference :", item["ref"][0], "\n")

# 5) automatic evaluation ---------------------------------------------------
metrics = evaluate(eval_data)   # <- pass through
print("=== Automatic Metrics ===")
print(f"BLEU        : {metrics['bleu_score']:.2f}")
print(f"ROUGE-L (F) : {metrics['rouge-l_score']:.3f}")
print(f"BERT-F1     : {metrics['bert_f1']:.3f}")