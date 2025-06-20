import json
import time
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer
from openai import OpenAI

client = OpenAI(
    api_key="xxx",  
    base_url="https://api.deepseek.com",  
)

MODEL_NAME = "deepseek-chat"
TEST_JSON_PATH = "test.json"
OUTPUT_JSON_PATH = "deepseek_v3_outputs.json"

with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)


results = []
for sample in tqdm(test_data, desc="Generating"):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nAnswer:"}
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    
    generated = resp.choices[0].message.content.strip()
    results.append({
        "instruction": sample["instruction"],
        "input": sample["input"],
        "reference": sample["output"],
        "prediction": generated,
    })
    time.sleep(0.1)  

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

refs = [[r["reference"]] for r in results]
hyps = [r["prediction"] for r in results]
bleu = sacrebleu.corpus_bleu(hyps, refs, force=True)
print(f"BLEU-4: {bleu.score:.2f}")


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
agg = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
for ref, hyp in zip(refs, hyps):
    scores = scorer.score(ref[0], hyp)
    for k in agg:
        agg[k] += scores[k].fmeasure

n = len(results)
print("Average ROUGE F1:")
print(f"  ROUGE-1: {agg['rouge1']/n*100:.2f}")
print(f"  ROUGE-2: {agg['rouge2']/n*100:.2f}")
print(f"  ROUGE-L: {agg['rougeL']/n*100:.2f}")