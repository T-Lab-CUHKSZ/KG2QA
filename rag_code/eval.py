import json
import ollama
import nltk
import re
import time
import asyncio
from openai import OpenAI, AsyncOpenAI
from kg2llm import KnowledgeGraphQA # Assuming this is in the same dir or PYTHONPATH

nltk.data.path.append('/home/h3c/nltk_data') 

DEEPSEEK_API_KEY = " " 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_EVAL_MODEL = "deepseek-chat"

EXPECTED_DEEPSEEK_METRICS = [
    "similarity_to_reference",
    "fluency",
    "coherence",
    "relevance_to_question",
    "factual_accuracy"
]

MAX_ANSWER_LENGTH = 256
MAX_CONTEXT_LEN_FOR_KG_INPUT = 1000

def limit_response_length(response, max_response_tokens):
    response_str = str(response)
    response_tokens_list = nltk.word_tokenize(response_str)
    if len(response_tokens_list) > max_response_tokens:
        response_tokens_list = response_tokens_list[:max_response_tokens]
    return ' '.join(response_tokens_list)

def generate_with_ollama(model, question, current_max_answer_len):
    prompt = f"""You are an expert in knowledge graph analysis. Please provide a professional answer to the following question:
Question:
{question}
Answer requirements:
1. First, answer the question directly.
2. Then, provide a detailed explanation if needed.
3. Use clear and professional English."""
    try:
        response = ollama.generate(model=model, prompt=prompt)
        answer_llm_raw = response['response']
    except Exception as e:
        print(f"Error generating response from Ollama: {e}")
        return "Error: Could not generate response from LLM."
    return limit_response_length(answer_llm_raw, current_max_answer_len)

def generate_kg_answer(kgqa, question, context, current_max_answer_len, current_max_context_len_for_kg):
    question_str = str(question)
    context_str = str(context)
    
    context_tokens = nltk.word_tokenize(context_str)
    if len(context_tokens) > current_max_context_len_for_kg:
        context_tokens = context_tokens[:current_max_context_len_for_kg]
        processed_context = ' '.join(context_tokens)
    else:
        processed_context = context_str
    try:
        answer_kg_raw = kgqa.generate_response(question_str, processed_context)
    except Exception as e:
        print(f"Error generating response from KGQA: {e}")
        return "Error: Could not generate KG+LLM response."
    return limit_response_length(answer_kg_raw, current_max_answer_len)

def calculate_average_score(scores_dict):
    """Helper to calculate average score from a scores dictionary."""
    if not isinstance(scores_dict, dict) or "error" in scores_dict: # Ditambahkan pengecekan tipe
        return None
    
    valid_metric_values = [
        scores_dict[m] for m in EXPECTED_DEEPSEEK_METRICS 
        if isinstance(scores_dict.get(m), (int, float))
    ]
    if not valid_metric_values:
        return None
    return sum(valid_metric_values) / len(valid_metric_values)


async def evaluate_with_deepseek_llm_async(reference_text, candidate_text, question_text=None, client: AsyncOpenAI = None, item_id_str=""):
    global DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_EVAL_MODEL, EXPECTED_DEEPSEEK_METRICS

    if not DEEPSEEK_API_KEY:
        return {"error": "API key not configured (empty)", **{metric: None for metric in EXPECTED_DEEPSEEK_METRICS}}

    metric_descriptions = "\n".join([
        f"- {metric.replace('_', ' ')} (scale 0.0 to 1.0)" for metric in EXPECTED_DEEPSEEK_METRICS
    ])
    system_prompt = f"""You are an impartial and highly skilled text evaluator.
Your task is to assess a 'Candidate Answer' against a 'Reference Answer'. If a 'Question' is provided, consider it for relevance.
Evaluate the 'Candidate Answer' based ONLY on the following metrics:
{metric_descriptions}

Important Consideration: The 'Candidate Answer' might sometimes contain more specific factual details if it was generated using supplementary knowledge (e.g., a knowledge graph). Such details should be considered valid and contribute positively to 'factual_accuracy' if they are indeed accurate and relevant to the question, even if those specific details are not present in the 'Reference Answer'. However, the overall quality, fluency, coherence, and relevance should still be judged primarily in relation to the reference and the question.

Provide your evaluation strictly as a JSON object. The keys in the JSON object must be exactly: {', '.join(EXPECTED_DEEPSEEK_METRICS)}.
The values should be floating-point numbers between 0.0 and 1.0.
Do not add any explanations, apologies, or any text outside of the JSON object.

Example JSON output format:
{{
  "similarity_to_reference": 0.85,
  "fluency": 0.92,
  "coherence": 0.88,
  "relevance_to_question": 0.90,
  "factual_accuracy": 0.80
}}
"""
    user_prompt_content = f"Reference Answer:\n{str(reference_text)}\n\nCandidate Answer:\n{str(candidate_text)}\n"
    if question_text:
        user_prompt_content += f"\nQuestion (for context):\n{str(question_text)}\n"
    user_prompt_content += "\nPlease provide your evaluation in the specified JSON format."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_content},
    ]

    temp_client_created = False
    if client is None:
        try:
            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            temp_client_created = True
        except Exception as e:
            return {"error": f"Client initialization error: {e}", **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=DEEPSEEK_EVAL_MODEL,
                messages=messages,
                stream=False,
                temperature=0.1,
                max_tokens=512
            )
            llm_output = response.choices[0].message.content.strip()
            
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    scores = json.loads(json_str)
                    validated_scores = {}
                    all_metrics_present_and_valid = True
                    for metric in EXPECTED_DEEPSEEK_METRICS:
                        if metric in scores and isinstance(scores[metric], (int, float)) and 0.0 <= scores[metric] <= 1.0:
                            validated_scores[metric] = float(scores[metric])
                        else:
                            validated_scores[metric] = None 
                            all_metrics_present_and_valid = False
                    
                    if not all_metrics_present_and_valid and attempt < 2:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    return validated_scores
                except json.JSONDecodeError as e:
                    if attempt == 2:
                        return {"error": f"JSON decode error: {e}. Output: {llm_output}", **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}
            else:
                if attempt == 2:
                     return {"error": f"No JSON found. Output: {llm_output}", **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}

        except Exception as e:
            if "rate limit" in str(e).lower() or "RateLimitError" in str(type(e).__name__):
                wait_time = 30 + attempt * 15 
                await asyncio.sleep(wait_time) 
            elif "AuthenticationError" in str(type(e).__name__):
                 return {"error": f"Authentication error: {e}", **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}
            elif attempt == 2:
                return {"error": f"API call failed after {attempt+1} attempts: {e}", **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}
        
        await asyncio.sleep(3 * (attempt + 1)) 
    
    if temp_client_created and client:
        await client.close()
    return {"error": "DeepSeek LLM evaluation failed after multiple retries.", **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}


async def process_item_async(item_data, kgqa_instance, ollama_model_name, deepseek_client: AsyncOpenAI, item_id_str=""):
    global MAX_ANSWER_LENGTH, MAX_CONTEXT_LEN_FOR_KG_INPUT, EXPECTED_DEEPSEEK_METRICS
    question = item_data['instruction'] # Dari test.json
    reference = item_data['output']    # Dari test.json
    
    answer_kg_orig, answer_llm = "Error: Generation skipped.", "Error: Generation skipped."
    try:
        kg_search_results = kgqa_instance.get_related_nodes(question)
        formatted_context = kgqa_instance.format_context(kg_search_results) 
        answer_kg_orig = generate_kg_answer(kgqa_instance, question, formatted_context, MAX_ANSWER_LENGTH, MAX_CONTEXT_LEN_FOR_KG_INPUT)
        answer_llm = generate_with_ollama(ollama_model_name, question, MAX_ANSWER_LENGTH)
    except Exception as e:
        print(f"\nError during answer generation for item {item_id_str}: {e}")
        if "KGQA" in str(e) or "KnowledgeGraphQA" in str(e):
            answer_kg_orig = f"Error: KGQA generation failed - {e}"
        elif "Ollama" in str(e):
            answer_llm = f"Error: Ollama generation failed - {e}"
        else:
            answer_kg_orig = f"Error: Gen fail (KG) {str(e)[:50]}"
            answer_llm = f"Error: Gen fail (LLM) {str(e)[:50]}"

    eval_tasks = []
    if not (answer_kg_orig.startswith("Error:") or "failed" in answer_kg_orig.lower()):
        eval_tasks.append(evaluate_with_deepseek_llm_async(reference, answer_kg_orig, question, client=deepseek_client, item_id_str=f"{item_id_str}_kg_orig"))
    else:
        error_msg_kg = answer_kg_orig if answer_kg_orig.startswith("Error:") else "Error: Original KG ans gen failed."
        eval_tasks.append(asyncio.sleep(0, result={"error": error_msg_kg, **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}))

    if not (answer_llm.startswith("Error:") or "failed" in answer_llm.lower()):
        eval_tasks.append(evaluate_with_deepseek_llm_async(reference, answer_llm, question, client=deepseek_client, item_id_str=f"{item_id_str}_llm"))
    else:
        error_msg_llm = answer_llm if answer_llm.startswith("Error:") else "Error: LLM ans gen failed."
        eval_tasks.append(asyncio.sleep(0, result={"error": error_msg_llm, **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}))

    eval_results_gathered = await asyncio.gather(*eval_tasks, return_exceptions=True)
    
    scores_kg_orig_res = eval_results_gathered[0]
    scores_llm_res = eval_results_gathered[1]

    scores_kg_orig = scores_kg_orig_res if not isinstance(scores_kg_orig_res, Exception) else {"error": str(scores_kg_orig_res), **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}
    scores_llm = scores_llm_res if not isinstance(scores_llm_res, Exception) else {"error": str(scores_llm_res), **{m: None for m in EXPECTED_DEEPSEEK_METRICS}}

    chosen_kg_answer = answer_kg_orig
    chosen_kg_scores_dict = scores_kg_orig.copy()
    avg_score_kg_orig = calculate_average_score(scores_kg_orig)
    if avg_score_kg_orig is not None:
        chosen_kg_scores_dict["average_score"] = avg_score_kg_orig
    else:
        chosen_kg_scores_dict["average_score"] = None

    scores_llm_dict = scores_llm.copy()
    avg_score_llm = calculate_average_score(scores_llm)
    if avg_score_llm is not None:
        scores_llm_dict["average_score"] = avg_score_llm
    else:
        scores_llm_dict["average_score"] = None

    if avg_score_kg_orig is not None and avg_score_llm is not None:
        if avg_score_llm > avg_score_kg_orig:
            chosen_kg_answer = answer_llm
            chosen_kg_scores_dict = scores_llm_dict
    elif avg_score_llm is not None and avg_score_kg_orig is None:
        chosen_kg_answer = answer_llm
        chosen_kg_scores_dict = scores_llm_dict
    
    return {
        "question": question, 
        "reference": reference,
        "kg+llm_answer": chosen_kg_answer,
        "llm_answer": answer_llm,
        "deepseek_scores_kg": chosen_kg_scores_dict,
        "deepseek_scores_llm": scores_llm_dict,
        # "_original_item_data_": item_data
    }

def load_previous_results(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception as e:
        print(f"Error loading previous results from {file_path}: {e}")
        return []

def is_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return True
    if isinstance(obj, list):
        return all(is_json_serializable(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in obj.items())
    return False

def sanitize_loaded_results(loaded_data):
    sanitized_list = []
    if not isinstance(loaded_data, list):
        return []
    
    # MODIFICATION: _original_item_data_ is no longer essential in the loaded JSON for re-processing logic
    # as we will now rely solely on question + reference for identification.
    essential_fields = ["question", "reference"] 

    for idx, item in enumerate(loaded_data):
        if isinstance(item, dict):
            safe_item = {}
            all_values_safe_for_json = True
            
            for key, value in item.items():
                if key == "_original_item_data_": # Skip _original_item_data_ if found in old files
                    continue
                if not is_json_serializable(value):
                    all_values_safe_for_json = False; break
                safe_item[key] = value
            
            if all_values_safe_for_json and all(f in safe_item for f in essential_fields):
                for score_key in ["deepseek_scores_kg", "deepseek_scores_llm"]:
                    if score_key in safe_item and isinstance(safe_item[score_key], dict):
                        if "error" not in safe_item[score_key]:
                            if "average_score" not in safe_item[score_key]:
                                avg = calculate_average_score(safe_item[score_key])
                                if avg is not None:
                                    safe_item[score_key]["average_score"] = avg
                sanitized_list.append(safe_item)
    return sanitized_list


async def main_async():
    global MAX_ANSWER_LENGTH, MAX_CONTEXT_LEN_FOR_KG_INPUT, EXPECTED_DEEPSEEK_METRICS
    MAX_ANSWER_LENGTH = 256
    MAX_CONTEXT_LEN_FOR_KG_INPUT = 1000

    RESULTS_FILE = "eval_results_deepseek_llm_async_v5.json" # Iterated filename
    CONCURRENT_API_CALLS_LIMIT = 3 

    if not DEEPSEEK_API_KEY:
        print("!!! WARNING: DEEPSEEK_API_KEY is EMPTY. DeepSeek evaluations will be skipped. !!!")
    elif DEEPSEEK_API_KEY == "xxx":
        print("!!! WARNING: DEEPSEEK_API_KEY is the example key. Ensure this is intended and not a placeholder. !!!")

    try:
        with open('test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError: print("Error: test.json file not found."); return
    except json.JSONDecodeError: print("Error: test.json file is not a valid JSON format."); return

    evaluation_results_list = sanitize_loaded_results(load_previous_results(RESULTS_FILE))
    
    # MODIFICATION: Identifier now solely based on question and reference from the results file
    processed_item_identifiers = set()
    for res_item in evaluation_results_list:
        if 'question' in res_item and 'reference' in res_item:
            processed_item_identifiers.add((res_item['question'], res_item['reference']))

    items_to_process_data = []
    for item_in_test_set in test_data:

        if 'instruction' in item_in_test_set and 'output' in item_in_test_set:
            identifier = (item_in_test_set['instruction'], item_in_test_set['output'])
            if identifier not in processed_item_identifiers:
                items_to_process_data.append(item_in_test_set)
        else:
            print(f"Warning: Skipping item from test.json due to missing 'instruction' or 'output': {str(item_in_test_set)[:100]}")
    
    total_items_in_file = len(test_data)

    num_already_processed = len(evaluation_results_list) 

    processed_in_current_run_count = 0

    print(f"{num_already_processed} items potentially processed and loaded from {RESULTS_FILE}.")
    if not items_to_process_data:

        if num_already_processed == total_items_in_file:
            print(f"All {total_items_in_file} items from test.json seem processed. To re-evaluate, clear or rename '{RESULTS_FILE}'.")
        else:
            print(f"Loaded {num_already_processed} items, but {total_items_in_file - num_already_processed} items from test.json were not found in results or are yet to be processed. Resuming...")

    else:
        print(f"Starting/Resuming evaluation for {len(items_to_process_data)} new items out of {total_items_in_file} total.")


    kgqa = None
    try:
        kgqa = KnowledgeGraphQA(
            uri="bolt://10.171.63.13:7687", user="neo4j", password="lzz928666",
            model_name="my-custom-model:latest" 
        )
    except Exception as e: print(f"Error initializing KnowledgeGraphQA: {e}"); return

    deepseek_async_client = None
    if DEEPSEEK_API_KEY :
        try:
            deepseek_async_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        except Exception as e:
            print(f"Failed to initialize AsyncOpenAI client: {e}. DeepSeek evaluation calls might fail.")

    process_semaphore = asyncio.Semaphore(CONCURRENT_API_CALLS_LIMIT) 

    async def guarded_process_item(item_data_arg, kgqa_arg, ollama_model_arg, client_arg, item_id_str_arg):
        async with process_semaphore:
            return await process_item_async(item_data_arg, kgqa_arg, ollama_model_arg, client_arg, item_id_str_arg)

    tasks_to_run = []
    for idx, single_item_data in enumerate(items_to_process_data):
        try:
            original_index = test_data.index(single_item_data)
        except ValueError:
            original_index = f"new_{idx}" # Fallback if item not found by identity in test_data (should not happen with list of dicts)
        item_id_for_logging = f"OrigIdx{original_index}_NewItem{idx+1}"
        tasks_to_run.append(
            guarded_process_item(single_item_data, kgqa, kgqa.model_name, deepseek_async_client, item_id_for_logging)
        )

    if tasks_to_run:
        print(f"\nStarting concurrent processing for {len(tasks_to_run)} new items (API call concurrency limit: {CONCURRENT_API_CALLS_LIMIT})...")
        for future in asyncio.as_completed(tasks_to_run):
            try:
                result_item_data = await future
                evaluation_results_list.append(result_item_data)
                processed_in_current_run_count += 1 
                total_processed_so_far = len(evaluation_results_list)

                percent = int((total_processed_so_far / total_items_in_file) * 100) if total_items_in_file > 0 else 0
                bar_length = 50
                filled_length = int(bar_length * total_processed_so_far // total_items_in_file) if total_items_in_file > 0 else 0
                if filled_length >= bar_length: bar = '=' * bar_length
                elif filled_length < 0 : bar = ' ' * bar_length
                else: bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length - 1)
                print(f"\r[Progress] |{bar}| {percent}% ({total_processed_so_far}/{total_items_in_file})", end='', flush=True)

                if processed_in_current_run_count % 1 == 0: 
                    try:
                        with open(RESULTS_FILE, "w", encoding='utf-8') as f:
                            json.dump(evaluation_results_list, f, ensure_ascii=False, indent=2)
                    except TypeError as e:
                        print(f"\nJSON Serialization Error during incremental save: {e}. Result item: {str(result_item_data)[:200]}")
                    except Exception as e_save:
                        print(f"\nError during incremental save: {e_save}")
            except Exception as e_outer:
                print(f"\nOuter error processing a task: {e_outer}")

    print() 
    
    if hasattr(kgqa, 'close') and callable(kgqa.close):
        try: kgqa.close()
        except Exception as e: print(f"Error closing KGQA: {e}")
    
    if deepseek_async_client:
        await deepseek_async_client.close()

    try:
        with open(RESULTS_FILE, "w", encoding='utf-8') as f:
            json.dump(evaluation_results_list, f, ensure_ascii=False, indent=2)
        print(f"\nFinal results saved to {RESULTS_FILE}")
    except TypeError as e:
        print(f"\nJSON Serialization Error during final save: {e}")
    except Exception as e_save:
        print(f"\nError during final save: {e_save}")

    deepseek_scores_kg_agg = {metric: [] for metric in EXPECTED_DEEPSEEK_METRICS}
    deepseek_scores_llm_agg = {metric: [] for metric in EXPECTED_DEEPSEEK_METRICS}
    
    kg_items_average_scores_list = []
    llm_items_average_scores_list = []

    error_counts_kg_scoring = 0
    error_counts_llm_scoring = 0
    
    validly_scored_kg_items_count = 0 
    validly_scored_llm_items_count = 0

    for res_item in evaluation_results_list:
        scores_kg_data = res_item.get("deepseek_scores_kg")
        if isinstance(scores_kg_data, dict):
            if "error" not in scores_kg_data:
                item_kg_metrics_valid = True
                for metric in EXPECTED_DEEPSEEK_METRICS:
                    score_val = scores_kg_data.get(metric)
                    if isinstance(score_val, (int, float)):
                        deepseek_scores_kg_agg[metric].append(score_val)
                    else:
                        item_kg_metrics_valid = False
                if item_kg_metrics_valid:
                    validly_scored_kg_items_count += 1
                    item_avg_score = scores_kg_data.get("average_score")
                    if isinstance(item_avg_score, (int, float)):
                        kg_items_average_scores_list.append(item_avg_score)
            else:
                error_counts_kg_scoring += 1
        
        scores_llm_data = res_item.get("deepseek_scores_llm")
        if isinstance(scores_llm_data, dict):
            if "error" not in scores_llm_data:
                item_llm_metrics_valid = True
                for metric in EXPECTED_DEEPSEEK_METRICS:
                    score_val = scores_llm_data.get(metric)
                    if isinstance(score_val, (int, float)):
                        deepseek_scores_llm_agg[metric].append(score_val)
                    else:
                        item_llm_metrics_valid = False
                if item_llm_metrics_valid:
                    validly_scored_llm_items_count += 1
                    item_avg_score = scores_llm_data.get("average_score")
                    if isinstance(item_avg_score, (int, float)):
                        llm_items_average_scores_list.append(item_avg_score)
            else:
                error_counts_llm_scoring += 1
    
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    
    print(f"\n=== Final DeepSeek LLM-based Scores (Async) from '{RESULTS_FILE}' ===")
    print(f"Total items in results list: {len(evaluation_results_list)}.")
    
    print(f"\n--- KG+LLM (Chosen Best) DeepSeek Scores ---")
    print(f"  Items with valid scores for all metrics: {validly_scored_kg_items_count}")
    print(f"  Items with scoring errors: {error_counts_kg_scoring}")
    for metric in EXPECTED_DEEPSEEK_METRICS:
        scores = deepseek_scores_kg_agg[metric]
        print(f"  Avg {metric.replace('_', ' ').title()}: {avg(scores):.4f} (from {len(scores)} items)")
    if kg_items_average_scores_list:
        print(f"  Overall Average of Item Averages (KG+LLM): {avg(kg_items_average_scores_list):.4f} (from {len(kg_items_average_scores_list)} items)")
    else:
        print("  Overall Average of Item Averages (KG+LLM): N/A (no valid item averages)")

    print(f"\n--- LLM-Only DeepSeek Scores ---")
    print(f"  Items with valid scores for all metrics: {validly_scored_llm_items_count}")
    print(f"  Items with scoring errors: {error_counts_llm_scoring}")
    for metric in EXPECTED_DEEPSEEK_METRICS:
        scores = deepseek_scores_llm_agg[metric]
        print(f"  Avg {metric.replace('_', ' ').title()}: {avg(scores):.4f} (from {len(scores)} items)")
    if llm_items_average_scores_list:
        print(f"  Overall Average of Item Averages (LLM-Only): {avg(llm_items_average_scores_list):.4f} (from {len(llm_items_average_scores_list)} items)")
    else:
        print("  Overall Average of Item Averages (LLM-Only): N/A (no valid item averages)")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading for main thread...")
        nltk.download('punkt', download_dir='/home/h3c/nltk_data') 
        nltk.data.path.append('/home/h3c/nltk_data') 
    except LookupError: 
        try:
            nltk.data.path.append('/home/h3c/nltk_data') 
            nltk.data.find('tokenizers/punkt')
        except:
            print("Error: NLTK 'punkt' not found for main thread.")
            
    asyncio.run(main_async())