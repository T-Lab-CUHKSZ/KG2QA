import os
import json
from tqdm import tqdm
from openai import OpenAI

# Configuration parameters
API_KEY = "xxx"
INPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutputc"
OUTPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutputd"
MODEL = "deepseek-chat"
MAX_TOKENS = 4000
BATCH_SIZE = 3
MIN_CONFIDENCE = 0.7

# Valid tail entity types
VALID_TAIL_ENTITY_TYPES = [
    "Identifier", "Structure/Composition", "Action",
    "Value", "Function", "Applicability/Context"
]

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")


def load_relationship_files():
    """Load all relationship JSON files from the directory"""
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    all_data = []

    for file in json_files:
        with open(os.path.join(INPUT_DIR, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    return all_data


def safe_float_convert(value, default=0.0):
    """Safely convert a value to float with error handling"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_filtered_tail_entities(context_data, retries=3):
    """Extract tail entities excluding any that match the head entity"""
    document_name = context_data.get('document', 'unknown')
    context_index = context_data.get('context_index', 0)
    original_context = context_data.get('context', '')
    head_entity_name = context_data.get('head_entity', {}).get('name', '')

    if not original_context:
        return {
            "tail_entities": [],
            "skipped": "Empty context"
        }

    prompt = f"""
    Analyze this exact context (Index {context_index}) from document '{document_name}':
    CONTEXT: {original_context}

    Extract all relevant tail entities and their types from this context, 
    EXCLUDING any entities that exactly match the head entity: "{head_entity_name}".

    Required JSON output format:
    {{
        "tail_entities": [
            {{
                "entity": "Exact entity name as it appears in the context",
                "type": "Must be one of: {', '.join(VALID_TAIL_ENTITY_TYPES)}",
                "confidence": "Estimated confidence score (0.0-1.0)",
                "evidence": "Exact text fragment from the ORIGINAL CONTEXT"
            }}
        ]
    }}

    Critical Rules:
    1. MUST EXCLUDE any entities that match "{head_entity_name}"
    2. YOU MUST USE THE EXACT ORIGINAL CONTEXT PROVIDED ABOVE
    3. All evidence text MUST come verbatim from the original context
    4. Never modify or paraphrase the original context
    5. For T-REC-E.166-199803-I!!PDF-E.pdf, index 1's context must remain:
       "ITU-T E.166/X.122 is a recommendation for numbering plan interworking."
    6. Only include entities with confidence >= {MIN_CONFIDENCE}
    """

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            filtered_tail_entities = []

            for entity in result.get("tail_entities", []):
                entity_confidence = safe_float_convert(entity.get("confidence", 0))

                # Additional filtering to ensure no head entity matches slip through
                if (entity_confidence >= MIN_CONFIDENCE and
                        entity.get("type") in VALID_TAIL_ENTITY_TYPES and
                        entity.get("entity") and
                        entity.get("evidence") and
                        entity["entity"].lower() != head_entity_name.lower()):

                    evidence_text = entity["evidence"]
                    if entity["entity"] in evidence_text:
                        filtered_tail_entities.append({
                            "entity": entity["entity"],
                            "type": entity["type"],
                            "confidence": entity_confidence,
                            "evidence": evidence_text
                        })

            return {
                "tail_entities": filtered_tail_entities
            }

        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt + 1}) for index {context_index}: {str(e)}")
        except Exception as e:
            print(f"Error processing index {context_index} (attempt {attempt + 1}): {str(e)}")

    return {
        "error": f"Failed after {retries} attempts"
    }


def process_document_with_filtered_tails(document_data):
    """Process a document to create output with filtered tail entities"""
    document_name = document_data.get('document', 'unknown')
    indexed_contexts = document_data.get('indexed_contexts', [])

    combined_results = []
    stats = {
        "total_contexts": len(indexed_contexts),
        "processed_contexts": 0,
        "total_tail_entities": 0
    }

    for context_data in indexed_contexts:
        combined_entry = {
            "context_index": context_data.get('context_index', 0),
            "context": context_data.get('context', ''),
            "head_entity": context_data.get('head_entity', {}),
            "relationships": context_data.get('relationships', [])
        }

        # Get filtered tail entities (excluding head entity matches)
        tail_result = extract_filtered_tail_entities({
            "document": document_name,
            "context_index": combined_entry["context_index"],
            "context": combined_entry["context"],
            "head_entity": combined_entry["head_entity"]
        })

        if tail_result.get("tail_entities"):
            combined_entry["tail_entities"] = tail_result["tail_entities"]
            stats["processed_contexts"] += 1
            stats["total_tail_entities"] += len(tail_result["tail_entities"])
        elif "error" in tail_result:
            combined_entry["error"] = tail_result["error"]

        combined_results.append(combined_entry)

    return {
        "document": document_name,
        "indexed_contexts": combined_results,
        "stats": stats
    }


def process_batches(all_data):
    """Process data in batches"""
    batch_results = []
    for doc_data in tqdm(all_data, desc="Processing documents"):
        result = process_document_with_filtered_tails(doc_data)
        batch_results.append(result)
    return batch_results


def save_results(results, batch_num):
    """Save results with proper indexing"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"filtered_tail_entities_batch_{batch_num}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    # Load the relationship data
    relationship_data = load_relationship_files()
    total_docs = len(relationship_data)
    print(f"Processing {total_docs} documents with filtered tail entities...")

    # Special handling for T-REC-E.166-199803-I!!PDF-E.pdf
    for doc in relationship_data:
        if doc.get('document') == "T-REC-E.166-199803-I!!PDF-E.pdf":
            for context in doc.get('indexed_contexts', []):
                if context.get('context_index') == 1:
                    context['context'] = "ITU-T E.166/X.122 is a recommendation for numbering plan interworking."

    # Process in batches
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Batches"):
        batch = relationship_data[i:i + BATCH_SIZE]
        batch_results = process_batches(batch)
        save_results(batch_results, i // BATCH_SIZE + 1)

    print(f"Completed! Filtered tail entity results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()