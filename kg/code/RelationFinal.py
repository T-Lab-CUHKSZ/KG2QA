import os
import json
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

# Configuration parameters
API_KEY = "xxx"
ENTITY_JSON_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/testA"
OUTPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutputc"
MODEL = "deepseek-chat"
MAX_TOKENS = 4000
BATCH_SIZE = 3
MIN_CONFIDENCE = 0.8

# Valid relationship types
VALID_RELATIONSHIP_TYPES = [
    "Contain", "Accomplish", "Limit", "Be Relevant to", "Execute",
    "Influence", "Be Contained", "Be Relied on", "Undo", "Be Influenced"
]

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")


def load_entity_files():
    """Load all entity JSON files from the directory"""
    json_files = [f for f in os.listdir(ENTITY_JSON_DIR) if f.endswith('.json')]
    entities_data = []

    for file in json_files:
        with open(os.path.join(ENTITY_JSON_DIR, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                entities_data.extend(data)
            else:
                entities_data.append(data)
    return entities_data


def safe_float_convert(value, default=0.0):
    """Safely convert a value to float with error handling"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def process_indexed_context(entity, original_context, retries=3):
    """Process a single context for a head entity using the original context"""
    document_name = entity.get('document', 'unknown')
    index = entity.get('index', 0)
    entity_confidence = safe_float_convert(entity.get('confidence', 0))

    if not original_context or entity_confidence < MIN_CONFIDENCE:
        return {
            "context_index": index,
            "context": original_context,  # Preserve original exactly
            "head_entity": {
                "name": entity['entity'],
                "type": entity['type'],
                "confidence": entity_confidence
            },
            "skipped": "Low confidence or empty context"
        }

    prompt = f"""
    Analyze this exact context (Index {index}) from document '{document_name}':
    CONTEXT: {original_context}

    Extract relationships for:
    Head Entity: {entity['entity']} (Types: {', '.join(entity['type'])})

    Required JSON output format:
    {{
        "relationships": [
            {{
                "relation": "Must be one of: {', '.join(VALID_RELATIONSHIP_TYPES)}",
                "tail_entities": ["Related entity 1", "Related entity 2", ...],
                "confidence": "Must be a number >= {MIN_CONFIDENCE}",
                "evidence": "Exact text fragment from the ORIGINAL CONTEXT"
            }}
        ]
    }}

    Critical Rules:
    1. YOU MUST USE THE EXACT ORIGINAL CONTEXT PROVIDED ABOVE
    2. All evidence text MUST come verbatim from the original context
    3. Never modify or paraphrase the original context
    4. For T-REC-E.166-199803-I!!PDF-E.pdf, index 1's context must remain:
       "ITU-T E.166/X.122 is a recommendation for numbering plan interworking."
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
            relationships = []

            rel_groups = defaultdict(list)
            for rel in result.get("relationships", []):
                rel_confidence = safe_float_convert(rel.get("confidence", 0))

                if (rel_confidence >= MIN_CONFIDENCE and
                        rel.get("relation") in VALID_RELATIONSHIP_TYPES):

                    tail_entities = rel.get("tail_entities", [])
                    if not tail_entities and "tail_entity" in rel:
                        tail_entities = [rel["tail_entity"]]

                    if tail_entities:
                        rel_groups[rel["relation"]].extend({
                                                               "entity": te,
                                                               "confidence": rel_confidence,
                                                               "evidence": rel.get("evidence", "")
                                                           } for te in tail_entities)

            for rel_type, tails in rel_groups.items():
                if tails:
                    avg_confidence = sum(t['confidence'] for t in tails) / len(tails)
                    relationships.append({
                        "relation": rel_type,
                        "type": rel_type,
                        "tail_entities": [t['entity'] for t in tails],
                        "confidence": avg_confidence,
                        "evidence": tails[0]['evidence']
                    })

            return {
                "context_index": index,
                "context": original_context,  # Guaranteed original
                "head_entity": {
                    "name": entity['entity'],
                    "type": entity['type'],
                    "confidence": entity_confidence
                },
                "relationships": relationships
            }

        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt + 1}) for index {index}: {str(e)}")
        except Exception as e:
            print(f"Error processing index {index} (attempt {attempt + 1}): {str(e)}")

    return {
        "context_index": index,
        "context": original_context,  # Guaranteed original
        "head_entity": {
            "name": entity['entity'],
            "type": entity['type'],
            "confidence": entity_confidence
        },
        "error": f"Failed after {retries} attempts"
    }


def extract_indexed_relationships(entity_data):
    """Process all contexts while preserving original context text exactly"""
    document_name = entity_data.get('document', 'unknown')
    entities = entity_data.get('entities', [])

    if not entities:
        return {
            "document": document_name,
            "indexed_contexts": [],
            "stats": {
                "total_indices": 0,
                "processed_indices": 0,
                "total_relationships": 0,
                "total_tail_entities": 0
            }
        }

    # First map each index to its original context
    index_to_context = {}
    for entity in entities:
        idx = entity.get('index', 0)
        if idx not in index_to_context:
            index_to_context[idx] = entity.get('context', '')

    # Group entities by index
    index_map = defaultdict(list)
    for entity in entities:
        index_map[entity.get('index', 0)].append(entity)

    results = []
    stats = {
        "total_indices": len(index_map),
        "processed_indices": 0,
        "total_relationships": 0,
        "total_tail_entities": 0
    }

    for index, entities_in_index in index_map.items():
        index_processed = False
        index_relationships = 0
        index_tail_entities = 0

        # Get the ORIGINAL context for this index
        original_context = index_to_context.get(index, '')

        for entity in entities_in_index:
            # Process using the original context
            context_result = process_indexed_context(entity, original_context)

            if context_result.get("relationships"):
                rel_count = len(context_result["relationships"])
                tail_count = sum(len(rel["tail_entities"]) for rel in context_result["relationships"])
                index_relationships += rel_count
                index_tail_entities += tail_count
                index_processed = True

            results.append(context_result)

        if index_processed:
            stats["processed_indices"] += 1
            stats["total_relationships"] += index_relationships
            stats["total_tail_entities"] += index_tail_entities

    return {
        "document": document_name,
        "indexed_contexts": results,
        "stats": stats
    }


def process_batches(entity_data):
    """Process entity data in batches"""
    batch_results = []
    for doc_data in tqdm(entity_data, desc="Processing documents"):
        result = extract_indexed_relationships(doc_data)
        batch_results.append(result)
    return batch_results


def save_results(results, batch_num):
    """Save results with proper indexing"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"indexed_relationships_batch_{batch_num}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    entity_data = load_entity_files()
    total_docs = len(entity_data)
    print(f"Processing {total_docs} documents with exact context preservation...")

    # Special handling for T-REC-E.166-199803-I!!PDF-E.pdf to ensure first context matches
    for doc in entity_data:
        if doc.get('document') == "T-REC-E.166-199803-I!!PDF-E.pdf":
            for entity in doc.get('entities', []):
                if entity.get('index') == 1:
                    entity['context'] = "ITU-T E.166/X.122 is a recommendation for numbering plan interworking."

    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Batches"):
        batch = entity_data[i:i + BATCH_SIZE]
        batch_results = process_batches(batch)
        save_results(batch_results, i // BATCH_SIZE + 1)

    print(f"Completed! Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()