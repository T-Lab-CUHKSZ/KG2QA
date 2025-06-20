import os
import json
from collections import defaultdict

INPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/testoutputf"
RELATION_JSON_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/testoutputh"
OUTPUT_FILE = "E:/Users/yeming/PycharmProjects/LLMProject/testoutputg/final_relationships.txt"

ENTITY_FILES = {
    "IDEN": "identifier_entities.txt",
    "STR_COM": "structure_composition_entities.txt",
    "APP_CON": "applicability_context_entities.txt",
    "ACT": "action_entities.txt",
    "VALUE": "value_entities.txt",
    "FUN": "function_entities.txt"
}

RELATION_MAPPING = {

    "contain": "contain",
    "isreliedon": "isReliedOn",
    "accomplish": "accomplish",
    "limit": "limit",
    "relevant": "relevant",
    "execute": "execute",
    "influence": "influence",
    "be contained": "isContained",
    "undo": "undo",
    "be influenced": "isInfluenced",

    "iscontained": "isContained",
    "isinfluenced": "isInfluenced",
    "rely on": "isReliedOn",
    "relied on": "isReliedOn",
    "is relied on": "isReliedOn",
    "relevance": "relevant",
    "related": "relevant"
}


def standardize_relation(relation):

    lower_rel = relation.lower().strip()

    if "rely" in lower_rel or "relied" in lower_rel:
        return "isReliedOn"
    if "relev" in lower_rel or "relat" in lower_rel:
        return "relevant"

    return RELATION_MAPPING.get(lower_rel, relation)


def load_entities():

    entities = defaultdict(dict)
    entity_info = defaultdict(dict)

    for prefix, filename in ENTITY_FILES.items():
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2] == prefix:
                    entity_id = parts[0]
                    entity_name = parts[1]
                    entities[prefix][entity_id] = entity_name
                    entity_info[entity_name] = {
                        'id': entity_id,
                        'type': prefix
                    }
    return entities, entity_info


def load_original_relations():
    relations = []
    for file in os.listdir(RELATION_JSON_DIR):
        if file.endswith('.json'):
            with open(os.path.join(RELATION_JSON_DIR, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                relations.extend(data if isinstance(data, list) else [data])
    return relations


def extract_relationships(entities, entity_info, original_relations):
    relationships = []

    for doc in original_relations:
        for context in doc.get('indexed_contexts', []):
            head = context.get('head_entity', {})
            head_name = head.get('name', '')

            if head_name not in entity_info:
                continue

            head_id = entity_info[head_name]['id']

            for rel in context.get('relationships', []):
                original_rel = rel.get('relation', '')
                standardized_rel = standardize_relation(original_rel)

                if standardized_rel not in RELATION_MAPPING.values():
                    continue

                for tail_name in rel.get('tail_entities', []):
                    if tail_name in entity_info:
                        relationships.append(
                            f"{head_id},{entity_info[tail_name]['id']},{standardized_rel}"
                        )

    return relationships


def save_relationships(relationships):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("from,to,relation\n")
        for rel in sorted(set(relationships)):
            f.write(f"{rel}\n")


def main():
    print("Begin...")

    entities, entity_info = load_entities()
    original_relations = load_original_relations()

    relationships = extract_relationships(entities, entity_info, original_relations)

    save_relationships(relationships)
    print(f"\nDown: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()