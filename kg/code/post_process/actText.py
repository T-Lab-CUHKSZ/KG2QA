import os
import json
from tqdm import tqdm

# Configuration parameters
INPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutputd"
OUTPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutpute"
OUTPUT_FILENAME = "action_entities.txt"
ENTITY_TYPE = "Action"
ENTITY_LABEL = "ACT"


def load_json_files():
    """Load all JSON files from the directory"""
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    all_data = []

    for file in json_files:
        file_path = os.path.join(INPUT_DIR, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    return all_data


def extract_action_entities(data):
    """Extract all entities with type 'Action' from the data"""
    action_entities = []

    for document in data:
        for context in document.get('indexed_contexts', []):
            # Check head entity
            head_entity = context.get('head_entity', {})
            if isinstance(head_entity.get('type'), list) and ENTITY_TYPE in head_entity.get('type', []):
                action_entities.append(head_entity['name'])

            # Check tail entities
            for tail_entity in context.get('tail_entities', []):
                if tail_entity.get('type') == ENTITY_TYPE:
                    action_entities.append(tail_entity['entity'])

    return action_entities


def process_and_save_entities(action_entities):
    """Process entities and save to TXT file"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in action_entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)

    # Generate ACT codes
    entity_records = []
    for i, name in enumerate(unique_entities, 1):
        entity_records.append(f"{ENTITY_LABEL}{i:03d},{name},{ENTITY_LABEL}")

    # Write to TXT file
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("ID,name,LABEL\n")

            # Write each record
            for record in entity_records:
                f.write(f"{record}\n")

        return len(entity_records)
    except Exception as e:
        print(f"Error writing to {output_path}: {str(e)}")
        return 0


def main():
    # Verify input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory does not exist: {INPUT_DIR}")
        return

    # Load the JSON data
    print("Loading JSON files...")
    all_data = load_json_files()

    # Extract action entities
    print(f"Extracting {ENTITY_TYPE} entities...")
    action_entities = extract_action_entities(all_data)

    # Process and save entities
    print("Processing and saving entities...")
    count = process_and_save_entities(action_entities)

    if count > 0:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        print(f"Completed! Saved {count} unique {ENTITY_TYPE} entities to {output_path}")
    else:
        print(f"No {ENTITY_TYPE} entities were saved due to errors")


if __name__ == "__main__":
    main()