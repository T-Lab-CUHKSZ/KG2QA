import os
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm
import json

# Configuration parameters
API_KEY = "xxx"  # Replace with your API Key
PDF_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutputa"  # PDF directory path
OUTPUT_DIR = "E:/Users/yeming/PycharmProjects/LLMProject/finaloutputb"  # Output directory path
MODEL = "deepseek-chat"  # DeepSeek model
MAX_TOKENS = 4000  # Maximum tokens (adjust according to model)
BATCH_SIZE = 5  # Batch size (controls API call frequency)

# Initialize DeepSeek client
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


def extract_entities_with_deepseek(text, pdf_name, retries=3):
    """Call DeepSeek API for entity recognition"""
    prompt = f"""
    You are a professional academic information extraction expert. Please extract head entities (head entities) from the following text as comprehensively as possible. Each document should extract 30-40 entities.

    # Extraction Requirements
    1. Head entities are the starting nodes in the knowledge graph's triple (Head Entity, Relation, Tail Entity), representing the entities that initiate the relationship or act as the subject of the relationship. For example:
       - In the triple "Apple Inc. - Founded in - 1976", "Apple Inc." is the head entity.
       - In "Beijing - Is - Capital of China", "Beijing" is the head entity.
    2. Each entity must include all applicable type labels (multiple selections are possible).
    3. Extract entities of the following types:
       - (Identifier)
       - (Structure/Composition)
       - (Action)
       - (Value)
       - (Function)
       - (Applicability/Context)

    # Example (Do not include this example in actual extraction)
    {{
        "document": "sample.md",
        "entities": [
            {{
                "index": 1,
                "entity": "ITU-T G.652 fiber",
                "type": ["Identifier", "Structure/Composition", "Applicability/Context"],
                "context": "ITU-T G.652 fiber was originally optimized for the 1310 nm wavelength region but can also be used in the 1550 nm region.",
                "confidence": 0.92,
            }},
            {{
                "index": 2,
                "entity": "Wavelength Division Multiplexing (WDM) technology",
                "type": ["Action", "Function"],
                "context": "Wavelength Division Multiplexing (WDM) technology can significantly increase the transmission capacity of optical fibers.",
                "confidence": 0.88,
            }}
        ]
    }}

    # Special Notes
    1. Do not omit any entities that may meet the criteria in the text.
    2. Label each entity with as many applicable types as possible (usually 2-3 types).
    3. For long documents, ensure that the number of extractions reaches 30-40.
    4. Head entities are the basic units of a knowledge graph, connecting to tail entities through relationships to form a knowledge network. For example: (Head Entity: Apple Inc.) → (Relationship: Headquarters) → (Tail Entity: Cupertino)

    # Text to be analyzed (from document: {pdf_name})
    {text[:MAX_TOKENS * 4]}  # Increase text truncation length

    # Please return the results strictly in the following JSON format:
    {{
        "document": "File Name",
        "entities": [
            {{
                "index": Index,
                "entity": "Entity Name",
                "type": ["Type 1", "Type 2", ...],  # Must use list format
                "context": "Context of Appearance",
                "confidence": Confidence (0-1),
            }}
        ],
        "extraction_stats": {{
            "total_entities": Total number of entities,
            "coverage_estimate": "Document coverage estimate"
        }}
    }}
    """




    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.1,  # Low randomness for stability
                response_format={"type": "json_object"}  # Force JSON output
            )

            # Parse and validate returned JSON
            result = json.loads(response.choices[0].message.content)
            if "entities" in result:
                # Add document name if not present
                result["document"] = pdf_name
                return result
            else:
                print(f"Unexpected response format: {result}")
                return {"document": pdf_name, "entities": []}

        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt + 1}/{retries}): {e}")
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{retries}): {e}")

    return {"document": pdf_name, "entities": []}


def process_pdf_batch(pdf_batch):
    """Process a batch of PDF files"""
    batch_results = []
    for pdf_name in pdf_batch:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        text = extract_text_from_pdf(pdf_path)
        if text:
            result = extract_entities_with_deepseek(text, pdf_name)
            batch_results.append(result)
    return batch_results


def save_results(results, batch_num):
    """Save results to JSON file"""
    output_path = os.path.join(OUTPUT_DIR, f"entities_batch_{batch_num}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files, starting processing...")

    # Process in batches
    for i in tqdm(range(0, total_files, BATCH_SIZE), desc="Processing"):
        batch = pdf_files[i:i + BATCH_SIZE]
        batch_results = process_pdf_batch(batch)
        save_results(batch_results, i // BATCH_SIZE + 1)

    print(f"Processing complete! Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()