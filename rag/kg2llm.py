import re
from neo4j import GraphDatabase
import ollama
import nltk 

class KnowledgeGraphQA:
    def __init__(self, uri, user, password, model_name="my-custom-model:latest"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_indexes = [
            "act_name_ft", "appcon_name_ft", "fun_name_ft",
            "iden_name_ft", "strcom_name_ft", "value_name_ft"
        ]
        self.model_name = model_name
        self.max_context_tokens_for_llm = 800 

    def close(self):
        self.driver.close()

    def query_neo4j(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    @staticmethod
    def escape_lucene_query(text: str) -> str:
        text = text.replace("&&", r"\&&").replace("||", r"\||")
        lucene_specials = r'([+\-!(){}\[\]^"~*?:\\/])'
        return re.sub(lucene_specials, r'\\\1', text)

    def get_related_nodes(self, keyword, limit_per_index=3): 
        escaped_keyword = self.escape_lucene_query(keyword)
        all_results = []

        for index_name in self.node_indexes:
            cypher = f"""
            CALL db.index.fulltext.queryNodes("{index_name}", "{escaped_keyword}")
            YIELD node, score
            MATCH (node)-[r]-(related)
            RETURN DISTINCT
                node,
                type(r) AS relationship_type,
                related,
                score,
                labels(node) AS node_labels,
                labels(related) AS related_labels
            ORDER BY score DESC
            LIMIT {limit_per_index} 
            """ 
            try:
                results = self.query_neo4j(cypher)
                all_results.extend(results)
            except Exception as e:
                print(f"Error querying Neo4j index {index_name}: {e}")
                continue
   
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:7] # Ambil top 7 hasil gabungan, misalnya

    @staticmethod
    def format_props(props):

        excluded_keys = ['name', 'NAME', 'id', 'ID', 'uri', 'embedding'] # Tambahkan properti yang tidak relevan
        items = []
        for k, v in props.items():
            if k.lower() not in excluded_keys and v is not None:
                v_str = str(v)
                if v_str.strip() and v_str.lower() not in ('nan', 'none', ''):
                    # Buat lebih natural: "property key is property value"
                    items.append(f"{k.replace('_', ' ')} is '{v_str}'")
        return "; ".join(items) if items else "" # Return empty string if no other properties

    def format_context(self, results):

        context_parts = []
        if not results:
            return "No specific information found in the knowledge graph for this query."

        for record in results:
            node = record['node']
            rel_type = record['relationship_type'].replace('_', ' ').lower() # Lebih natural
            related = record['related']
            
            node_name = node.get('name', node.get('NAME', 'Unnamed Node'))
            related_name = related.get('name', related.get('NAME', 'Unnamed Related Item'))

            node_labels_str = "/".join(record['node_labels']) if record['node_labels'] else "entity"
            related_labels_str = "/".join(record['related_labels']) if record['related_labels'] else "entity"

            node_props_str = self.format_props(node)
            related_props_str = self.format_props(related)
            
            part = f"The {node_labels_str} '{node_name}'"
            if node_props_str:
                part += f" (which {node_props_str})"
            part += f" is {rel_type} the {related_labels_str} '{related_name}'"
            if related_props_str:
                part += f" (which {related_props_str})"
            part += "."
            context_parts.append(part)
        
        full_context = " ".join(context_parts) 
        
        tokens = nltk.word_tokenize(full_context)
        if len(tokens) > self.max_context_tokens_for_llm:
            tokens = tokens[:self.max_context_tokens_for_llm]
        
        final_context_for_llm = " ".join(tokens)
        if not final_context_for_llm.strip() or final_context_for_llm == "No specific information found in the knowledge graph for this query.":
             return "No specific, actionable information was found in the knowledge graph for this query."
        
        return f"Here is some potentially relevant information from a knowledge graph: {final_context_for_llm} Use this information to ensure your answer is factually accurate and detailed where appropriate."


    def generate_response(self, question, context): # context sekarang sudah diformat dan dibatasi

        prompt = f"""You are an expert in the domain of the question and a skilled professional communicator.
Your goal is to provide the best possible answer to the question.
Use the 'Related knowledge graph context' provided below *only* to ensure factual accuracy and to add specific, relevant details if they enhance the answer's quality and completeness.
Do not simply repeat or list information from the context if it doesn't naturally fit into a high-quality answer.
The answer should be primarily in your own words, well-structured, clear, concise, and in professional English.

Question:
{question}

Related knowledge graph context:
{context} 

Based on your expertise and the provided context (if relevant and accurate), please answer the question:
Answer:
"""
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            print(f"Error generating Ollama response in KGQA: {e}")
            return f"Error generating KG+LLM response: {e}"


def main():
    print("Knowledge Graph Question-Answer System (Modified)")
    print("Type 'exit' or 'quit' at any prompt to leave.\n")

    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', download_dir='/home/h3c/nltk_data') 
        nltk.data.path.append('/home/h3c/nltk_data') 

    except LookupError: # Jika sudah diunduh tapi path belum benar
        try:
            nltk.data.path.append('/home/h3c/nltk_data') 
            nltk.data.find('tokenizers/punkt')
        except:
            print("Error: NLTK 'punkt' not found. Please ensure it's downloaded and nltk.data.path is set.")
            return

    kgqa = KnowledgeGraphQA(
        uri="bolt://10.171.63.13:7687",
        user="neo4j",
        password="lzz928666",
        model_name="my-custom-model:latest"
    )

    try:
        while True:
            search_term = input("Enter a keyword to search the knowledge graph (or 'exit'): ").strip()
            if search_term.lower() in ["exit", "quit"]:
                print("Exiting system. Goodbye!")
                break
            if not search_term:
                continue

            print("Searching knowledge graph...")
            results = kgqa.get_related_nodes(search_term)
            
            formatted_context = kgqa.format_context(results) 
            print(f"\nFormatted KG Context (will be part of LLM prompt):\n{formatted_context}\n")

            if "No specific" in formatted_context or "No actionable" in formatted_context :
                print("No sufficiently relevant information found in KG to form a detailed context. You can still ask a general question.\n")

            while True:
                question = input("Enter your question (or type 'new' for new keyword, 'exit' to quit): ").strip()
                if question.lower() in ["exit", "quit"]:
                    print("Exiting system. Goodbye!")
                    kgqa.close()
                    return
                if question.lower() == "new":
                    print("\nStarting new keyword search...\n")
                    break
                if not question:
                    print("Please input a valid question.")
                    continue

                print("\nGenerating KG+LLM answer...")
                answer = kgqa.generate_response(question, formatted_context) 
                print(f"\nProfessional Answer (KG+LLM):\n{answer}\n")

    finally:
        kgqa.close()

if __name__ == "__main__":
    main()