from flask import Flask, request, render_template_string, redirect, url_for
from neo4j import GraphDatabase
import ollama
import concurrent.futures

class KnowledgeGraphQA:
    def __init__(self, uri, user, password, model_name="my-custom-model:latest"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_indexes = [
            "act_name_ft",
            "appcon_name_ft",
            "fun_name_ft",
            "iden_name_ft",
            "strcom_name_ft",
            "value_name_ft"
        ]
        self.model_name = model_name

    def close(self):
        self.driver.close()

    def query_neo4j(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    def get_related_nodes(self, keyword):
        all_results = []
        for index_name in self.node_indexes:
            cypher = f"""
            CALL db.index.fulltext.queryNodes("{index_name}", "{keyword}")
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
            LIMIT 5
            """
            results = self.query_neo4j(cypher)
            all_results.extend(results)
        return all_results

    @staticmethod
    def format_props(props):
        excluded = ['name', 'NAME', 'id', 'ID']
        items = [f"{k}: {v}" for k, v in props.items() if k not in excluded]
        return ", ".join(items) if items else "no other properties"

    def format_context(self, results):
        context_parts = []
        for record in results:
            node = record['node']
            rel_type = record['relationship_type']
            related = record['related']
            node_labels = record['node_labels']
            related_labels = record['related_labels']

            node_props = {k: v for k, v in node.items() if v is not None and str(v).lower() not in ('nan', 'none', '') and k.lower() not in ('id')}
            related_props = {k: v for k, v in related.items() if v is not None and str(v).lower() not in ('nan', 'none', '') and k.lower() not in ('id')}

            node_name = node_props.get('name', 'Unknown Entity')
            related_name = related_props.get('name', 'Unknown Entity')

            part = (
                f"{'/'.join(node_labels)} entity '{node_name}' "
                f"(properties: {self.format_props(node_props)})\n"
                f"is connected via '{rel_type}' relationship to\n"
                f"{'/'.join(related_labels)} entity '{related_name}' "
                f"(properties: {self.format_props(related_props)})"
            )
            context_parts.append(part)
        return "\n\n".join(context_parts)

    def generate_response(self, question, context):
        prompt = f"""
You are an expert in knowledge graph analysis. Please provide a professional answer based on the following question and knowledge graph information:

Question:
{question}

Related knowledge graph context:
{context}

Answer requirements:
1. First, answer the question directly.
2. Then, provide a detailed explanation combined with the knowledge graph information.
3. Use clear and professional English.
"""
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error generating response: {e}"

app = Flask(__name__)
kgqa = KnowledgeGraphQA(
    uri="bolt://10.171.63.13:7687",
    user="neo4j",
    password="lzz928666",
    model_name="my-custom-model:latest"
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

session_data = {
    "keyword": None,
    "context": None,
    "dialogue": []  # [(question, answer), ...]
}

HTML = """
<!doctype html>
<html>
<head>
    <title>Knowledge Graph Multi-turn QA</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; display:flex; gap:30px; }
        #left, #right { flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 6px; max-height: 80vh; overflow-y: auto; }
        h2 { margin-top: 0; }
        form { margin-bottom: 20px; }
        textarea, input[type=text] { width: 100%; padding: 8px; margin-top: 6px; box-sizing: border-box; }
        pre { background: #f0f0f0; padding: 10px; white-space: pre-wrap; border-radius: 4px; }
        .dialogue-block { margin-bottom: 20px; }
        .question { font-weight: bold; }
        .answer { margin-top: 5px; white-space: pre-wrap; background:#eef9ff; padding:10px; border-radius:4px; }
        button { padding: 10px 20px; margin-right: 10px; }
    </style>
</head>
<body>

<div id="left">
    <h2>Step 1: Knowledge Graph Search</h2>
    <form method="post" action="/search">
        <label>Enter Keyword:</label>
        <input type="text" name="keyword" required value="{{ keyword or '' }}">
        <button type="submit">Search</button>
    </form>
    {% if context %}
    <h3>Knowledge Graph Context:</h3>
    <pre>{{ context }}</pre>
    {% endif %}
</div>

<div id="right">
    <h2>Step 2: Dialogue</h2>
    {% if dialogue %}
        {% for q,a in dialogue %}
        <div class="dialogue-block">
            <div class="question">Q: {{ q }}</div>
            <div class="answer">A: {{ a }}</div>
        </div>
        {% endfor %}
    {% else %}
        <p>No dialogue yet. Please enter a question below.</p>
    {% endif %}

    {% if context %}
    <form method="post" action="/ask">
        <label>Enter your question:</label>
        <input type="text" name="question" required autofocus>
        <div style="margin-top:10px;">
            <button type="submit" name="action" value="ask">Ask / Continue</button>
            <button type="submit" name="action" value="restart">Restart (New Keyword)</button>
        </div>
    </form>
    {% else %}
        <p>Please search a keyword first to start.</p>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML,
                                  keyword=session_data["keyword"],
                                  context=session_data["context"],
                                  dialogue=session_data["dialogue"])

@app.route("/search", methods=["POST"])
def search():
    keyword = request.form.get("keyword").strip()
    if not keyword:
        return redirect(url_for("index"))

    results = kgqa.get_related_nodes(keyword)
    context = kgqa.format_context(results) if results else "No relevant information found."

    session_data["keyword"] = keyword
    session_data["context"] = context
    session_data["dialogue"] = []

    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    if not session_data["context"]:

        return redirect(url_for("index"))

    question = request.form.get("question").strip()
    action = request.form.get("action")

    if action == "restart":

        session_data["keyword"] = None
        session_data["context"] = None
        session_data["dialogue"] = []
        return redirect(url_for("index"))

    if not question:
        return redirect(url_for("index"))

    future = executor.submit(kgqa.generate_response, question, session_data["context"])
    answer = future.result()

    session_data["dialogue"].append((question, answer))

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
