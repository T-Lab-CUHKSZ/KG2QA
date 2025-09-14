# KG2QA: Knowledge Graph-enhanced Retrieval-Augmented Generation for Communication Standards Question Answering

<div align="center">

**Paper:** [https://arxiv.org/abs/2506.07037](https://arxiv.org/abs/2506.07037) <br>

**Email:** luozhongze0928@foxmail.com <br>

</div>

The rapid evolution of communication technologies has led to an explosion of standards, rendering traditional expert-dependent consultation methods inefficient and slow. To address this challenge, we propose **KG2QA**, an intelligent question-answering system for communication standards that integrates fine-tuned large language models (LLMs) with a domain-specific knowledge graph (KG) via a Retrieval-Augmented Generation (RAG) framework. We construct a high-quality dataset of 6,587 QA pairs from ITU-T recommendations and fine-tune Qwen2.5-7B-Instruct, achieving significant performance gains: BLEU-4 increases from 18.86 to 66.90, outperforming both the base model and Llama-3-8B-Instruct. A structured KG containing 13,906 entities and 13,524 relations is built using LLM-assisted triple extraction based on a custom ontology. In our RAG pipeline, the fine-tuned LLM first retrieves relevant knowledge from the KG, enabling more accurate and factually grounded responses. Evaluated by DeepSeek-V3 as a judge, the KG-enhanced system improves performance across five dimensions, with an average score increase of 2.26\%, demonstrating superior factual accuracy and relevance. Integrated with Web system and API, KG2QA delivers an efficient and interactive user experience. Our code and data have been open-sourced.

<div align="center">

</div>

## Citation

```
@article{luo2025kg2qa,
  title={KG2QA: Knowledge Graph-enhanced Retrieval-Augmented Generation for Communication Standards Question Answering},
  author={Luo, Zhongze and Wan, Weixuan and Zheng, Qizhi and Bai, Yanhong and Sun, Jingyun and Wang, Jian and Wang, Dan},
  journal={arXiv preprint arXiv:2506.07037},
  year={2025}
}
```

![img](./kg2qa.png)

## ft

This dataset can be used for fine-tuning large language models in the field of communication standards. The data is sourced from `ITU-T Recommendations` and the language is English. You can click [here](https://www.itu.int/en/ITU-T/publications/Pages/recs.aspx) to visit the ITU-T official website and view the source file.

The dataset is divided into the training set and the test set. The training set is composed of data of the three categories of `G, V, X`, all of which are saved in `JSON`, and the test set is `test.json`.

The dataset was obtained by QA extraction of the source file PDF using the `Deepseek V3 API`, and the data volume is as follows.

288 PDF source files are stored in the `G, V, X` folders corresponding to their category sources.

| Category | Quantity of QA |
| :----: | :--: |
| G.json | 898 |
| V.json | 1,101 |
| X.json | 3,779 |
| test.json | 809 |
| All Categories | 6,587 |

The data is derived from ITU-T Recommendations and is taken from three series of recommendations. A total of `288 PDF source files` were selected for QA construction in the three series. The specific category sources are as follows.

```
G series： Transmission systems and media, digital systems and networks (A total of 34 PDF source files)
  G.600-G.699：Transmission media and optical systems characteristics
    G.650-G.659：Optical fibre cables
    G.660-G.679：Characteristics of optical components and subsystems
    G.680-G.699： Characteristics of optical systems

V series：Data communication over the telephone network (A total of 77 PDF source files)
  V.1-V.9：General
  V.10-V.34：Interfaces and voiceband modems
  V.35-V.39：Wideband modems
  V.40-V.49：Error control
  V.50-V.59：Transmission quality and maintenance
  V.60-V.99：  Simultaneous transmission of data and other signals
  V.100-V.199：Interworking with other networks
  V.200-V.249：Interface layer specifications for data communication
  V.250-V.299：Control procedures
  V.300-V.399：Modems on digital circuits

X series：Data networks, open system communications and security (A total of 177 PDF source files)
  X.1-X.199：Public data networks
    X.1-X.19：Services and facilities
    X.20-X.49：Interfaces
    X.50-X.89：Transmission, signalling and switching
    X.90-X.149：Network aspects
    X.150-X.179：Maintenance
    X.180-X.199：Administrative arrangements
  X.300-X.399：Interworking between networks
    X.300-X.349：General
    X.350-X.369：Satellite data transmission systems
    X.370-X.379：IP-based networks
  X.1000-X.1099：Information and network security
    X.1000-X.1011：Security orchestration and service access
    X.1030-X.1049： Network security
    X.1050-X.1079：Security management
    X.1080-X.1099：Telebiometrics
  X.1700-X.1729：Quantum communication
    X.1702-X.1704：Quantum random number generator
    X.1705-X.1749：Quantum Key Distribution Network (QKDN)
  X.1750-X.1799：Data security
    X.1750-X.1759：Big Data Security
    X.1770-X.1799：Data protection (II)
```

### qwen.yaml
Fine-tuning Experimental Setting

### deepseek_eval.py
Fine-tuned Model Evaluation Code (DeepSeek example)

## kg

### View the ontology on Protege

Ontology are saved in the ``kg`` folder. After opening ``Protege.exe``, click File-Open and select ``on.rdf`` to open it

### View the knowledge graph on Neo4j

1. Node and relationship data are saved in the ``import`` folder in CSV UTF-8 format. Save this folder in the local Neo4j path, and then ForPKG can be stored and displayed.

```
D:\neo4j-community-5.18.1\import
```

2. Restart neo4j
On the computer, type ``cmd`` to enter the command line. Go to ``neo4j-community-4.3.18\bin``, type ``neo4j restart`` to restart neo4j. In the browser, type ``localhost:7474/browser/`` to enter neo4j.

```
cd D:\neo4j-community-5.18.1\bin
neo4j restart
```

3. Click the database icon and click ``:dbs`` of DBMS

### Explanation

#### Neo4j

```
1. Import a specific entity type
LOAD CSV WITH HEADERS FROM 'file:///file_name.csv' AS line # file_name represents a specific entity type, and file_name.csv is the CSV file for that entity type  
MERGE (:file_name { ID: line.ID, name: line.name, LABEL: line.LABEL })

2. Delete a specific entity type  
MATCH (r:`file_name`) DETACH DELETE r  // file_name represents a specific entity type  

3. Create all relationships  
LOAD CSV WITH HEADERS FROM 'file:///roles.csv' AS row  // roles.csv is the CSV file containing all relationships  
MATCH (fromNode {ID: row.from}), (toNode {ID: row.to})  
CALL apoc.create.relationship(fromNode, row.relation, {}, toNode) YIELD rel  
RETURN rel  

4. Delete a specific relationship  
MATCH ()-[r:duty]-() DETACH DELETE r  

5. Delete all relationships  
MATCH ()-[r]-() DETACH DELETE r  

6. Display all entities and relationships  
MATCH (n) RETURN n  
```
#### The format of file_name.csv
```
ID,name,LABEL  
```
#### The format of roles.csv  
```
from,to,relation  
```
## rag

### kg2llm.py
Ollama Model Remote Access to Knowledge Graph
### eval.py
Quantitative Evaluation of Answer Quality using an LLM-based Judge
### flaskweb.py
Knowledge Graph QA Web Service Platform
### fastapi.py
Knowledge Graph QA API Service Platform
