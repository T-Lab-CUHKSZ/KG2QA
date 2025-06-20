# KG2QA: Knowledge Graph-enhanced Retrieval-Augmented Generation for Communication Standards Question Answering

<div align="center">

**Paper:** [https://arxiv.org/abs/2506.07037](https://arxiv.org/abs/2506.07037) <br>

[Zhongze Luo](https://luozhongze.github.io)#, Weixuan Wan#, Qizhi Zheng, Yanhong Bai, Jingyun Sun, Jian Wang♠, Dan Wang♠  <br>
\# Equal Contribution; ♠ Corresponding authors <br>

**Email:** luozhongze0928@foxmail.com <br>

</div>

There are many types of standards in the field of communication. The traditional consulting model has a long cycle and relies on the knowledge and experience of experts, making it difficult to meet the rapidly developing technological demands. This paper combines the fine-tuning of large language models with the construction of knowledge graphs to implement an intelligent consultation and question-answering system for communication standards. The experimental results show that after LoRA tuning on the constructed dataset of 6,587 questions and answers in the field of communication standards, Qwen2.5-7B-Instruct demonstrates outstanding professional capabilities in the field of communication standards on the test set. BLEU-4 rose from 18.8564 to 66.8993, and evaluation indicators such as ROUGE also increased significantly, outperforming the fine-tuning effect of the comparison model Llama-3-8B-Instruct. Based on the ontology framework containing 6 entity attributes and 10 relation attributes, a knowledge graph of the communication standard domain containing 13,906 entities and 13,524 relations was constructed, showing a relatively good query accuracy rate. The intelligent consultation and question-answering system enables the fine-tuned model on the server side to access the locally constructed knowledge graph and conduct graphical retrieval of key information first, which is conducive to improving the question-answering effect. The evaluation using DeepSeek as the Judge on the test set shows that our RAG framework enables the fine-tuned model to improve the scores at all five angles, with an average score increase of 2.26%. And combined with web services and API interfaces, it has achieved very good results in terms of interaction experience and back-end access, and has very good practical application value.

<div align="center">

</div>

## 1. ft_dataset

This dataset can be used for fine-tuning large language models in the field of communication standards. The data is sourced from `ITU-T Recommendations` and the language is English. You can click [here](https://www.itu.int/en/ITU-T/publications/Pages/recs.aspx) to visit the ITU-T official website and view the source file.

The dataset is divided into the training set and the test set. The training set is composed of data of the three categories of `G, V, X`, all of which are saved in `JSON`, and the test set is `test.json`.

The dataset was obtained by QA extraction of the source file PDF using the `Deepseek V3 API`, and the data volume is as follows.

288 PDF source files are stored in the `G, V, X` folders corresponding to their category sources.

| Category | Quantity of QA |
| :----: | :--: |
| G.json | 898 |
| V.json | 1101 |
| X.json | 3779 |
| test.json | 809 |
| All Categories | 6587 |

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

## 2. kg

## 3. rag_code

