
# Ref

[[논문리뷰]Clinical Natural Language Processing for Radiation Oncology: A Review and Practical Prime(Red journal, Jan 2021)](https://velog.io/@sjinu/Clinical-Natural-Language-Processing-for-Radiation-Oncology-A-Review-and-Practical-PrimeJan-2021)
[[논문리뷰]A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios(ACL Anthology, Jun 2021)](https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0A-Survey-on-Recent-Approaches-for-Natural-Language-Processing-in-Low-Resource-ScenariosACL-Anthology-Jun-2021)
[[논문리뷰] Deep learning in clinical natural language processing: a methodical review](https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Deep-learning-in-clinical-natural-language-processing-a-methodical-review)
[[논문정리] Survey : Survey papers for Clincal NLP(for 6 papers)](https://velog.io/@sjinu/Abstracts-of-several-review-paper-Clinical-NLP)
[[논문리뷰] Medical Visual Question Answering: A Survey](https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Medical-Visual-Question-Answering-A-Survey)

---



# Low-Resource Techniques


> [[논문리뷰]A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios(ACL Anthology, Jun 2021)](https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0A-Survey-on-Recent-Approaches-for-Natural-Language-Processing-in-Low-Resource-ScenariosACL-Anthology-Jun-2021)

## Data Augmentation


>어떤 방식으로든 데이터를 늘릴 수 있는 기법들


- **Data Augmentation**
    - 필요 : labeled data
    - 결과 : additional labeled data
        - Token-level
            - 동의어, 같은 type의 entity, 같은 형태소, language model 등 이용
        - Sentence-level
            - 피동형↔ 사동형, 문장 단순화, 역번역(한국어→일본어→한국어, 등)
            - Adversarial Methods 활용(라벨이 바뀌지 않는 선에서 문장을 변형(간섭))
- **Distant Supervision**
    - 필요 : unlabeled data
    - 결과 : additional labeled data
        - 외부 데이터 베이스, Rule, Deep Learning 등을 이용해 라벨링을 하는 방법론들
            - e.g. [META: Metadata-Empowered Weak Supervision for Text Classification](https://aclanthology.org/2020.emnlp-main.670/)
        - Pre-trained language model을 이용해 input에 제일 가까운 label sentence를 생성한다거나..
            - e.g. “This Movie was ~~. and ~~. ~~~” 등의 document를 받았을 때, “So, it was (   )” 라는 문장을 채우게끔 Language 모델링을 활용할 수 있고, 위의 빈칸은 Label로 활용할 수 있음.
- **Cross-lingual projection**
    - 필요 : low-resource unlabeled data, high-resource labeled data, cross-lingual alignment)
    - 결과 : additional labeled data
        - Label이 풍부한 도메인을 이용해 Label이 별로 없는 도메인의 데이터를 늘리는 방법.
        - Parallel corpora + 두 도메인을 연결짓는 alignment를 활용.
- **Other Technique for handling noisy label**
    - Noise Filtering
    - noise Modeling
        - 위에서 말한 방법으로 Labeled data를 보충한다면, 애초에 완벽하지 않은 모델을 사용해 Data를 늘리기 때문에, error가 쌓일 수밖에 없다(pseudo labeling은 조금 위험).
        - 이런 noisy label을 다루기 위한 테크닉들은 반 필수적.

## Representation-related

>Language Model, Transfer Learning 등을 이용해 language representation을 활용하는 방법들


- **Subword-based embeddings**
    - byte-pair-encoding embeddings
        - 이와 같은 방법으로 인코딩을 하면 voca의 복잡도도 낮출 수 있고, low-resource task에서 좋은 결과를 보인다는 연구 결과가 있음.
        - Open-AI의 연구 DALL-E가 이 방법을 활용해 Text를 인코딩함.
- **Embeddings & Pre-trained LMs**
    - 필요 : Unlabeled data(**non-clinical texts**)
    - 결과 : better language representation
        - BERT, GPT 등.
- **LM Domain Adaptations**
    - 필요 : existing LM , unlabeled domain data(**clinical texts**)
    - 결과 : domain-specific language representation
        - [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/)
        - BioBERT, ClinicalBERT(?) 등이 여기에 해당
- **Multilingual LMs**
    - 필요 : multilingual unlabeled data
    - 결과 : multilingual feature representation
- **Adversarial Discriminator**
    - 필요 : additional datasets
    - 결과 : independent representation
        - pre-trained domain과 target domain(downstream domain) 간의 차이를 고려하는 방법
        - 즉, 모델이 general한 feature embeddings을 학습하게끔 적대적 학습을 이용.
        - 최근까지 꽤나 많이 쓰이고 있음.
- **Meta-embeddings**
    - 필요 : domain-specific Embedding,  domain-agnostic(general domain) Embedding
    - 결과 : Meta-embeddings
        - Concatenation, outer-product, bi-linear Pooling, element-wise 등 다양한 방법으로 두 개의 embedding을 fusion해 사용할 수 있음.
        - 때로는 Adversarial discriminator를 이용할 수도 있음.
            - [Adversarial learning of feature-based meta-embeddings](https://arxiv.org/pdf/2010.12305.pdf)
- **Other Discussion**
    - 단, Pre-trained Representation 모델들은 일반적으로 모델이 크기 때문에 사용하기 힘들 수 있다.
    - 이 때는 Large-scale Model의 성능을 훨씬 적은 parameter의 model로 따라잡으려는 여러 연구들을 참고할 수 있다.
        - [It’s not just size that matters: Small language models are also few-shot learners](https://arxiv.org/abs/2009.07118)
        - 말고도 GPT의 1/100 size로 성능을 비슷하게 맞춘 연구도 있었던 걸로 기억.

## Others

- Meta-Learning
    - 필요 : multiple auxiliary tasks
    - 결과 : better target task performance
        - 사실 활용하기가 쉽지는 않음(애초에 Few-Shot 주제를 잡고 연구를 진행하지 않는 이상)

---

# Tasks

## Only Text

- **Text Classification**
    - Document-level : 일반적
    - Sentence-level : 조금 드물다.
- **Named Entity Recognition**
    - 의료 문서에서 질병 : (   )  치료법 : (   )  환자 정보 : (   )
    - Language Model 이용해서 해도 재밌을듯
- **Relation Extraction(RE)**
    - entity 간의 관계 파악.
    - 첫번째 진단과 그 다음 진단과의 관계..
    - 역시 Generative Language Model로?
- **Question Answering**
    - 외부 지식을 사용하거나(information retrieval(검색)-based_
    - 내부 지식만을 사용하거나(knowledge-based)
- **Summarization**
- **Natural Language Inference**
    - 두 문장에 대한 관계분류
    - entailment, contradiction, neutral
- **Automated inclusion of radiation details into cancer registries(자동 기록)**
- **Event extraction**
    - radiation treatment를 했는지 안 했는지
- **Temporal expression** extraction
    - radiation therapy를 받은 기간을 라벨링..
    - Rule-based or Pattern learning-based(Pan et al)
    - extraction →classification → normalization
- **Template filling**
    - Language Model의 극한적 활용
- **Negation(반대는 attribute)**
    - 주어진 context를 기반으로 특정 entity가 존재하지 않는지, 존재하는지
    - ‘~~~부스트샷을 맞는 게 좋다. 하지만 이 환자는 나이가 30살이라 실시하지 않는다.’ → Positive
- **Cross-lingual concept extraction**
    - 언어 2개를 사용한다면.
- **Natural Language Generation**
- 

- ~~Machine Translation(pass)~~
- ~~Chatbot(pass)~~

## Text & Vision

- Image Captioning
- VQA

---

# Survey papers

## Text

[Clinical Decision Support Systems: A Survey of NLP-Based Approaches from Unstructured Data | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/7406286)

[JMIR Medical Informatics - Health Natural Language Processing: Methodology Development and Applications](https://medinform.jmir.org/2021/10/e23898/)

[Clinical Natural Language Processing for Radiation Oncology: A Review and Practical Primer - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0360301621001188)

[Deep learning in clinical natural language processing: a methodical review | Journal of the American Medical Informatics Association | Oxford Academic (oup.com)](https://academic.oup.com/jamia/article/27/3/457/5651084?login=true)

[[2010.12309] A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios (arxiv.org)](https://arxiv.org/abs/2010.12309)

[SECNLP: A survey of embeddings in clinical natural language processing - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046419302436)

## Image & Text

[Medical Visual Question Answering: A Survey](https://arxiv.org/pdf/2111.10056.pdf)



---

# Models(Architectures)

## VQA

- MedFuseNet(2021.10 SoTA)

## Transfer Learning

### Pre-trained Model

- **BioBERT**
- ClinicalBERT
- PubMedBERT

### Pre-training articles

- **Biomedical literature (PubMed)**
- Clinical notes (MIMIC - |||)
- Health websites

## Adaptation Methods

> 위에서 살짝 언급되어있긴 하지만.

- Domain Adaptations
- Adapter

---

# Data

## Metrics

- Accuracy
- Recall
- Precision
- F1 score


## Unstructured data(free-text)

- diagnosis text(진단 텍스트)
- discharge summaries(퇴원 요약서)
- 온라인 의학 토론
- Biomedical literature
  - PubMed
- Clinical notes
  - MIMIC - 3
- Health websites



## Structured data

### Data type

- Structured data in **EMR**
   - diagnosis codes(진단 코드)
   - admission events(입원 관련)
   - discharge evenets(퇴원 관련)
   - survival outcomes(생존 결과..?)

### Kaggle

- Clinical Trials on Cancer(Eligibility criteria for **Text classification**)
    - Intervention & Cancer type → Label(eligible / not eligible)
    - [https://www.kaggle.com/auriml/eligibilityforcancerclinicaltrials/version/1](https://www.kaggle.com/auriml/eligibilityforcancerclinicaltrials/version/1%E2%80%8B)
- **Sentiment Analysis** for Medical Drugs(Medical Drugs Sentiment Analysis with Multi-classes
    - Text(comments) & Drug(entity, medication) →Sentiment(neutral , negative, positive)
- Medical text for **Text Classification**
    - Medical abstract →5 classes(degestive, cardiovascular, ,.,,)
    - [https://www.kaggle.com/chaitanyakck/medical-text?select=train.dat](https://www.kaggle.com/chaitanyakck/medical-text?select=train.dat%E2%80%8B)
- Coronavirus tweets NLP - **Sentiment Analysis**
    - Text(tweets) →Sentiment (negative, positive, neutral, extremly negative, extreamly positive)
    - [https://www.kaggle.com/datatattle/covid-19-nlp-text-classification](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification%E2%80%8B)
- **Medical Transcriptions**
    - Transcriptions → Medical Specialities
    - [https://www.kaggle.com/tboyle10/medicaltranscriptions](https://www.kaggle.com/tboyle10/medicaltranscriptions%E2%80%8B)
    

### Benchmark

**corpus** 

- **i2b2 challenges**
    - NER(Named Entity Recognition) / Concept extraction
    - RE(Relation Extraction)
- **SemEval challenges**
    - Temporal event / relations
- **MIMIC data**
    - Text Classification
- **CCKS challenge**
- **MADE challenge**
- **THYME public data**






---


# Research

## According to Task

### Clinical timeline creation

- SemEval-2016 Task 12: Clinical TempEval. 
- SemEval-2017 Task 12: Clinical TempEval.
- Neural architecture for temporal relation extraction: A Bi-LSTM approach for detecting narrative containers. 
- Representations of time expressions for temporal relation extraction with convolutional neural networks
- Neural temporal relation extraction.

### Automated clinical trial matching

#### eligibility criteria
- Automated classification of eligibility criteria in clinical trials to facilitate patient-trial matching for specific patient populations.

#### patient specific eligibility
- Increasing the efficiency of trialpatient matching: Automated clinical trial eligibility pre-screening for pediatric oncology patients.
  - 일종의 **trial eligibility prescreening**이라고 보면 됨.
- Automatic trial eligibility surveillance based on unstructured clinical data. 
 
### NER

- Bitterman et al. Extracting radiotherapy treatment details using neural network-based natural language processing. : **High Performing deep learning model for NER**

### RE
- Bitterman et al. Extracting relations between radiotherapy treatment details. :  **High Performing deep learning model for RE**


## According to Domain
### Cancer(암)에 대한 연구

#### Document-level

- Validation of case finding algorithms for hepatocellular cancer from administrative data and electronic health records using natural language processing. 
- Application of text information extraction system for real-time cancer case identification in an integrated healthcare organization. 
- Pathologic findings inreduction mammoplasty specimens: A surrogate for the population prevalence of breast cancer and high-risk lesions.
- Development and validation of a natural language processing algorithm for surveillance of cervical and anal cancer and precancer: A split-validation study
- [Assessment of deep natural language processing in ascertaining oncologic outcomes from radiology reports](https://jamanetwork.com/journals/jamaoncology/fullarticle/2738774)는 
  - **CNN**과  **radiology reports**를 활용해 **cancer output**을 식별


#### EMR record-level
- Automated ancillary cancer history classification for mesothelioma
patients from free-text clinical reports.
- Extracting and integrating data from entire electronic health records for detecting colorectal cancer cases.

#### Extracting cancer attributes

**cancer attributes**

- primary site
- tumor
- location
- stage

**Radiology**

- Automated identification of patients with pulmonary nodules in an integrated health system using administrative health plan data, radiology reports, and natural language processing.
- Automated annotation and classification of BI-RADS assessment from radiology reports. 

## According to Gold-annotated clinical texts dataset
$*$  **SemEval** : Proceedings of the 11th International Workshop on Semantic Evaluation

11. DeepPhe: A natural language processing system for extracting cancer phenotypes from clinical
records. 
63. SemEval-2015 Task 6: Clinical TempEval(SemEval 2015).
93. SemEval-2016 Task 12: Clinical TempEval(SemEval-2016).
94. SemEval-2017 Task 12: Clinical TempEval(SemEval-2017). 


118. 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text.
119. SemEval-2015 task 14: Analysis of clinical text(SemEval 2015)
120. DDiscovering body site and severity modifiers in clinical texts.
121. Evaluating the state of the art in coreference resolution for electronic
medical records. 
122. Towards generalizable entity-centric clinical coreference resolution. 
123. Evaluating the state-of-the-art in automatic de-identification. 
124. Recognizing obesity and comorbidities in sparse data.
125. Automated systems for the deidentification of longitudinal clinical narratives: Overview of 2014 i2b2/UTHealth shared task Track 1. 
126. De-identification of psychiatric intake records: Overview of 2016 CEGS N-GRID shared tasks Track
127. A shared task involving multi-label classification of clinical free text.
128. Towards comprehensive syntactic and semantic annotations of the clinical narrative. 

## According to Method

### Federated Learning

129. Federated learning of predictive models from federated Electronic
Health Records.
130. FADL: Federatedautonomous deep learning for distributed electronic health record.
131. Two-stage federated phenotyping and
patient representation learning. 
132. Communication-efficient learning of deep networks from decentralized data.
133. Federated learning: Strategies for improving communication efficiency.
134. Patient clustering improves efficiency of federated machine learning to
predict mortality and hospital stay time using distributed electronic medical records.
135. LoAdaBoost: Lossbased AdaBoost Federated Machine Learning on medical data.

### Pre-trained Langauge Model
- nonclinical texts를 이용하거나, non-radiation-specific texts를 학습해 radiation oncology에 이용하거나.

39. BERT: Pre-training of deep bidirectional transformers for language understanding. 
40. Deep contextualized word representations. 
136. Contextual string embeddings for sequence labeling.

---


# Software(?)

## Extracting features in EMR texts

- Deep Phe
  - 암에 대한 document- & patient-level의 summarization 추출
  - cancer attributes 추출
- cTAKES(Apache Clinical Text Analysis Knowledge Extraction System) 
  - unstructured radiation therapy로부터 toxicity data를 추출
  
  
