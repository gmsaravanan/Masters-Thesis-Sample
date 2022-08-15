# Potential Topics

List of Potential Topics to consider for the Thesis

# NLP
## Text Generation

### Text Generation

### Story Generation

<details><summary>Plot Writing From Pre-Trained Language Models</summary><p>

**Abstract** - Pre-trained language models (PLMs) fail to generate long-form narrative text because they do not consider global structure. As a result, the generated  texts are often incohesive, repetitive, or lack content. Recent work in story generation reintroduced explicit content planning in the form of prompts, keywords, or semantic frames. Trained on large parallel corpora, these models can generate more logical event sequences and thus more contentful stories. However, these intermediate epresentations are often not in natural language and cannot be utilized by PLMs without fine-tuning. We propose generating story plots using offthe-shelf PLMs while maintaining the benefit of content planning to generate cohesive and contentful stories. Our proposed method, SCRATCHPLOT, first prompts a PLM to compose a content plan. Then, we generate the story’s body and ending conditioned on the content plan. Furthermore, we take a generate-and-rank approach by using additional PLMs to rank the generated (story, ending) pairs. We benchmark our method with various baselines and achieved superior results in both human and automatic evaluation 1.

**Link** - https://paperswithcode.com/paper/plot-writing-from-pre-trained-language-models-1
</p></details>

<details><summary>Efficient Training of Language Models to Fill in the Middle</summary><p>

**Abstract** - We show that autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to its end. While this data augmentation has garnered much interest in recent years, we provide extensive evidence that training models with a large fraction of data transformed in this way does not harm the original left-to-right generative capability, as measured by perplexity and sampling evaluations across a wide range of scales. Given the usefulness, simplicity, and efficiency of training models to fill-in-the-middle (FIM), we suggest that future autoregressive language models be trained with FIM by default. To this end, we run a series of ablations on key hyperparameters, such as the data transformation frequency, the structure of the transformation, and the method of selecting the infill span. We use these ablations to prescribe strong default settings and best practices to train FIM models. We have released our best infilling model trained with best practices in our API, and release our infilling benchmarks to aid future research. 

**Link** - https://arxiv.org/abs/2207.14255
</p></details>

## Text-To-Speech Synthesis

## Text-based-Games

## Machine Translation

## Question Answering

### Question Answering

### Community Question Answering

Community question answering is the task of answering questions on a Q&A forum or board, such as Stack Overflow or Quora.

### Conversational Question Answering

<details><summary>Learning Dialogue Representations from Consecutive Utterances</summary><p>

**Abstract** - Learning high-quality dialogue representations is essential for solving a variety of dialogue-oriented tasks, especially considering that dialogue systems often suffer from data scarcity. In this paper, we introduce Dialogue Sentence Embedding (DSE), a self-supervised contrastive learning method that learns effective dialogue representations suitable for a wide range of dialogue tasks. DSE learns from dialogues by taking consecutive utterances of the same dialogue as positive pairs for contrastive learning. Despite its simplicity, DSE achieves significantly better representation capability than other dialogue representation and universal sentence representation models. We evaluate DSE on five downstream dialogue tasks that examine dialogue representation at different semantic granularities. Experiments in few-shot and zero-shot settings show that DSE outperforms baselines by a large margin. For example, it achieves 13% average performance improvement over the strongest unsupervised baseline in 1-shot intent classification on 6 datasets. We also provide analyses on the benefits and limitations of our model. 

**Link** - https://paperswithcode.com/paper/learning-dialogue-representations-from
</p></details>

<details><summary>Dialog Inpainting: Turning Documents into Dialogs</summary><p>

**Abstract** - Many important questions (e.g. "How to eat healthier?") require conversation to establish context and explore in depth. However, conversational question answering (ConvQA) systems have long been stymied by scarce training data that is expensive to collect. To address this problem, we propose a new technique for synthetically generating diverse and high-quality dialog data: dialog inpainting. Our approach takes the text of any document and transforms it into a two-person dialog between the writer and an imagined reader: we treat sentences from the article as utterances spoken by the writer, and then use a dialog inpainter to predict what the imagined reader asked or said in between each of the writer's utterances. By applying this approach to passages from Wikipedia and the web, we produce WikiDialog and WebDialog, two datasets totalling 19 million diverse information-seeking dialogs -- 1,000x larger than the largest existing ConvQA dataset. Furthermore, human raters judge the answer adequacy and conversationality of WikiDialog to be as good or better than existing manually-collected datasets. Using our inpainted data to pre-train ConvQA retrieval systems, we significantly advance state-of-the-art across three benchmarks (QReCC, OR-QuAC, TREC CAsT) yielding up to 40% relative gains on standard evaluation metrics. 

**Link** - https://paperswithcode.com/paper/dialog-inpainting-turning-documents-into
</p></details>

### Answer Selection

Answer Selection is the task of identifying the correct answer to a question from a pool of candidate answers. This task can be formulated as a classification or a ranking problem.

### Knowledge Base Question Answering

Knowledge Base Q&A is the task of answering questions from a knowledge base.

![](https://production-media.paperswithcode.com/thumbnails/task/task-0000000172-35d6c9b2.jpg)

### Mathematical Question Answering

Building systems that automatically answer mathematical questions.

### Cross-Lingual Question Answering

<details><summary>ByT5: Towards a token-free future with pre-trained byte-to-byte models </summary><p>

**Abstract** - Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units. By comparison, token-free models that operate directly on raw text (bytes or characters) have many benefits: they can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with minimal modifications to process byte sequences. We characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our experiments. 

**Link** - https://paperswithcode.com/paper/byt5-towards-a-token-free-future-with-pre
</p></details>

### Logical Reasoning Question Answering

Introduced by ReClor (ICLR 2020), logical reasoning is to evaluate the logical reasoning ability of models for question answering.

## Question Generation

## Visual Question Answering



# Others
## Active Learning



<details><summary>XXXXX</summary><p>

**Abstract** - XXXXX

**Link** - XXXXX
</p></details>
