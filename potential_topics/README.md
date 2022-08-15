# Potential Topics

List of Potential Topics to consider for the Thesis.

The below content is largely sourced from paperswithcode.com.

Click on the arrow drop-downs to open the collapsed sections.

# NLP
## Text Generation

### Text Generation

Text generation is the task of generating text with the goal of appearing indistinguishable to human-written text. This task if more formally known as "natural language generation" in the literature.

Text generation can be addressed with Markov processes or deep generative models like LSTMs. Recently, some of the most advanced methods for text generation include BART, GPT and other GAN-based approaches. Text generation systems are evaluated either through human ratings or automatic evaluation metrics like METEOR, ROUGE, and BLEU.

Further readings:
- [The survey: Text generation models in deep learning](https://www.sciencedirect.com/science/article/pii/S1319157820303360)
- [Modern Methods for Text Generation](https://arxiv.org/abs/2009.04968)

### Dialogue Generation

Dialogue generation is the task of "understanding" natural language inputs - within natural language processing in order to produce output. The systems are usually intended for conversing with humans, for instance back and forth dialogue with a conversation agent like a chatbot. Some example benchmarks for this task (see others such as Natural Language Understanding) include FusedChat and Ubuntu DIalogue Corpus (UDC). Models can be evaluated via metrics such as BLEU, ROUGE, and METEOR albeit with challenges in terms of weak correlation with human judgement, that may be addressed by new ones like UnSupervised and Reference-free (USR) and Metric for automatic Unreferenced dialog evaluation (MaUde).

![](https://production-media.paperswithcode.com/thumbnails/task/task-0000000208-e1ec79a9_KqtqUPC.jpg)

### Data-to-Text Generation

A classic problem in natural-language generation (NLG) involves taking structured data, such as a table, as input, and producing text that adequately and fluently describes this data as output. Unlike machine translation, which aims for complete transduction of the sentence to be translated, this form of NLG is usually taken to require addressing (at least) two separate challenges: what to say, the selection of an appropriate subset of the input data to discuss, and how to say it, the surface realization of a generation. 

### Multi-Document Summarization

Multi-Document Summarization is a process of representing a set of documents with a short piece of text by capturing the relevant information and filtering out the redundant information. Two prominent approaches to Multi-Document Summarization are extractive and abstractive summarization. Extractive summarization systems aim to extract salient snippets, sentences or passages from documents, while abstractive summarization systems aim to concisely paraphrase the content of the documents.

### Text Style Transfer

Text Style Transfer is the task of controlling certain attributes of generated text. The state-of-the-art methods can be categorized into two main types which are used on parallel and non-parallel data. Methods on parallel data are typically supervised methods that use a neural sequence-to-sequence model with the encoder-decoder architecture. Methods on non-parallel data are usually unsupervised approaches using Disentanglement, Prototype Editing and Pseudo-Parallel Corpus Construction.

The popular benchmark for this task is the Yelp Review Dataset. Models are typically evaluated with the metrics of Sentiment Accuracy, BLEU, and PPL.

#### Papers

+ <details><summary>So Different Yet So Alike! Constrained Unsupervised Text Style Transfer </summary><p>

    **Abstract** - Automatic transfer of text between domains has become popular in recent times. One of its aims is to preserve the semantic content of text being translated from source to target domain. However, it does not explicitly maintain other attributes between the source and translated text, for e.g., text length and descriptiveness. Maintaining constraints in transfer has several downstream applications, including data augmentation and de-biasing. We introduce a method for such constrained unsupervised text style transfer by introducing two complementary losses to the generative adversarial network (GAN) family of models. Unlike the competing losses used in GANs, we introduce cooperative losses where the discriminator and the generator cooperate and reduce the same loss. The first is a contrastive loss and the second is a classification loss, aiming to regularize the latent space further and bring similar sentences across domains closer together. We demonstrate that such training retains lexical, syntactic, and domain-specific constraints between domains for multiple benchmark datasets, including ones where more than one attribute change. We show that the complementary cooperative losses improve text quality, according to both automated and human evaluation measures. 

    **Link** - https://paperswithcode.com/paper/so-different-yet-so-alike-constrained-1
    </p></details>

### Paraphrase Generation

Paraphrase Generation involves transforming a natural language sentence to a new sentence, that has the same semantic meaning but a different syntactic or lexical surface form.

#### Papers

+ <details><summary>'John ate 5 apples' != 'John ate some apples': Self-Supervised Paraphrase Quality Detection for Algebraic Word Problems </summary><p>

    **Abstract** - This paper introduces the novel task of scoring paraphrases for Algebraic Word Problems (AWP) and presents a self-supervised method for doing so. In the current online pedagogical setting, paraphrasing these problems is helpful for academicians to generate multiple syntactically diverse questions for assessments. It also helps induce variation to ensure that the student has understood the problem instead of just memorizing it or using unfair means to solve it. The current state-of-the-art paraphrase generation models often cannot effectively paraphrase word problems, losing a critical piece of information (such as numbers or units) which renders the question unsolvable. There is a need for paraphrase scoring methods in the context of AWP to enable the training of good paraphrasers. Thus, we propose ParaQD, a self-supervised paraphrase quality detection method using novel data augmentations that can learn latent representations to separate a high-quality paraphrase of an algebraic question from a poor one by a wide margin. Through extensive experimentation, we demonstrate that our method outperforms existing state-of-the-art self-supervised methods by up to 32% while also demonstrating impressive zero-shot performance. 

    **Link** - https://paperswithcode.com/paper/john-ate-5-apples-john-ate-some-apples-self
    </p></details>

+ <details><summary>Diverse Text Generation via Variational Encoder-Decoder Models with Gaussian Process Priors </summary><p>

    **Abstract** - Generating high quality texts with high diversity is important for many NLG applications, but current methods mostly focus on building deterministic models to generate higher quality texts and do not provide many options for promoting diversity. In this work, we present a novel latent structured variable model to generate high quality texts by enriching contextual representation learning of encoder-decoder models. Specifically, we introduce a stochastic function to map deterministic encoder hidden states into random context variables. The proposed stochastic function is sampled from a Gaussian process prior to (1) provide infinite number of joint Gaussian distributions of random context variables (diversity-promoting) and (2) explicitly model dependency between context variables (accurate-encoding). To address the learning challenge of Gaussian processes, we propose an efficient variational inference approach to approximate the posterior distribution of random context variables. We evaluate our method in two typical text generation tasks: paraphrase generation and text style transfer. Experimental results on benchmark datasets demonstrate that our method improves the generation quality and diversity compared with other baselines. 

    **Link** - https://paperswithcode.com/paper/diverse-text-generation-via-variational
    </p></details>

### Story Generation

Story generation is the task of automatically generating a coherent narrative, often from a set of premises or a brief summary.

#### Papers

+ <details><summary>Plot Writing From Pre-Trained Language Models</summary><p>
    
    **Abstract** - Pre-trained language models (PLMs) fail to generate long-form narrative text because they do not consider global structure. As a result, the generated  texts are often incohesive, repetitive, or lack content. Recent work in story generation reintroduced explicit content planning in the form of prompts, keywords, or semantic frames. Trained on large parallel corpora, these models can generate more logical event sequences and thus more contentful stories. However, these intermediate epresentations are often not in natural language and cannot be utilized by PLMs without fine-tuning. We propose generating story plots using offthe-shelf PLMs while maintaining the benefit of content planning to generate cohesive and contentful stories. Our proposed method, SCRATCHPLOT, first prompts a PLM to compose a content plan. Then, we generate the story’s body and ending conditioned on the content plan. Furthermore, we take a generate-and-rank approach by using additional PLMs to rank the generated (story, ending) pairs. We benchmark our method with various baselines and achieved superior results in both human and automatic evaluation 1.

    **Link** - https://paperswithcode.com/paper/plot-writing-from-pre-trained-language-models-1
    </p></details>

+ <details><summary>Efficient Training of Language Models to Fill in the Middle</summary><p>
    
    **Abstract** - We show that autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to its end. While this data augmentation has garnered much interest in recent years, we provide extensive evidence that training models with a large fraction of data transformed in this way does not harm the original left-to-right generative capability, as measured by perplexity and sampling evaluations across a wide range of scales. Given the usefulness, simplicity, and efficiency of training models to fill-in-the-middle (FIM), we suggest that future autoregressive language models be trained with FIM by default. To this end, we run a series of ablations on key hyperparameters, such as the data transformation frequency, the structure of the transformation, and the method of selecting the infill span. We use these ablations to prescribe strong default settings and best practices to train FIM models. We have released our best infilling model trained with best practices in our API, and release our infilling benchmarks to aid future research. 

    **Link** - https://arxiv.org/abs/2207.14255
    </p></details>

+ <details><summary>Contextualized Scene Imagination for Generative Commonsense Reasoning </summary><p>
    
    **Abstract** - Humans use natural language to compose common concepts from their environment into plausible, day-to-day scene descriptions. However, such generative commonsense reasoning (GCSR) skills are lacking in state-of-the-art text generation methods. Descriptive sentences about arbitrary concepts generated by neural text generation models (e.g., pre-trained text-to-text Transformers) are often grammatically fluent but may not correspond to human common sense, largely due to their lack of mechanisms to capture concept relations, to identify implicit concepts, and to perform generalizable reasoning about unseen concept compositions. In this paper, we propose an Imagine-and-Verbalize (I&V) method, which learns to imagine a relational scene knowledge graph (SKG) with relations between the input concepts, and leverage the SKG as a constraint when generating a plausible scene description. We collect and harmonize a set of knowledge resources from different domains and modalities, providing a rich auxiliary supervision signal for I&V. The experiments demonstrate the effectiveness of I&V in improving language models on both concept-to-sentence and concept-to-story generation tasks, while enabling the model to learn well from fewer task examples and generate SKGs that make common sense to human annotators. 

    **Link** - https://paperswithcode.com/paper/contextualized-scene-imagination-for-1
    </p></details>

+ <details><summary>DiscoDVT: Generating Long Text with Discourse-Aware Discrete Variational Transformer </summary><p>
    
    **Abstract** - Despite the recent advances in applying pre-trained language models to generate high-quality texts, generating long passages that maintain long-range coherence is yet challenging for these models. In this paper, we propose DiscoDVT, a discourse-aware discrete variational Transformer to tackle the incoherence issue. DiscoDVT learns a discrete variable sequence that summarizes the global structure of the text and then applies it to guide the generation process at each decoding step. To further embed discourse-aware information into the discrete latent representations, we introduce an auxiliary objective to model the discourse relations within the text. We conduct extensive experiments on two open story generation datasets and demonstrate that the latent codes learn meaningful correspondence to the discourse structures that guide the model to generate long texts with better long-range coherence. 

    **Link** - https://paperswithcode.com/paper/discodvt-generating-long-text-with-discourse
    </p></details>

+ <details><summary>A Plug-and-Play Method for Controlled Text Generation </summary><p>

    **Abstract** - Large pre-trained language models have repeatedly shown their ability to produce fluent text. Yet even when starting from a prompt, generation can continue in many plausible directions. Current decoding methods with the goal of controlling generation, e.g., to ensure specific words are included, either require additional models or fine-tuning, or work poorly when the task at hand is semantically unconstrained, e.g., story generation. In this work, we present a plug-and-play decoding method for controlled language generation that is so simple and intuitive, it can be described in a single sentence: given a topic or keyword, we add a shift to the probability distribution over our vocabulary towards semantically similar words. We show how annealing this distribution can be used to impose hard constraints on language generation, something no other plug-and-play method is currently able to do with SOTA language generators. Despite the simplicity of this approach, we see it works incredibly well in practice: decoding from GPT-2 leads to diverse and fluent sentences while guaranteeing the appearance of given guide words. We perform two user studies, revealing that (1) our method outperforms competing methods in human evaluations; and (2) forcing the guide words to appear in the generated text has no impact on the fluency of the generated text. 

    **Link** - https://paperswithcode.com/paper/a-plug-and-play-method-for-controlled-text
    </p></details>

+ <details><summary>Plug-and-Blend: A Framework for Controllable Story Generation with Blended Control Codes </summary><p>
    
    **Abstract** - Large pre-trained neural language models (LM) have very powerful text generation capabilities. However, in practice, they are hard to control for creative purposes. We describe a Plug-and-Play controllable language generation framework, Plug-and-Blend, that allows a human user to input multiple control codes (topics). In the context of automated story generation, this allows a human user loose or fine-grained control of the topics and transitions between them that will appear in the generated story, and can even allow for overlapping, blended topics. Automated evaluations show our framework, working with different generative LMs, controls the generation towards given continuous-weighted control codes while keeping the generated sentences fluent, demonstrating strong blending capability. A human participant evaluation shows that the generated stories are observably transitioning between two topics. 

    **Link** - https://paperswithcode.com/paper/plug-and-blend-a-framework-for-controllable
    </p></details>

### Spelling Correction

Spelling correction is the task of detecting and correcting spelling mistakes.

### Table-to-Text Generation

Table-to-Text Generation is to generate a description from the structured table.

### Conditional Text Generation

The task of generating text according to some pre-specified conditioning (e.g. topic or sentiment)

### Visual Storytelling

![](https://production-media.paperswithcode.com/thumbnails/task/task-0000001749-3e0b64f9.jpg)

#### Papers

+ <details><summary>Plot and Rework: Modeling Storylines for Visual Storytelling</summary><p>
    
    **Abstract** - Writing a coherent and engaging story is not easy. Creative writers use their knowledge and worldview to put disjointed elements together to form a coherent storyline, and work and rework iteratively toward perfection. Automated visual storytelling (VIST) models, however, make poor use of external knowledge and iterative generation when attempting to create stories. This paper introduces PR-VIST, a framework that represents the input image sequence as a story graph in which it finds the best path to form a storyline. PR-VIST then takes this path and learns to generate the final story via an iterative training process. This framework produces stories that are superior in terms of diversity, coherence, and humanness, per both automatic and human evaluations. An ablation study shows that both plotting and reworking contribute to the model's superiority. 

    **Link** - https://paperswithcode.com/paper/plot-and-rework-modeling-storylines-for
    </p></details>

### Text Infilling

Text Infilling is the task of predicting missing spans of text which are consistent with the preceding and subsequent text. Text Infilling is a generalization of the cloze task—cloze historically refers to infilling individual words.

### Question-Answer-Generation

### Story Completion

Given a story prefix and two possible endings, determining which one is the correct (coherent) ending of the story.

### News Generation

Generation of larger segments of text with consistent topic and evolving story.

### Concept-To-Text Generation

Generating natural language text from a conceptualized representation, such as an ontology.

### Sonnet Generation

Generating a poetry in the form of a sonnet.


## Text-To-Speech Synthesis

## Text-based-Games

## Machine Translation

### Machine Translation

Machine translation is the task of translating a sentence in a source language to a different target language.

Approaches for machine translation can range from rule-based to statistical to neural-based. More recently, encoder-decoder attention-based architectures like BERT have attained major improvements in machine translation.

One of the most popular datasets used to benchmark machine translation systems is the WMT family of datasets. Some of the most commonly used evaluation metrics for machine translation systems include BLEU, METEOR, NIST, and others. 

![](https://production-media.paperswithcode.com/thumbnails/task/task-0000000257-2b560008_M7RFnV9.jpg)

### Transliteration

Transliteration is a mechanism for converting a word in a source (foreign) language to a target language, and often adopts approaches from machine translation. In machine translation, the objective is to preserve the semantic meaning of the utterance as much as possible while following the syntactic structure in the target language. In Transliteration, the objective is to preserve the original pronunciation of the source word as much as possible while following the phonological structures of the target language.

For example, the city’s name “Manchester” has become well known by people of languages other than English. These new words are often named entities that are important in cross-lingual information retrieval, information extraction, machine translation, and often present out-of-vocabulary challenges to spoken language technologies such as automatic speech recognition, spoken keyword search, and text-to-speech.

### Unsupervised Machine Translation

Unsupervised machine translation is the task of doing machine translation without any translation resources at training time.

![](https://production-media.paperswithcode.com/thumbnails/task/task-0000001097-dc9057dd.jpg)

#### Papers

+ <details><summary>Leveraging Automated Unit Tests for Unsupervised Code Translation </summary><p>

    **Abstract** - With little to no parallel data available for programming languages, unsupervised methods are well-suited to source code translation. However, the majority of unsupervised machine translation approaches rely on back-translation, a method developed in the context of natural language translation and one that inherently involves training on noisy inputs. Unfortunately, source code is highly sensitive to small changes; a single token can result in compilation failures or erroneous programs, unlike natural languages where small inaccuracies may not change the meaning of a sentence. To address this issue, we propose to leverage an automated unit-testing system to filter out invalid translations, thereby creating a fully tested parallel corpus. We found that fine-tuning an unsupervised model with this filtered data set significantly reduces the noise in the translations so-generated, comfortably outperforming the state-of-the-art for all language pairs studied. In particular, for Java -> Python and Python -> C++ we outperform the best previous methods by more than 16% and 24% respectively, reducing the error rate by more than 35%. 

    **Link** - https://paperswithcode.com/paper/leveraging-automated-unit-tests-for-1
    </p></details>

### Automatic Post-Editing

Automatic post-editing (APE) is used to correct errors in the translation made by the machine translation systems.

### Multimodal Machine Translation

Multimodal machine translation is the task of doing machine translation with multiple data sources - for example, translating "a bird is flying over water" + an image of a bird over water to German text.

![](https://production-media.paperswithcode.com/thumbnails/task/task-0000001101-fb2e2264.jpg)

#### Papers

+ <details><summary>VISA: An Ambiguous Subtitles Dataset for Visual Scene-Aware Machine Translation </summary><p>

    **Abstract** - Existing multimodal machine translation (MMT) datasets consist of images and video captions or general subtitles, which rarely contain linguistic ambiguity, making visual information not so effective to generate appropriate translations. We introduce VISA, a new dataset that consists of 40k Japanese-English parallel sentence pairs and corresponding video clips with the following key features: (1) the parallel sentences are subtitles from movies and TV episodes; (2) the source subtitles are ambiguous, which means they have multiple possible translations with different meanings; (3) we divide the dataset into Polysemy and Omission according to the cause of ambiguity. We show that VISA is challenging for the latest MMT system, and we hope that the dataset can facilitate MMT research. The VISA dataset is available at: https://github.com/ku-nlp/VISA. 

    **Link** - https://paperswithcode.com/paper/visa-an-ambiguous-subtitles-dataset-for
    </p></details>

### Low-Resource Neural Machine Translation

Low-resource machine translation is the task of machine translation on a low-resource language where large data may not be available.

## Text Classification

### Text Classification

### Topic Models

A topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for the discovery of hidden semantic structures in a text body.

### Document Classification

### Sentence Classification

### Emotion Classification

Emotion classification, or emotion categorization, is the task of recognising emotions to classify them into the corresponding category. Given an input, classify it as 'neutral or no emotion' or as one, or more, of several given emotions that best represent the mental state of the subject's facial expression, words, and so on. Some example benchmarks include ROCStories, Many Faces of Anger (MFA), and GoEmotions. Models can be evaluated using metrics such as the Concordance Correlation Coefficient (CCC) and the Mean Squared Error (MSE).

### Multi-Label Text Classification

### Semi-Supervised Text Classification

### Coherence Evaluation

Evaluating the overall coherence of text as measured by its readability and flow through ideas.

#### Papers

+ <details><summary>Transformer Models for Text Coherence Assessment </summary><p>

    **Abstract** - Coherence is an important aspect of text quality and is crucial for ensuring its readability. It is essential desirable for outputs from text generation systems like summarization, question answering, machine translation, question generation, table-to-text, etc. An automated coherence scoring model is also helpful in essay scoring or providing writing feedback. A large body of previous work has leveraged entity-based methods, syntactic patterns, discourse relations, and more recently traditional deep learning architectures for text coherence assessment. Previous work suffers from drawbacks like the inability to handle long-range dependencies, out-of-vocabulary words, or model sequence information. We hypothesize that coherence assessment is a cognitively complex task that requires deeper models and can benefit from other related tasks. Accordingly, in this paper, we propose four different Transformer-based architectures for the task: vanilla Transformer, hierarchical Transformer, multi-task learning-based model, and a model with fact-based input representation. Our experiments with popular benchmark datasets across multiple domains on four different coherence assessment tasks demonstrate that our models achieve state-of-the-art results outperforming existing models by a good margin. 

    **Link** - https://paperswithcode.com/paper/transformer-models-for-text-coherence
    </p></details>


## Question Answering

### Question Answering

### Community Question Answering

Community question answering is the task of answering questions on a Q&A forum or board, such as Stack Overflow or Quora.

### Conversational Question Answering

#### Papers

+ <details><summary>Learning Dialogue Representations from Consecutive Utterances</summary><p>

    **Abstract** - Learning high-quality dialogue representations is essential for solving a variety of dialogue-oriented tasks, especially considering that dialogue systems often suffer from data scarcity. In this paper, we introduce Dialogue Sentence Embedding (DSE), a self-supervised contrastive learning method that learns effective dialogue representations suitable for a wide range of dialogue tasks. DSE learns from dialogues by taking consecutive utterances of the same dialogue as positive pairs for contrastive learning. Despite its simplicity, DSE achieves significantly better representation capability than other dialogue representation and universal sentence representation models. We evaluate DSE on five downstream dialogue tasks that examine dialogue representation at different semantic granularities. Experiments in few-shot and zero-shot settings show that DSE outperforms baselines by a large margin. For example, it achieves 13% average performance improvement over the strongest unsupervised baseline in 1-shot intent classification on 6 datasets. We also provide analyses on the benefits and limitations of our model. 

    **Link** - https://paperswithcode.com/paper/learning-dialogue-representations-from
    </p></details>

+ <details><summary>Dialog Inpainting: Turning Documents into Dialogs</summary><p>

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

#### Papers

+ <details><summary>ByT5: Towards a token-free future with pre-trained byte-to-byte models </summary><p>

    **Abstract** - Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units. By comparison, token-free models that operate directly on raw text (bytes or characters) have many benefits: they can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with minimal modifications to process byte sequences. We characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our experiments. 

    **Link** - https://paperswithcode.com/paper/byt5-towards-a-token-free-future-with-pre
    </p></details>

### Logical Reasoning Question Answering

Introduced by ReClor (ICLR 2020), logical reasoning is to evaluate the logical reasoning ability of models for question answering.

## Question Generation

## Visual Question Answering



# Others

## Active Learning



# Collapsible Markdown

<details><summary>XXXXX</summary><p>

**Abstract** - XXXXX

**Link** - XXXXX
</p></details>

+ <details><summary>XXXXX</summary><p>

    **Abstract** - XXXXX

    **Link** - XXXXX
    </p></details>