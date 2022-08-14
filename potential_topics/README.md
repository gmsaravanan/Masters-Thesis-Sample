# List of Potential Topics to consider for the Thesis

## [Story Generation](https://paperswithcode.com/task/story-generation/latest)
- [Plot Writing From Pre-Trained Language Models](https://paperswithcode.com/paper/plot-writing-from-pre-trained-language-models-1)
    - **Abstract** - Pre-trained language models (PLMs) fail to generate long-form narrative text because they do not consider global structure. As a result, the generated  texts are often incohesive, repetitive, or lack content. Recent work in story generation reintroduced explicit content planning in the form of prompts, keywords, or semantic frames. Trained on large parallel corpora, these models can generate more logical event sequences and thus more contentful stories. However, these intermediate epresentations are often not in natural language and cannot be utilized by PLMs without fine-tuning. We propose generating story plots using offthe-shelf PLMs while maintaining the benefit of content planning to generate cohesive and contentful stories. Our proposed method, SCRATCHPLOT, first prompts a PLM to compose a content plan. Then, we generate the story’s body and ending conditioned on the content plan. Furthermore, we take a generate-and-rank approach by using additional PLMs to rank the generated (story, ending) pairs. We benchmark our method with various baselines and achieved superior results in both human and automatic evaluation 1.
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
    - **Abstract** - We show that autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to its end. While this data augmentation has garnered much interest in recent years, we provide extensive evidence that training models with a large fraction of data transformed in this way does not harm the original left-to-right generative capability, as measured by perplexity and sampling evaluations across a wide range of scales. Given the usefulness, simplicity, and efficiency of training models to fill-in-the-middle (FIM), we suggest that future autoregressive language models be trained with FIM by default. To this end, we run a series of ablations on key hyperparameters, such as the data transformation frequency, the structure of the transformation, and the method of selecting the infill span. We use these ablations to prescribe strong default settings and best practices to train FIM models. We have released our best infilling model trained with best practices in our API, and release our infilling benchmarks to aid future research. 

## [Text Generation](https://paperswithcode.com/task/text-generation)

## [Active Learning](https://paperswithcode.com/task/active-learning)

## [Text-To-Speech Synthesis](https://paperswithcode.com/task/text-to-speech-synthesis)

## [Text-based-Games](https://paperswithcode.com/task/text-based-games)

## [Machine Translation](https://paperswithcode.com/task/machine-translation)

## [Question Answering](https://paperswithcode.com/task/question-answering)

## [Question Generation](https://paperswithcode.com/task/question-generation)

## [Visual Question Answering](https://paperswithcode.com/task/visual-question-answering)