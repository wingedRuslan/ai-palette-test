# Sentiment Analysis of Thai language reviews

**Goal:** Implement a classification model that can identify the rating [1-5] from the review.    

**Dataset:** restaurant domain data   
*Training Dataset.* Each row in training data corresponding to a record contains a review along with a rating score that ranges from 1 to 5 stars. It contains close to 40,000 reviews.   
*Test Dataset* consists of a review ID and a review. We need to predict the rating for the review. The test data has close to 6,000 reviews.   

**Requirements:**

- Final submission data csv file should have 2 columns - ID (corresponding to review intest data) and its predicted rating.
- A rating prediction model using only textual information.
- Pre-trained weights are ok but the prediction model has to be your own. A proper justification of the techniques used is required.

**Evaluation:** For measuring the output quality, Mean F1 score will be used. The F1 score measures accuracy using the statistics precision p and recall r. Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn). The F1 metric weights recall and precision equally, and a good algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.     

## Project implementation
 
### Ideation Phase

General Sentiment Analysis Pipeline:  
- Data Acquisition (Removing duplicates and None values in dataset)
- Data Pre-processing (Cleaning and preprocessing dataset, tokenization)
- Feature Extraction* (Numerical Representation of the input text)
- Classifier (classification task)

Each step in this pipeline is important and contributes towards achieving high accuracy in the text classification task. However, the key factor boosting performance is the Feature Extraction part. The numerical representation of the input text helps classifier to better distinguish between classes and thus make better decisions.    

Feature extraction could be:
- as simple as the Bag-of-Words model (text is represented as the bag of its words, disregarding grammar and even word order).
- word embeddings (vector representation of words in the text). Depending on the context,words could have different meanings. This aspect is not captured by word embeddings.
- any other language representations, e.g. contextual representations by BERT or otherlanguage models

First of all, I approached the current Sentiment Analysis problem by reviewing the papers-with-code page for the state-of-the-art methods/models applied in this NLP subfield (Link to papers-with-code). As well as in other NLP tasks, using pre trained language models in the Text Classification task leads to the best performance on different datasets. Their success could be explained by the fact that these models provide rich language representation of the input text capturing semantics and meaning precisely.    

Hence, Approach 1 to solve the current problem would be the following:
1) Find a pre-trained language model
2) Fine-tune on our dataset and task
3) Use fine-tuned model for predictions (inference time)  


However, there are a few limitations related to our use-case. Since the dataset is mainly in the Thai language, some pre-trained language models can not be used, as they are restricted to certain languages. There is also a lack of pre-trained models in the Thai language. In addition, text preprocessing and tokenization are specific to the Thai language. Finally, the text classification into 5 classes is more challenging than binary classification (negative/positive labels) or into 3 classes (negative/neutral/positive).    

To address the above-mentioned limitations, I have come up with the following ideas:
**Approach 2** - using pre-trained multilingual language models (that support the Thai language) to obtain a numerical representation of input text. For example, universal-sentence-encoder-multilingual by Google (link) takes as input variable-length text in the Thai language (and many more) and gives a 512-dimensional vector as an output. We can use this vector as input to the classifier (e.g. SVM, NN with softmax) to predict the ranting of a review.      

Another example is to use LaBSE (link) - language-agnostic BERT Sentence Encoder (LaBSE) to get sentence embedding of reviews in Thai, which will be further used as input to classifier.    


**Approach 3** - translating Thai reviews into English and then leveraging a wide-variety of pre-trained language models for the Sentiment Analysis Task.   

Translation from Thai to English could be achieved by using a transformer model from HuggigFace - Helsinki-NLP/opus-mt-th-en (link). Without using any third-party APIs, this model provides a good enough translation quality. After transitioning to the English language text, we are no longer limited in terms of pre-trained models and thus we can choose from many pre-trained models available at HuggingFace. For example, I could fine-tune the T5 model in the sentiment analysis setting and use the fine-tuned model in the inference time to predict new samples.        


*Note:* I did not go with this approach as translating 40k training samples from Thai to English would take 33 hours if the time to translate one sample = 3 seconds. Being limited by the time constraints and computational power available, this approach is not feasible right now. In addition, predicting a rating for a new review at a test time would take more time, since firstly we would need to translate the review to English and then predict rating by applying fine-tuned T5 model. 

**Approach 4.** - training a model from scratch.        

Due to computational power limitations, I could not train transformer-based language models from scratch. Therefore, I chose to train a Bidirectional LSTM model for this task. For sequences other than time series (e.g. text), it is often the case that a RNN model can perform better if it not only processes sequence from start to end but also backward. For example, to classify the text, it is often useful to have the context around each individual word, not only just the words that come before it.    


LSTM (alternatively GRU) - better captures long-range dependencies in the input text, helps with vanishing gradient problem. LSTM is more powerful than GRU. Since GRU is a simpler model, it is easier (faster) to build bigger networks with its units.     