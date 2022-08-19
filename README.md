# Sentiment Analysis - Amazon Review

##Introduction
Making buy decisions has become supplemented with other consumers’ opinions, resulting in a tedious and
time-consuming task. As one of the largest ecommerce platforms in the world, holding 50% market share,
Amazon’s customer-generated content such as star ratings (1-5) and reviews continue to support other users
purchase decisions. These data points have been proved to be valuable for both customers to gauge how
others felt about the product after purchase and make more informative decisions as well as for sellers or
merchants to assess their customer satisfaction and understand the underlying factors. However, these
reviews tend to be longer and contain text which customers must read and digest to understand the
sentiment. Many products on amazon have accrued a plethora of reviews which becomes very excessive to
sift through and use for purchase decisions.

![image](https://user-images.githubusercontent.com/43327902/185546075-39f7dfd8-bbec-49e6-ad73-d4885a3434ec.png)

Figure 1: Amazon Rating System

Thanks to the tremendous advance of computing capacity and progress of Natural Language Processing fueled by machine learning and artificial intelligence, customers’ sentiments can be analyzed and interpreted by a trained ML model in a more automatic manner and on a larger scale. Sentiment Analysis is one of the major tasks of Natural Language Processing which is one of the most common classification tools used to analyze text and automatically classify the sentiment expressed and has obtained much attention in recent years (Zunic, Anastazia et al. 2020).
In this research, we aim to tackle the problem by developing a model to scrape online text reviews from Amazon, extract features through NLP algorithms such as Word2Vec and TFIDF, and conduct various machine learning algorithms for classification, including traditional ML algorithms such as Logistic Classification and cutting-edge algorithms of neural networks such as CNNs and RNNs. One of the model outputs we expect to generate is to segregate different factors, such as product quality, price, packaging, durability, and design based on word frequency and understand how each is correlated or impacts the positivity / negativity of reviews.

##Data

The data used for this analysis is a set of product reviews on the Mr. Coffee 12 Cup Programmable Coffee Maker (product review page link).

![image](https://user-images.githubusercontent.com/43327902/185546130-5ae69eca-d34e-40d0-acef-68fc1556f570.png)

Figure 2 Amazon Product Review Page - Coffee Maker

The data collected spans from 2013 to present, with 6,789 reviews and 7,221 global ratings. Each review is received from a customer that has purchased the product previously and provided their feedback on the product. Each rating ranges from 1 to 5 star with 5 stars representing the best. The attributes extracted from reviews include Profile Name, Review, Datetime, and Rating as described in the table below.
The raw date was scraped through ParseHub, a specialized web scrapping tool, into CSV file and then loaded into Python script for further exploration and manipulation.

![image](https://user-images.githubusercontent.com/43327902/185546211-470becb2-cbf0-41a5-91fb-d95852616f34.png)

Figure 3 Review Dataset Snapshot

##Data exploration and engineering
Exploratory Data Analysis (EDA) was conducted to visualize and enable general understanding of the review data on both label and input side. For “Rating” data, a discrete histogram was generated. After data cleaning and preprocessing, there are 4,990 samples remaining. As shown in Figure. 4, the data is skewed towards star rating 1 and 5 indicating a strong sentiment separation. To simplify the data for the following binary sentiment classification, the star ratings were encoded to represent positive or negative sentiment with ratings 1-3 encoded as “Negative” and 4-5 as “Positive”. The class distribution in Figure 5 illustrates that 3,028 positive reviews versus 1,962 negative ones with improved balance between sample size of the two categories, mitigating the risk that the classification model is skewed by one class data.

![image](https://user-images.githubusercontent.com/43327902/185546284-fd6cf530-1bb4-41b6-aca7-806798c4974a.png)

Figure 4 Distribution of Review by Star Rating

![image](https://user-images.githubusercontent.com/43327902/185546322-b0bf52ac-d0c3-4272-bb6e-097c53eaf2eb.png)

Figure 5 Distribution of Reviews by Sentiment

A deep dive into review text data was also conducted specifically on the range of n-values for different word (n-grams) to explore how exactly these word combinations convey underlying sentiment from syntax and semantic perspectives as reference for hyper parameter tunning in the training phase. Unigram, bigram and trigram were investigated for positive and negative reviews separately with TF-IDF matrix generated as shown in Figure 6.


![image](https://user-images.githubusercontent.com/43327902/185546351-417ad4df-4470-45b5-b622-3f8eb3ca1d5d.png)
![image](https://user-images.githubusercontent.com/43327902/185546396-e5df9b2a-ea6e-424a-a492-3c9900fbc142.png)

Figure 6 Screenshots of TF-IDF Matrix


##Method
This section provides an overview of the proposed methodology of sentiment analysis for reviews of Mr. Coffee machine. Figure. 3 depict the overall process with key phases laid out starting from data collection to model result evaluation and interpretation.

![image](https://user-images.githubusercontent.com/43327902/185546479-eebac7da-714b-405d-a60d-ec8e9f6d43f3.png)

Figure 7 Research Methodology Flowchart

Text preprocessing is the first but most important step that cleans up and transforms the raw textual data into the format ready for following feature extraction. As shown in Figure. 4, reviews that are mostly made up with a mix of capital and lowercase were converted to lowercase; for example, “Coffee” and “Maker” are altered to “coffee” and “maker”. All punctuation and stop words that generally appear frequently but carry less meaningful information were removed. Each sentence of reviews was split and reshaped into a

##**Model**
#### Data pre-processing
Basically, I used multiple tokenizer(including TFIDF and spaCy) to extract features from the cleaned data and feed the processed data into ML and deep learning classification models (including Logistic Classification, RandomForest, CNN and RNN). Here is the general process

![image](https://user-images.githubusercontent.com/43327902/147958661-c7a19ed1-2266-4cb8-95e0-1d0744e8dc45.png)

sequence of words called “tokens” through lemmatization. The star ratings were encoded and mapped to binary sentiment categories as “positive” and “negative”. The processed dataset is split by 70 : 30 ratio into training dataset for model learning and testing data for performance evaluation.

#### Feature extraction
To help computers to understand and learn from the textural data, the textual data needs to be converted to a numerical vector. The two libraries used in this project are TfidfVectorizer from scikit-learn and customized vectorizer based on spaCy. TfidfVectorizer is feature extraction toolkit that leverages Term Frequency - Inverse Document Frequency technique. It is used as one of basic building blocks for the NLP pipeline and delivers better performance as it takes the importance of words in a document into account. spaCy is an open-source NLP library with advanced built-in functionalities such as Part-Of-Speech (POS)

![image](https://user-images.githubusercontent.com/43327902/185546613-d701595e-927e-4c29-97bf-e6ad57b43963.png)

Figure 8 Data Preprocessing Flowchart

Tagging, Dependency Parsing, and Entity Detection. It uses the latest algorithms and has some advantages in word tokenization and POS-tagging. Grid Search is a method that exhaustively testifies various combinations of hyperparameters and identifies the optimal one or combination for the model training. The range of n-values for different word (n-grams) has a direct effect on the feature extraction consequently impacting how well the model learns from textual data. The approach is primarily applied on “ngram_range” as the key parameters that defines the lower and upper boundary of the range of n-values for different word n-grams or char n-grams in TfidfVectorizer.Five alternatives were experimented including ((1,1), (1,2), (2,2), (1,3), (3,3)) and (1,2) was returned as most optimal parameters with over 95% accuracy. The option (1,2) was then used in TfidfVectorizer for feature extraction across all models.

#### Model design and implementation
Features that were extracted from review textual data were then fed into models that classify data into binary sentiment categories (e.g., “positive” and “negative”). In this study, five model scenarios were designed and constructed with variant NLP algorithms for feature extraction and classification algorithms including Logistic Regression (LR), Random Forest (RF), Convolutional Neural Network (CNN) and Long short-term memory (LSTM). A random forest is an ensemble learning method that fits bunch of decision trees on various sub-samples of the datasets and leverage averaging to balance out over-fitting problems and improve model performance. Logistic regression is a statistical model which basically uses a logistic function to estimate the possibility that the dependent variable falls into a certain category based on one or more independent variables. Both Convolutional Neural Network (CNN) and Long short-term memory (LSTM) are members of the artificial neural network family, which are constructed with multilayer perceptrons but have slightly different layers in between. After learning from training datasets, classifiers are tasked to predict the sentiment orientation of the testing dataset.

![image](https://user-images.githubusercontent.com/43327902/185546692-632cf56b-4cf3-4c93-862d-418dea69ee41.png)

#### Performance evaluation parameters

The performance of models is evaluated in terms of Accuracy, Recall, and Precision as key metrics that are especially useful to assess the performance of supervised learning algorithms. To better visualize of the performance of each model, a confusion matrix is generated and plotted, which lays out classification metrics such as “True Positive (TP)”, “False Positive (FP)”, “True Negative (TF)”, and “False Negative (FN)” in a matrix format as shown below. True Positive and True Negative stands for positive reviews that are correctly classified to their sentiment categories. False Positive represents actual positive reviews that are mis-classified as negative while False Negative suggests the opposite.

![image](https://user-images.githubusercontent.com/43327902/185546746-286283f1-320a-4660-9c86-3e932ed34c7a.png)

Figure 9 Confusion Matrix

Accuracy is the ratio of reviews that correctly mapped to their sentiment category out of the total number of reviews. Recall is the ratio equals the number of reviews that are classified correctly as positive divided by total number of reviews classified positively. Precision is defined as the ratio of the number of reviews correctly assigned as positive to the total number of actual positive reviews. The formulas of each of the metrics are listed in the table below.

![image](https://user-images.githubusercontent.com/43327902/185546776-dcabd534-bdf2-4cd4-a39f-4e216b6bb29a.png)


## Results
The prediction performance on testing dataset by each model is evaluated in accuracy, precision and recall as shown in Table 4. The Random Forest model using TfidfVectorizer (Model 1) outperforms other models with highest accuracy 86.24% and higher Recall 93.67%. The LSTM model (Model 3) performs slight better on FP identification with 87.70% Precision. Logistic Classification model also delivers a good performance with second highest accuracy 85.37%. The Convolutional Neural Network (CNN) model that was constructed with deep stack of layers delivered lowest accuracy 68.27% out of the five models.

![image](https://user-images.githubusercontent.com/43327902/185546836-5377d8d7-f8f5-48b3-b7e2-f85ac3b74ca5.png)

![image](https://user-images.githubusercontent.com/43327902/185546848-656a2333-63bc-4e9e-aa0f-435c159cc10d.png)

Figure 10 Model Performance Comparison by Metrics

The Random Forest model using TfidfVectorizer (Model 1) performs better than the Random Forest model using self-defined vectorizer based on spaCy (Model 5) with 81.10% accuracy driven by higher degrees of both Precision and Recall, suggesting a slightly better capability of detecting FP and FN.

## Analysis and interpretation 
A post-training investigation was conducted to revisit and dive deep into the misclassified instances. One of useful findings is that many misclassified reviews have both positive and negative sentiment elements in their textual data, which apparently confused the training model. Figure 11 shows an example of False Positive (FP). The words such as “does not work” “return” express negative sentiment, but the words such as “liked” provides some positive color and might mislead the model as “like” was identified as positive token with higher weight.

## Conclusion
Sentiment analysis using Natural Language Processing (NLP) and Machine Learning (ML) provides an approach to extract insight from massive textual data of customer reviews in higher efficiency and larger scale. It enables a broader range of applications including identifying leading factors that impact customer decision making and enabling predictive capability based on textual data of customer feedback. In this study, binary classification on sentiment of Amazon product reviews is experimented using multiple supervised learning algorithms including Random Forest, Logistic Regression, LSTM, CNN with different feature extraction approach. Random Forest classification model delivers the best performance with above 86% accuracy followed by Logistic Regression model with 85% accuracy using NLTK-based approach for feature extraction. The traditional Machine Learning algorithms (Random Forest, Logistic Regression) outperforms neural network models (LSTM, CNN), suggesting potential areas for further study and improvement. The classification model using NLTK-based feature extraction performs slightly better than the model using same classified and customized vectorizer built upon spaCy.
