# Sentiment Analysis - Amazon Review

##**Introduction**
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

##**Data**

The data used for this analysis is a set of product reviews on the Mr. Coffee 12 Cup Programmable Coffee Maker (product review page link).

![image](https://user-images.githubusercontent.com/43327902/185546130-5ae69eca-d34e-40d0-acef-68fc1556f570.png)

Figure 2 Amazon Product Review Page - Coffee Maker

The data collected spans from 2013 to present, with 6,789 reviews and 7,221 global ratings. Each review is received from a customer that has purchased the product previously and provided their feedback on the product. Each rating ranges from 1 to 5 star with 5 stars representing the best. The attributes extracted from reviews include Profile Name, Review, Datetime, and Rating as described in the table below.
The raw date was scraped through ParseHub, a specialized web scrapping tool, into CSV file and then loaded into Python script for further exploration and manipulation.

![image](https://user-images.githubusercontent.com/43327902/185546211-470becb2-cbf0-41a5-91fb-d95852616f34.png)

Figure 3 Review Dataset Snapshot

##**Data exploration and engineering**
Exploratory Data Analysis (EDA) was conducted to visualize and enable general understanding of the review data on both label and input side. For “Rating” data, a discrete histogram was generated. After data cleaning and preprocessing, there are 4,990 samples remaining. As shown in Figure. 4, the data is skewed towards star rating 1 and 5 indicating a strong sentiment separation. To simplify the data for the following binary sentiment classification, the star ratings were encoded to represent positive or negative sentiment with ratings 1-3 encoded as “Negative” and 4-5 as “Positive”. The class distribution in Figure 5 illustrates that 3,028 positive reviews versus 1,962 negative ones with improved balance between sample size of the two categories, mitigating the risk that the classification model is skewed by one class data.

![image](https://user-images.githubusercontent.com/43327902/185546284-fd6cf530-1bb4-41b6-aca7-806798c4974a.png)

Figure 4 Distribution of Review by Star Rating

![image](https://user-images.githubusercontent.com/43327902/185546322-b0bf52ac-d0c3-4272-bb6e-097c53eaf2eb.png)

Figure 5 Distribution of Reviews by Sentiment

A deep dive into review text data was also conducted specifically on the range of n-values for different word (n-grams) to explore how exactly these word combinations convey underlying sentiment from syntax and semantic perspectives as reference for hyper parameter tunning in the training phase. Unigram, bigram and trigram were investigated for positive and negative reviews separately with TF-IDF matrix generated as shown in Figure 6.


![image](https://user-images.githubusercontent.com/43327902/185546351-417ad4df-4470-45b5-b622-3f8eb3ca1d5d.png)
![image](https://user-images.githubusercontent.com/43327902/185546396-e5df9b2a-ea6e-424a-a492-3c9900fbc142.png)

Figure 6 Screenshots of TF-IDF Matrix


##**Method**


##**Model**

Basically, I used multiple tokenizer(including TFIDF and spaCy) to extract features from the cleaned data and feed the processed data into ML and deep learning classification models (including Logistic Classification, RandomForest, CNN and RNN). Here is the general process
![image](https://user-images.githubusercontent.com/43327902/147958661-c7a19ed1-2266-4cb8-95e0-1d0744e8dc45.png)



