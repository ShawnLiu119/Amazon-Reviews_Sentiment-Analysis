# Sentiment Analysis - Amazon Review

**Data**

The data used for this analysis is a set of product reviews on the Mr. Coffee 12 Cup Programmable Coffee Maker.The data collected spans from 2017 to present, with 6,789 reviews and 7,221 global ratings. Each review is received from a customer that has purchased the product previously who gives their opinion on the product for others to review. Each rating is based on a 5-star scale, resulting in all the ratings to be ranged from 1-star to 5-star with no half or quarter stars. The fields include Profile Name, Review, Datetime, and Rating as shown in the table below.

![image](https://user-images.githubusercontent.com/43327902/147958242-1cb6bff0-bc28-4c1f-b0a2-7f392c236547.png)


**Model**


4.1 Data pre-processing
Figure 7 Research Methodology Flowchart
Text preprocessing is the first but most important step that cleans up and transforms the raw textual data into the format ready for following feature extraction. As shown in Figure. 4, reviews that are mostly made up with a mix of capital and lowercase were converted to lowercase; for example, “Coffee” and “Maker” are altered to “coffee” and “maker”. All punctuation and stop words that generally appear frequently but carry less meaningful information were removed. Each sentence of reviews was split and reshaped into a
8
 sequence of words called “tokens” through lemmatization. The star ratings were encoded and mapped to
 binary sentiment categories as “positive” and “negative”. The processed dataset is split by 70 : 30 ratio into training dataset for model learning and testing data for performance evaluation.
 4.2 Feature extraction
To help computers to understand and learn from the textural data, the textual data needs to be converted to a numerical vector. The two libraries used in this project are TfidfVectorizer from scikit-learn and
 customized vectorizer based on spaCy. TfidfVectorizer is feature extraction toolkit that leverages Term Frequency - Inverse Document Frequency technique. It is used as one of basic building blocks for the NLP
 pipeline and delivers better performance as it takes the importance of words in a document in to account. spaCy is an open-source NLP library with advanced built-in functionalities such as Part-Of-Speech (POS)
 Data Pre-processing Steps
 Figure 8 Data Preprocessing Flowchart
Tagging, Dependency Parsing, and Entity Detection. It uses the latest algorithms and has some advantages in word tokenization and POS-tagging.
Grid Search is a method that exhaustively testifies various combinations of hyperparameters and identifies
 the optimal one or combination for the model training. The range of n-values for different word (n-grams) has a direct effect on the feature extraction consequently impacting how well the model learns from textual data. The approach is primarily applied on “ngram_range” as the key parameters that defines the lower and upper boundary of the range of n-values for different word n-grams or char n-grams in TfidfVectorizer.

9
Five alternativeswere experimentedincluding((1,1),(1,2),(2,2),(1,3),(3,3))and(1,2)was
returned as most optimal parameters with over 95% accuracy. The option (1,2) was then used in TfidfVectorizer for feature extraction across all models.
4.3 Model design and implementation
Features that were extracted from review textual data were then fed into models that classify data into binary sentiment categories (e.g., “positive” and “negative”). In this study, five model scenarios were designed
and constructed with variant NLP algorithms for feature extraction and classification algorithms including Logistic Regression (LR), Random Forest (RF), Convolutional Neural Network (CNN) and Long short-
term memory (LSTM).
A random forest is an ensemble learning method that fits bunch of decision trees on various sub-samples of the datasets and leverage averaging to balance out over-fitting problems and improve model
performance. Logistic regression is a statistical model which basically uses a logistic function to estimate the possibility that the dependent variable falls into a certain category based on one or more independent
variables. Both Convolutional Neural Network (CNN) and Long short-term memory (LSTM) are members of the artificial neural network family, which are constructed with multilayer perceptrons but have slightly different layers in between. After learning from training datasets, classifiers are tasked to predict the
sentiment orientation of the testing dataset.
Table 2. Model Design
Model #
Vectorizer
Classification
1 TfidfV ect orizer
2 TfidfV ect orizer
3 Keras Tokenizer and Pad_sequence
4 Keras Tokenizer and Pad_sequence
5 Self-defin ed vect orizer bu ilt u pon spaCy
4.4 Performance evaluation parameters
Random Forest Logistic Regression LSTM
CNN
Random Forest
