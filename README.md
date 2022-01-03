# Sentiment Analysis - Amazon Review

**Data**
The data used for this analysis is a set of product reviews on the Mr. Coffee 12 Cup Programmable Coffee Maker (product review page link). ![image](https://user-images.githubusercontent.com/43327902/147957742-be21c0b4-7801-4ae6-a586-e9096486986b.png)
The data collected spans from 2017 to present, with 6,789 reviews and 7,221 global ratings. Each review is received from a customer that has purchased the product previously who gives their opinion on the product for others to review. Each rating is based on a 5-star scale, resulting in all the ratings to be ranged from 1-star to 5-star with no half or quarter stars. The fields include Profile Name, Review, Datetime, and Rating as shown in the table below.
Table 1. Amazon Review Data Description
Features	Description	Role	Data Type
Profile Name	User profile name		Object
Rating	Star ratings [1-5]	Transformed to binary sentiment category as label	Float
Datetime	Date time that review was posted		Datetime
Review	Customer feedbacks on the product	Textual data from which features are extracted as training input	Object
![image](https://user-images.githubusercontent.com/43327902/147957822-a9937364-a9bf-4fa4-b35a-f94ecb480594.png)


**Model**
