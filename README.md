# Capstone 1 Mathias Stensrud
## Summary
  My project was to analyze data from the CDC Youth Risk Behavior Survey, and look for differences between LGBT and non LGBT youth drinking rates.

  My hypothesis was that LGBT youth would be at a greater risk for underage drinking or usage of drugs, potentially due to various social factors. I chose this due to my own observations in high school and college, and my involvement with various LGBT organizations.

#### Data
My data was pulled from the CDC YRBS, a survey given to middle- and high-schools across America. The specific data I pulled was from the 2015 survey, seeing as it had ~3,000 more usable responses than the 2017 compiled survey. I had started with the 2017 survey, but decided to expand my data set and in doing so realized I could use the 2017 version. This data was in an ASCII format, with every line of text representing a single students various answers. The survey pulled from randomly selected schools from across the country, attempting to ensure a more representative sample, though there are errors that could of course come up, as I do not know what restrictions they put on their data selection.

 #### EDA
 I was forced to remove about ~4,000 rows of my data due to missing answers. Imputing this data would not make sense as the answers were categorical and individual to each student, making them risky to change. Once I removed this data I was left with ~13,200 rows of answers that could be used in my analysis.

  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/img/drinking.png)
  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/img/q_rates.png)
  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/img/legend.png)

 My initial EDA found that close to half of LGBT students surveyed said they had drunk alcohol in the past 30 days, as compared to only 15% of their straight peers.
  The four categories that students identified as were Straight, Bisexual, Gay/Lesbian(which I later separated further), and Questioning. All of the groups apart from the straight students had over 40% of their population drinking at least 1-2 times in the previous 30 days, as well as more than 10% of Gay and Bi students drinking 3-5 times in the last 30 days. Questioning students had the highest proportion of their population engage in underage drinking, as well as the highest proportion involved in everyday drinking.

#### Model Training
  I chose to use a Logistic regression model for my data. This was technically a LogisticRegressionCV model with the binary conditions of having drank in the past 30 days or not. My features were age, sexuality, and gender. The logistic regression model managed to reach 70% accuracy based on these conditions alone, which could be raised an additional 3-5% by including data about cigarette smoking. This does seem to provide a correlation between sexuality and drinking, especially since accuracy only dropped by a few percent when gender and age were removed. Given the amount of accuracy that could be achieved with only a few conditions, I feel this is a good point to reach with this project.
  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/img/ROC.png)

#### Results
  With an average accuracy score of 68% using only sexuality, age, and gender, as well as a r2 of .55, I can say that this is not the most accurate model in the world. However, this model does showcase a trend of underage drinking in LGBT youth. The fact that bisexual and questioning students had the highest drinking rates as compared to their straight peers is concerning but not unexpected. There have been multiple studies showcasing the fact that bisexual youth have some of the highest depression rates among LGBT youth, and underage substance usage is a not unknown condition of that. Adding answers surveying mental health from the survey increased accuracy to 73%, but this was only a quick check and not an in depth addition to the model.

#### Future work
  In the future I would like to both look at the same survey from other years, as well as look at the combined YRBS dataset, which was not in a format I was able to properly read as opposed to the versions I used.
#### References
https://www.cdc.gov/healthyyouth/data/yrbs/data.htm

https://www.lgbtdata.com/data.html
