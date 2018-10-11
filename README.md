# Capstone 1 Mathias Stensrud
## Summary
  My project was to analyze data from the CDC Youth Risk Behavior Survey, and look for differences between LGBT youth's underage drinking, and their non LGBT peers.

  My hypothesis was that LGBT youth would be at a greater risk for underage drinking or usage of drugs, potentially due to various social factors. I chose this due to my own observations in high school and college, and my involvement with various LGBT organizations.
  
### Data
My data was pulled from the CDC YRBS, a survey given to middle- and high-schools across America. The specific data I pulled was from the 2015 survey, seeing as it had ~3,000 more usable responses than the 2017 compiled survey. This data was in an ASCII format, with every line of text representing a single students various answers.

 #### EDA
 I was forced to remove about ~4,000 rows of my data due to missing answers. Imputing this data would not make sense as the answers were categorical and individual to each student, making them risky to change. Once I removed this data I was left with ~13,200 rows of answers that could be used in my analysis.
 
  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/drinking.png)
  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/q_rates.png)
  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/legend.png)
 
 My initial EDA found that close to half of LGBT students surveyed said they had drunk alcohol in the past 30 days, as compared to only 15% of their straight peers.
  The four categories that students identified as were Straight, Bisexual, Gay/Lesbian(which I later separated further), and questioning. All of the groups apart from the straight students had over 40% of their population drinking at least 1-2 times in the previous 30 days, as well as more than 10% of Gay and Bi students drinking 3-5 times in the last 30 days.

#### Model Training
  I chose to use a Logistic regression model to model my data, with the conditions of either participating in underage drinking or not. My features were age, sexuality, and gender, and my model, with cross validation, managed to reach 75% accuracy based on the conditions alone, which could be raised easily by including data about cigarette smoking.

  ![](https://github.com/MathiasStensrud/capstone_1/blob/master/ROC.png)
