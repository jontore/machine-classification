## Fertility Data set
http://archive.ics.uci.edu/ml/datasets/Fertility
Fertility of men, 100 samples, already scaled.

- Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1)
- Age at the time of analysis. 18-36 (0, 1)
- Childish diseases (ie , chicken pox, measles, mumps, polio) 1) yes, 2) no. (0, 1)
- Accident or serious trauma 1) yes, 2) no. (0, 1)
- Surgical intervention 1) yes, 2) no. (0, 1)
- High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1)
- Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1)
- Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1)
- Number of hours spent sitting per day ene-16 (0, 1)
- Output: Diagnosis normal (N), altered (O)


### Feature selection
Using SelectKBest to find the best features, and evaluating them.
Also tried other such as SelectFwe which selects based on the family wise error rate however there is little or no difference, also SelectFpr selects based on false positive rate, and SelectFdr based on false positive rate. So simply select based on the highest score is sufficient.

### Important features
The two most important features identified by SelectKBest are alcohol and age.

### Prediction
Used different classifiers even though the score is good, however plotting the prediction reveals that the estimations are not very good. Predicting that everyone is fertile gives a relative high score since there are only 9% infertile in the data set, so if everyone is fertile the accuracy is 90%.  
