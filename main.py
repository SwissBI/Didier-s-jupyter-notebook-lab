## import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

import matplotlib.pyplot as plt

immodata = pd.read_csv("Immo.csv")

## have a peak at the data
print(immodata.head()) # print first 4 rows of data
plt.show()
print(immodata.info()) # get some info on variables
plt.show()

# scatter plot charges ~ age
immodata.plot.scatter(x='SURF', y='Pr')
plt.show()

# scatter plot charges ~ bmi
immodata.plot.scatter(x='ROM', y='Pr')
plt.show()

print(immodata.boxplot(column = 'Pr', by = 'ROM'))
plt.show()

print(immodata.boxplot(column = 'Pr', by = 'AH'))
plt.show()


# simple linear regression using age as a predictor
X = immodata["SURF"] ## the input variables
y = immodata["Pr"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model1 = sm.OLS(y, X).fit()
predictions = model1.predict(X) # make the predictions by the model

# Print out the statistics
model1.summary()

# view the dataset
print(model1.summary())
