
#******************************OK*********************************

# import module
import pandas

# load the csv
data = pandas.read_csv("nba.csv")



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time..... 
fig,ax0 = plt.subplots(figsize=(5.3,4))

# loading dataset using seaborn
df = seaborn.load_dataset('tips')

# pairplot with hue sex
seaborn.pairplot(df,hue='size')
plt.show()
plt.pause(10)




#******************************OK*********************************

# importing packages
# importing packages
import seaborn
import pandas
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time..... 
fig,ax0 = plt.subplots(figsize=(5.3,4))

# load the csv
from matplotlib import pyplot as plt

data = pandas.read_csv("nba.csv")
seaborn.pairplot(data.head(), hue = 'Age')

plt.show()


#******************************OK*********************************
# Import necessary libraries
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# Generating dataset of random numbers
np.random.seed(1)
num_var = np.random.randn(1000)
num_var = pd.Series(num_var, name = "Numerical Variable")

# Plot histogram
sns.histplot(data = num_var, kde = True)


#******************************OK*********************************


# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# Load dataset
tips = sns.load_dataset("tips")

# Plot histogram
sns.histplot(data = tips, x = "size", stat = "probability", discrete = True)
plt.show()


#******************************OK*********************************

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# loading dataset
data = sns.load_dataset("tips")

# plot the swarmplot
# size set to 5
sns.swarmplot(x="day", y="total_bill",
              data=data, size=5)
plt.show()
#******************************OK*********************************

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
data = sns.load_dataset("tips")

# plot the swarmplot
# hue by size
# oriented to horizontal
sns.swarmplot(y = "day", x = "total_bill", hue = "size",
			orient = "h", data = data)
plt.show()
#******************************OK*********************************
import seaborn

import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.violinplot(x="timepoint",
                   y="signal",
                   data=fmri)

plt.show()

#******************************OK*********************************


import seaborn

import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.violinplot(x="timepoint",
                   y="signal",
                   hue="region",
                   style="event",
                   data=fmri)

plt.show()


#******************************OK*********************************
import seaborn

import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))


seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.violinplot(x='day', y='tip', data=tip)

plt.show()
#******************************OK*********************************

# importing the required library

import seaborn as sns
import matplotlib.pyplot as plt


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# read a tips.csv file from seaborn library
df = sns.load_dataset('tips')

# count plot on single categorical variable
sns.countplot(x='sex', data=df)

# Show the plot
plt.show()
#******************************OK*********************************



# importing the required library

import seaborn as sns
import matplotlib.pyplot as plt


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# read a tips.csv file from seaborn library
df = sns.load_dataset('tips')

# count plot on two categorical variable
sns.countplot(x='sex', hue="smoker", data=df)

# Show the plot
plt.show()

#******************************OK*********************************
# importing the required library

import seaborn as sns
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# read a tips.csv file from seaborn library
df = sns.load_dataset('tips')

# count plot along y axis
sns.countplot(y ='sex', hue = "smoker", data = df)

# Show the plot
plt.show()
#******************************OK*********************************

# importing the required library

import seaborn as sns
import matplotlib.pyplot as plt

# read a tips.csv file from seaborn library
df = sns.load_dataset('tips')

# use a different colour palette in count plot
sns.countplot(x ='sex', data = df, palette = "Set2")

# Show the plot
plt.show()


#******************************OK*********************************

# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# loading of a dataframe from seaborn
df = seaborn.load_dataset('tips')

############# Main Section		 #############
# Form a facetgrid using columns with a hue
graph = seaborn.FacetGrid(df, col ="sex", hue ="day")
# map the above form facetgrid with some attributes
graph.map(plt.scatter, "total_bill", "tip", edgecolor ="w").add_legend()
# show the object
plt.show()

# This code is contributed by Deepanshu Rustagi.



#******************************OK*********************************

# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# loading of a dataframe from seaborn
df = seaborn.load_dataset('tips')

############# Main Section		 #############
# Form a facetgrid using columns with a hue
graph = seaborn.FacetGrid(df, row ='smoker', col ='time')
# map the above form facetgrid with some attributes
graph.map(plt.hist, 'total_bill', bins = 15, color ='orange')
# show the object
plt.show()

# This code is contributed by Deepanshu Rustagi.







#******************************OK*********************************

# importing packages
import seaborn
import matplotlib.pyplot as plt



# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# loading of a dataframe from seaborn
df = seaborn.load_dataset('tips')

############# Main Section		 #############
# Form a facetgrid using columns with a hue
graph = seaborn.FacetGrid(df, col ='time', hue ='smoker')
# map the above form facetgrid with some attributes
graph.map(seaborn.regplot, "total_bill", "tip").add_legend()
# show the object
plt.show()

# This code is contributed by Deepanshu Rustagi.
#******************************OK*********************************

# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# loading dataset
df = seaborn.load_dataset('tips')

# PairGrid object with hue
graph = seaborn.PairGrid(df, hue ='day')
# type of graph for diagonal
graph = graph.map_diag(plt.hist)
# type of graph for non-diagonal
graph = graph.map_offdiag(plt.scatter)
# to add legends
graph = graph.add_legend()
# to show
plt.show()
# This code is contributed by Deepanshu Rusatgi.


#******************************OK*********************************



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))




seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.scatterplot(x="timepoint",
					y="signal",
					data=fmri)

# Show the plot
plt.show()

#******************************OK*********************************


# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.scatterplot(x="timepoint",
                    y="signal",
                    hue="region",
                    style="event",
                    data=fmri)
# Show the plot
plt.show()

#******************************OK*********************************


# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)
# Show the plot
plt.show()

#******************************OK*********************************


# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)

seaborn.scatterplot(x='day', y='tip', data= tip, marker = '+')

# Show the plot
plt.show()



#******************************OK*********************************



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)

seaborn.scatterplot(x='day', y='tip', data=tip, hue="time", style="time")




# Show the plot
plt.show()
#******************************OK*********************************



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)

seaborn.scatterplot(x='day', y='tip', data=tip, hue='time', palette='pastel')





# Show the plot
plt.show()

#******************************OK*********************************



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)

seaborn.scatterplot(x='day', y='tip', data=tip ,hue='size', size = "size")






# Show the plot
plt.show()

#******************************OK*********************************



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)

seaborn.scatterplot(x='day', y='tip', data=tip, hue='day',
					sizes=(30, 200), legend='brief')


# Show the plot
plt.show()

#******************************OK*********************************



# importing packages
import seaborn
import matplotlib.pyplot as plt

# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.scatterplot(x='day', y='tip', data=tip)

seaborn.scatterplot(x='day', y='tip', data=tip, alpha = 0.1)



# Show the plot
plt.show()

#******************************OK*********************************

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
dataset = pd.read_csv('AgeCSell.csv')

sns.relplot(x ="Pr", y ="SURF",
			data = dataset);


# Show the plot
plt.show()


#******************************OK*********************************
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
dataset = pd.read_csv('Immo.csv')

sns.relplot(x ="Pr", y ="SURF",
			data = dataset);

sns.relplot(x ="Pr", y ="SURF",
			hue ="AH", data = dataset);



# Show the plot
plt.show()


#******************************OK BA Arbeit*********************************


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
dataset = pd.read_csv('HomegateADsale.csv')

sns.relplot(x ="Pr", y ="SURF",
			data = dataset);

sns.relplot(x ="Pr", y ="SURF",
			hue ="STA", data = dataset);

sns.relplot(x ="Pr", y ="SURF",
			hue ="STA", style ="STA",
			data = dataset);




# Show the plot
plt.show()


#******************************OK BA Arbeit*********************************

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
df = pd.read_csv('HomegateADsale.csv')

# use regplot
sns.regplot(x = "SURF",
			y = "Pr",
			ci = None,
			data = df)




# Show the plot
plt.show()


#******************************OK BA Arbeit*********************************


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
df = pd.read_csv('HomegateADsale.csv')

# use regplot
sns.regplot(x = "SURF",
			y = "Pr",
			ci = None,
			data = df)

# plotting scatterplot with histograms for features total bill and tip.
sns.jointplot(data= df, x="SURF", y="Pr")


# Show the plot
plt.show()

#******************************OK*********************************


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
df = pd.read_csv('HomegateADsale.csv')

# use regplot
sns.regplot(x = "SURF",
			y = "Pr",
			ci = None,
			data = df)

# plotting scatterplot with histograms for features total bill and tip.
sns.jointplot(data= df, x="SURF", y="Pr",kind="reg", marker="*")


# Show the plot
plt.show()

#******************************OK*********************************


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# withouth the following line the plt closes after a short display time.....
fig,ax0 = plt.subplots(figsize=(5.3,4))

# set grid style
sns.set(style ="darkgrid")

# import dataset
df = pd.read_csv('HomegateADsale.csv')

# use regplot
sns.regplot(x = "SURF",
			y = "Pr",
			ci = None,
			data = df)


sns.jointplot(data=df, x="SURF", y="Pr", hue="STA")

# Show the plot
plt.show()

#******************************OK*********************************

import seaborn
import matplotlib.pyplot as plt


seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.boxplot(x="timepoint",
				y="signal",
				data=fmri)



plt.show()

#******************************OK*********************************
import seaborn
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.boxplot(x='day', y='tip', data=tip)



plt.show()



#******************************OK*********************************

import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tip = seaborn.load_dataset("tips")

seaborn.boxplot(x=tip['total_bill'])



plt.show()


#******************************OK*********************************
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tip = seaborn.load_dataset("tips")
seaborn.boxplot(x='tip', y='day', data=tip)



plt.show()

#******************************OK*********************************

import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

tip = seaborn.load_dataset("tips")
seaborn.boxplot(x = 'day', y = 'tip',
                data = tip,
                linewidth=2.5)



plt.show()

#******************************OK*********************************
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tip = seaborn.load_dataset("tips")
seaborn.boxplot(data = tip,orient="h")



plt.show()


#******************************OK*********************************

import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tip = seaborn.load_dataset("tips")
seaborn.boxplot(data = tip,orient="v")




plt.show()

#******************************OK*********************************

import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tip = seaborn.load_dataset("tips")
seaborn.boxplot(x='day', y='tip', data=tip, color="green")




plt.show()

#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

# Just switch x and y
sns.boxplot(y=df["species"], x=df["sepal_length"])



plt.show()


#******************************OK*********************************
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
ax = sns.boxplot(data=tips, orient="h", palette="Set2")



plt.show()


#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt

# using titanic dataset from
# seaborn library
df = sns.load_dataset("titanic")

# to see first 5 rows of dataset
print(df.head())



plt.show()



Print result: 

   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True


#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt

# using titanic dataset from
# seaborn library
df = sns.load_dataset("titanic")

# to see first 5 rows of dataset
print(df.head())

# to plot a boxplot of
# age vs survived feature
plt.figure(figsize=(10, 8))
sns.boxplot(x='survived',
			y='age',
			data=df)
plt.ylabel("Age", size=14)
plt.xlabel("Survived", size=14)
plt.title("Titanic Dataset", size=18)


plt.show()



#******************************OK*********************************


import seaborn as sns
import matplotlib.pyplot as plt

# using titanic dataset from
# seaborn library
df = sns.load_dataset("titanic")

# to see first 5 rows of dataset
print(df.head())

# boxplot with showmeans
plt.figure(figsize=(10, 8))
sns.boxplot(x='survived',
			y='age',
			data=df,
			showmeans=True) # notice the change
plt.ylabel("Age", size=14)
plt.xlabel("Survived", size=14)
plt.title("Titanic Dataset", size=18)



plt.show()
#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset
tips = sns.load_dataset('tips')

# display top most rows
tips.head()

print(tips.head())

plt.show()

Print result:

   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4



#******************************OK*********************************
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset
tips = sns.load_dataset('tips')

# display top most rows
tips.head()

print(tips.head())

# illustrate box plot
fx = sns.boxplot(x='day', y='total_bill', data=tips, hue='sex', palette='Set2')


plt.show()


#******************************OK*********************************
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load dataset
tips = sns.load_dataset('tips')

# display top most rows
tips.head()

print(tips.head())


# illustrating box plot with order
fx = sns.boxplot(x='day', y='total_bill', data=tips, order=[
				'Sun', 'Sat', 'Fri', 'Thur'], hue='sex', palette='Set2')



plt.show()


#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load the dataset
data = sns.load_dataset('tips')

# view the dataset
print(data.head(5))



plt.show()

Print result:

   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4


#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load the dataset
data = sns.load_dataset('tips')

# view the dataset
print(data.head(5))

# create grouped boxplot
sns.boxplot(x = data['day'],
			y = data['total_bill'],
			hue = data['sex'])


plt.show()


#******************************OK*********************************

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load the dataset
data = sns.load_dataset('tips')

# view the dataset
print(data.head(5))

# create another grouped boxplot
sns.boxplot(x = data['day'],
			y = data['total_bill'],
			hue = data['smoker'],
			palette = 'Set2')



plt.show()
#******************************OK*********************************
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load the dataset
data = sns.load_dataset('tips')

# view the dataset
print(data.head(5))

# create 3rd grouped boxplot
sns.boxplot(x = data['day'],
			y = data['total_bill'],
			hue = data['size'],
			palette = 'husl')




plt.show()




#******************************OK*********************************

import seaborn
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset("tips")

seaborn.stripplot(x="day", y="total_bill", data=tip)

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style = 'whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")
seaborn.stripplot(x=tips["total_bill"])

plt.show()

#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style = 'whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.stripplot(x="day", y="total_bill", data=tips, jitter=0.1)


plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style='whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.stripplot(y="total_bill", x="day", data=tips,
				  linewidth=3)

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style='whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.stripplot(y="total_bill", x="day", data=tips,
				linewidth=2,edgecolor='green')


plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style='whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.stripplot(x="sex", y="total_bill", hue="day", data=tips)

plt.show()




#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style='whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.stripplot(x="day", y="total_bill", hue="smoker",
				data=tips, palette="Set2", dodge=True)


plt.show()



#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style='whitegrid')

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.stripplot(x="day", y="total_bill", hue="smoker",
                  data=tips, palette="Set1", size=20,
                  marker="s", alpha=0.2)

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.swarmplot(x="timepoint",
				  y="signal",
				  data=fmri)

plt.show()


#******************************OK*********************************
# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")

seaborn.swarmplot(x="timepoint",
				  y="signal",
				  hue="region",
				  data=fmri)

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')
tip = seaborn.load_dataset('tips')

seaborn.swarmplot(x='day', y='tip', data=tip)

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

# view the dataset
print(tips.head(500))

seaborn.swarmplot(x=tips["total_bill"])

plt.show()


Print result:

     total_bill   tip     sex smoker   day    time  size
0         16.99  1.01  Female     No   Sun  Dinner     2
1         10.34  1.66    Male     No   Sun  Dinner     3
2         21.01  3.50    Male     No   Sun  Dinner     3
3         23.68  3.31    Male     No   Sun  Dinner     2
4         24.59  3.61  Female     No   Sun  Dinner     4
..          ...   ...     ...    ...   ...     ...   ...
239       29.03  5.92    Male     No   Sat  Dinner     3
240       27.18  2.00  Female    Yes   Sat  Dinner     2
241       22.67  2.00    Male    Yes   Sat  Dinner     2
242       17.82  1.75    Male     No   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2


#******************************OK*********************************

 Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x="total_bill", y="day", data=tips)


# view the dataset
print(tips.head(500))

plt.show()


#******************************OK*********************************
# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x="day", y="total_bill", hue="time", data=tips)


# view the dataset
print(tips.head(500))

plt.show()



#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x="day", y="total_bill", data=tips,
				  linewidth=2)


# view the dataset
print(tips.head(5))

plt.show()

#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(y="total_bill", x="day", data=tips,
				linewidth=2,edgecolor='green')


# view the dataset
print(tips.head(5))

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x="day", y="total_bill", hue="smoker",
				  data=tips, palette="Set2", dodge=True)


# view the dataset
print(tips.head(5))

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x="day", y="total_bill", hue="smoker",
				  data=tips, palette="Set2", size=20, marker="D",
				  edgecolor="gray", alpha=.25)


# view the dataset
print(tips.head(5))

plt.show()


#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x="time", y="tip", data=tips,
				  order=["Dinner", "Lunch"])


# view the dataset
print(tips.head(5))

plt.show()

#******************************OK*********************************

# Python program to illustrate
# Stripplot using inbuilt data-set
# given in seaborn

# importing the required module
import seaborn
import matplotlib.pyplot as plt

# use to set style of background of plot
seaborn.set(style="whitegrid")

# loading data-set
tips = seaborn.load_dataset("tips")

seaborn.swarmplot(x='day', y='total_bill', data=tips,
				  hue='smoker', size=10)


# view the dataset
print(tips.head(5))

plt.show()


#******************************OK*********************************







plt.show()



# importing packages
import seaborn
import pandas
import matplotlib.pyplot as plt

fig,ax0 = plt.subplots(figsize=(5.3,4))

# load the csv
data = pandas.read_csv("nba.csv")

# pairplot
seaborn.pairplot(data)
plt.show()


*************************************************
*************************************************
Multiple LinearRegression Analysis using pandas notebook
https://jupyter.org/try-jupyter/lab/

*************************************************
*************************************************
*************************************************

# Importing necessary libraries
import numpy as np
import pandas as pd

housing = pd.read_csv("Housing.csv")
housing.head()
*************************************************
# Checking for null values
print(housing.info())

# Checking for outliers
print(housing.describe())

*************************************************
# Converting the categorical variable into numerical
varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# Check the housing dataframe now
housing
*************************************************
# Creating dummy variable
status = pd.get_dummies(housing['furnishingstatus'])

# Check what the dataset 'status' looks like
status
*************************************************
# Dropping the first column from status dataset
status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)

# Adding the status to the original housing dataframe
housing = pd.concat([housing, status], axis = 1)

# Dropping 'furnishingstatus' as we have created the dummies for it
housing.drop(['furnishingstatus'], axis = 1, inplace = True)

housing
*************************************************
*************************************************
from sklearn.model_selection import train_test_split

# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)
*************************************************
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train
*************************************************

# Dividing the training data set into X and Y
y_train = df_train.pop('price')
X_train = df_train
*************************************************
#Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)

lr_1 = sm.OLS(y_train, X_train_lm).fit()

lr_1.summary()
*************************************************
*************************************************
# Checking for the VIF values of the variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Creating a dataframe that will contain the names of all the feature variables and their VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
*************************************************
# Dropping highly correlated variables and insignificant variables
X = X_train.drop('semi-furnished', 1,)

# Build a fitted model after dropping the variable
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(y_train, X_train_lm).fit()

# Printing the summary of the model
print(lr_2.summary())
*************************************************
# Calculating the VIFs again for the new model after dropping semi-furnished

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
*************************************************
# Now, the variable bedroom has a high VIF (6.6) and a p-value (0.206).
# Hence, it isn’t of much use and should be dropped from the model. 
# We’ll repeat the same process as before

X = X.drop('bedrooms', 1)
# Build a second fitted model
X_train_lm = sm.add_constant(X)
lr_3 = sm.OLS(y_train, X_train_lm).fit()

# Printing the summary of the model
print(lr_3.summary())
*************************************************
# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
*************************************************
# After dropping all the necessary variables one by one, the final model will be
X = X.drop('basement', 1)
X_train_lm = sm.add_constant(X)

lr_4 = sm.OLS(y_train, X_train_lm).fit()
print(lr_4.summary())
*************************************************
# Calculate the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

*************************************************
*************************************************

# Residual Analysis of the train data
# We have to check if the error terms are normally distributed (which is one of 
# the major assumptions of linear regression); let us plot the error terms’ histogram.

import seaborn as sns
y_train_price = lr_4.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
*************************************************

# Similar to the training dataset. First, we have to scale the test data.
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_test[num_vars] = scaler.transform(df_test[num_vars])

df_test
*************************************************

# Dividing the test data into X and Y, after that, 
# we’ll drop the unnecessary variables from the test data based on our model.

y_test = df_test.pop('price')
X_test = df_test

# Adding constant variable to test dataframe
X_test_m4 = sm.add_constant(X_test)

# Creating X_test_m4 dataframe by dropping variables from X_test_m4
X_test_m4 = X_test_m4.drop(["bedrooms", "semi-furnished", "basement"], axis = 1)

# Making predictions using the final model
y_pred_m4 = lr_4.predict(X_test_m4)
*************************************************
# We do that by importing the r2_score library from sklearn
from sklearn.metrics import r2_score
r2_score(y_true = y_test, y_pred = y_pred_m4)

*************************************************
*************************************************
# Recursive Feature Elimination (RFE)
# RFE is an automatic process where we don’t need to select variables manually. 
# We follow the same steps we have done earlier until Re-scaling the features and 
# dividing the data into X and Y.

# We will use the LinearRegression function from sklearn for 
# RFE (which is a utility from sklearn)

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
*************************************************
# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

*************************************************
**********************Building Model*************
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]

# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

print(lm.summary())
*************************************************
# Since the bedrooms column is insignificant to other variables, it can be dropped from the model.
X_train_new = X_train_rfe.drop(["bedrooms"], axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

print(lm.summary())
*************************************************
# Now, we calculate the VIFs for the model.

X_train_new = X_train_new.drop(['const'], axis=1)
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

*************************************************
*****************Residual Analysis***************

y_train_price = lm.predict(X_train_lm)
# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label

********************Evaluating the model on test data*****************************
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_test[num_vars] = scaler.transform(df_test[num_vars])

y_test = df_test.pop('price')
X_test = df_test

# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

# Making predictions
y_pred = lm.predict(X_test_new)


*************************************************
from sklearn.metrics import r2_score
r2_score(y_true = y_test, y_pred = y_pred)

#The R² value for the test data = 0.64, which is pretty similar to the train data.