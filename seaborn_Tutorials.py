#https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/

#Seaborn data visualisation
#type of plots
    '''
    01. sns.distplot() singlevariate numerical
    02. sns.jointplot() bivariate numerical
    03. sns.pairplot() multivariate numerical
    04. sns.rugplot() singlevariate numerical
    05. sns.barplot() categorical
    06. sns.countplot() categorical
    07. sns.boxplot() categorical and numerical
    08. sns.violinplot() categorical and numerical
    09. sns.stripplot() categorical and numerical
    10. sns.factorplot() categorical anf numerical
    11. sns.heatmap() matrixplot type
    12. sns.clustermap() matrixplot type
    13. sns.kdeplot() numerical
    #special plot sns.PairGrid()
    
    
    '''
#histogram/distribution plot

import seaborn as sns


#load tips dataset built in sns
tips=sns.load_dataset('tips')

tips.head()

#To plot a distribution plot with kde(kernel density estimation)
sns.distplot(tips['total_bill'])

#To plot a distribution plot without kde
sns.distplot(tips['total_bill'],kde=False)

#To plot a distribution plot with variable(n) number of bins
sns.distplot(tips['total_bill'],bins=n)

#To plot joint plot with scatter points
sns.jointplot(x='total_bill', y='tip', data=tips)

#To plot joint plot with hex kind of distribution representation    
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')

#To plot a joint plot with regression kind of distribution
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')

#To plot a joint plot with contour/2-D kde kind of representation
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')

#To plot a pair plot without a hue
sns.pairplot(tips)

#To plot a pair plot with a hue
sns.pairplot(tips, hue='sex')

#To plot a pair plot with a color palette
sns.pairplot(tips, palette='coolwarm')

#To plot a rug plot
sns.rugplot(tips['total_bill'])

#To plot a normal distriution for every rug plot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#create a random dataset
dataset=np.random.randn(25)
#create a rug plot
sns.rugplot(dataset)
xmin=dataset.min()-2
xmax=dataset.max()+2
#to create 100 equal spaced points from xmin to xmax
xaxis=np.linspace(xmin, xmax, 100)
#setup the bandwidth
bandwidth=((4*dataset.std()**5)/(3*len(dataset)))**2
#create an empty kernel list
kernellist=[]
#plot each bases function
for datapoint in dataset:
    #create a kernel for each point and append to list
    kernel=stats.norm(datapoint, bandwidth).pdf(xaxis)
    kernellist.append(kernel)
    
    #scale for plotting
    kernel=kernel/kernel.max()
    kernel=kernel*0.4
    plt.plot(xaxis,kernel,color='grey',alpha=0.5)
plt.ylim(0,1)

#to sum all the gaussian kernel to generate a kde
#plot the sum of bases function
sumofkde=np.sum(kernellist, axis=0)
#plot figure
fig=plt.plot(xaxis, sumofkde, color='red')
#To plot rug plot again
sns.rugplot(dataset, color='red')
#remove yticks
plt.yticks([])
#set title
plt.suptitle(''sum of bases function)

#To plot a bar plot
sns.barplot(x='sex', y='total_bill',data=tips)

#To plot a bar plot with estimator as standard deviation
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)

#To plot a count plot
sns.countplot(x='sex', data=tips)

#To plot box plot
sns.boxplot(x='sex', y='total_bill', data=tips)

#To plot box plot with hue
sns.boxplot(x='sex', y='total_bill', data=tips, hue='smoker')

#To plot a violin plot
sns.violinplot(x='sex', y='total_bill', data=tips)

#To plot a violin plot with a hue
sns.violinplot(x='sex', y='total_bill', data=tips, hue='smoker')

#To plot a violin plot with a split
sns.violinplot(x='sex', y='total_bill', data=tips, hue='smoker', split=True)

#To plot a strip plot
sns.stripplot(x='sex', y='total_bill', data=tips)

#To plot a strip plot with jitter
sns.stripplot(x='sex', y='total_bill', data=tips, jitter=True)

#To plot a strip plot with hue
sns.stripplot(x='sex', y='total_bill', data=tips, hue='smoker')

#To plot a swarm plot
sns.swarmplot(x='sex', y='total_bill', data=tips)

#To plot a factor plot
sns.factorplot(x='sex', y='total_bill', data=tips, kind='bar')

#To plot matrix plot(heatmaps, clustermap) process data
tips=sns.load_dataset('tips')
flight=sns.load_dataset('flights')
#matrix/heatplot using correlation
tc=tips.corr()
#matrix/heatplot using pivot tables
fp=flight.pivot_table(index='month', columns='year', values='passengers')


#To plot heatmap
sns.heatmap(tc)
#heatmap with values
sns.heatmap(tc, annot=True)
#heatmap with cmap
sns.heatmap(tc, cmap='coolwarm')
#heatmap with lines
sns.heatmap(fp, linecolor='white', linewidth=1)

#To plot clustermap
sns.clustermap(fp)
#To plot clustermap with cmap
sns.clustermap(fp, cmap='coolwarm')
#To plot clustermap with lines
sns.clustermap(fp, linecolor='white', linewidth=1)
#To plot clustermap with standard scale
sns.clustermap(fp, standard_scale=1)

#To plot PairGrid plot
iris=sns.load_dataset('iris')
g=sns.PairGrid(iris)
g.map(sns.scatter)
#To plot diagonal grid type plots
g.map_diag(sns.distplot)
#To plot upper triangle grid type plot
g.map_upper(sns.scatter)
#To plot lower triangle grid type plot
g.map_lower(sns.kdeplot)

#To plot FacetGrid
g=sns.FacetGrid(data=tips, col='sex', row='total_bill')
g.map(sns.distplot, 'total_bill')

#Regression plots
#To plot a lmplot
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex')

#To style seaborn plots
sns.set_styles('ticks') #{ticks,white,whitegrid,darkgrid}
#To customize spine in seaborn
sns.despine(left=True, bottom=Fals)
#To change figure size
plt.figure(figsize=(12,3)) 
sns.set_context('poster', font_scale=2)


#Notes
'''

joint plot gives p value and pearsons correlation for two given variates 
pair plots plot histogram for column vs same columns
pair plot plots scatter plpots for column vs other columns
rugplot is related to kde
box and violin plot are retaed
violin plot gives more detailed insights but harder to read
box plots can be used to identify outliers
swarm plots are combination of violin and strip plot
Hue divides the plot into categories of categorical data
pair grid are base of pairplots where control is given to the user


to chage the aspect ratio use aspect=0.6 to change the width and height of the plot
to change the size of the plot use size=8
use row='column3', col='column4', hue='column5' to add more variables into a plot
to seperate the plottings based on diffrent plotrs instead of colors use col='' instead of hue=''





'''

sns.set_style(style='darkgrid')
sns.set()
sns.set_context("paper")
