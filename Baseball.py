
# coding: utf-8

# In[ ]:

'''Baseball Data - A data set containing complete batting and pitching statistics from 1871 to 2014, plus fielding statistics,
standings, team stats, managerial records, post-season data, and more.

The following is an analysis of the batters and pitchers based on their performance. 

The data shown below helps to make critical decision during the transfer season and picking the required players to play for
their team. The data of the player performances have been taken and analysed below and the conclusions have been drawn at the 
end in graphical format.

'''


# In[ ]:

'''
Here is a list of Abbreviations used in baseball:

yearID: Year
teamID: Team
Rank: Position in final standings
R: Runs scored
RA: Opponents runs scored
G: Games played
W: Wins
H: Hits by batters
BB: Walks by batters
HBP: Batters hit by pitch
AB: At bats
SF: Sacrifice flies
HR: Homeruns by batters
2B: Doubles
3B: Triples
'''


# In[186]:

#importing the Files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the files from the directory
allstar=pd.read_csv("C:/Users/Prajval/baseball/AllstarFull.csv")
AwardsManagers=pd.read_csv("C:/Users/Prajval/baseball/AwardsManagers.csv")
AwardsPlayers=pd.read_csv("C:/Users/Prajval/baseball/AwardsPlayers.csv")
Batting=pd.read_csv("C:/Users/Prajval/baseball/Batting.csv")
Fielding=pd.read_csv("C:/Users/Prajval/baseball/Fielding.csv")
HallOfFame=pd.read_csv("C:/Users/Prajval/baseball/HallOfFame.csv")
Managers=pd.read_csv("C:/Users/Prajval/baseball/Managers.csv")
Pitching=pd.read_csv("C:/Users/Prajval/baseball/Pitching.csv")
Salaries=pd.read_csv("C:/Users/Prajval/baseball/Salaries.csv")
Teams=pd.read_csv("C:/Users/Prajval/baseball/Teams.csv")
Master=pd.read_csv("C:/Users/Prajval/baseball/Master.csv")


# In[ ]:

'''

The Batting Average is defined by the number of hits divided by at bats. It can be calculated using the formula below:
BA = H/AB

On-base Percentage is a measure of how often a batter reaches base for any reason other than a fielding error, fielder's choice, dropped/uncaught third strike, fielder's obstruction, or catcher's interference. It can be calculated using the formula below:
OBP = (H+BB+HBP)/(AB+BB+HBP+SF)

Slugging Percentage is a measure of the power of a hitter. It can ve calculated using the formula below:
SLG = H+2B+(2*3B)+(3*HR)/AB


We will add these 3 measures to our teams DataFrame by running the following commands:

'''


# In[188]:

#Batting Statistics

Batting['BA'] = Batting['H']/Batting['AB']
Batting['OBP'] = (Batting['H'] + Batting['BB'] + Batting['HBP']) / (Batting['AB'] + Batting['BB'] + Batting['HBP'] + Batting['SF'])
Batting['SLG'] = (Batting['H'] + Batting['2B'] + (2*Batting['3B']) + (3*Batting['HR'])) / Batting['AB']


# In[189]:

'''
To predict the value of the batter, we have to consider the statistics based on attributes like Runs scored, Batting average, 
On base Percentage and so on.

We will use the linear regression model to verify which baseball stats are more important to predict runs.

We will build 3 different models: T
he first one will have as features OBP, SLG and BA. 
The second model will have as features OBP and SLG. 
The third one will have as feature BA only.

'''

import statsmodels.formula.api as sm

#First Model
runs_reg_model1 = sm.ols("R~OBP+SLG+BA",Batting)
runs_reg1 = runs_reg_model1.fit()
#Second Model
runs_reg_model2 = sm.ols("R~OBP+SLG",Batting)
runs_reg2 = runs_reg_model2.fit()
#Third Model
runs_reg_model3 = sm.ols("R~BA",Batting)
runs_reg3 = runs_reg_model3.fit()

'''Looking at the outputs of these models'''

runs_reg1.summary()
#runs_reg2.summary()
#runs_reg3.summary()


# In[ ]:

'''
The first model has an Adjusted R-squared of 0.918, with 95% confidence interval of BA between -283 and 468. 

This is counterintuitive, since we expect the BA value to be positive. This is due to a multicollinearity between the variables.

The second model has an Adjusted R-squared of 0.919, and the last model an Adjusted R-squared of 0.500.

Based on this analysis, we could confirm that the second model using OBP and SLG is the best model for predicting Run Scored.
'''


# In[190]:

Master["FullName"] = Master["nameFirst"].map(str) + Master["nameLast"]


# In[191]:

'''Best Hitters Of all time'''

alltime = Batting
#alltime.set_index(['playerID'], inplace=True)
alltime = alltime.reset_index().groupby('playerID').sum()
alltime = alltime.sort_values(['OBP'], ascending=False)
alltime.drop(alltime[alltime.G < 100].index, inplace=True)

#Merge

alltime.reset_index(inplace=True)
mrg = pd.merge(alltime, Master, on='playerID', how='inner')

top = ['FullName', 'G', 'R', 'debut', 'AB', 'OBP', 'SLG', 'AB' ]
top = mrg[top]
top.head(20)


# In[179]:

'''Best Hitters of Current Generation'''

new = Batting
new = new.drop(new[new.yearID < 2015].index)
new.drop(new[new.G < 50].index, inplace=True)
new = new.sort_values(['OBP'], ascending=False)

#Merge

new.reset_index(inplace=True)

mrge = pd.merge(new, Master, on='playerID', how='inner')

topnew = ['FullName', 'G', 'R', 'debut', 'AB', 'OBP', 'SLG', 'AB' ]
topnew = mrge[topnew]
topnew.head(26)


# In[70]:

'''Most valuable players of current generation'''


# In[192]:

y = Batting
y = y.drop(y[y.yearID < 2010].index)
y.drop(y[y.G < 10].index, inplace=True)
y.drop(y[y.R < 10].index, inplace=True)
y = y.sort_values(['OBP'], ascending=False)
y

#Merge

y.reset_index(inplace=True)
y

mg = pd.merge(y, Salaries, on='playerID', how='left')
mg = mg.sort_values(['salary'], ascending=True)

mgnew = pd.merge(mg, Master, on='playerID', how='left')
mgnew = mgnew.set_index(['playerID'])
mgnew = mgnew.groupby(mgnew.index).first()
mgnew = mgnew.reset_index()

mn = ['FullName', 'G', 'R', 'debut', 'AB', 'OBP', 'SLG', 'AB' ,'salary']
mn = mgnew[mn]
mn.head(51)


# In[195]:

'''Most Under-rated Players'''
#mn.set_index(['FullName'], inplace=True)
mn = mn.sort_values(['salary'], ascending=True)
mn


# In[194]:

'''Best Pitchers of current generation(Based on ERA)'''


p = Pitching
#p.set_index(['playerID'], inplace=True)

#alltime = alltime.reset_index().groupby('playerID').sum()
#alltime = alltime.sort_values(['OBP'], ascending=False)
#p.drop(p[p.G < 50].index, inplace=True)
p = p.sort_values(['ERA'], ascending=False)

p = p.drop(p[p.yearID < 2010].index)

#Merge

p.reset_index(inplace=True)
pm = pd.merge(p, Master, on='playerID', how='inner')

top1 = ['FullName', 'yearID', 'G', 'R', 'debut', 'G', 'W', 'L', 'H', 'ER', 'ERA' ]
top1 = pm[top1]
top1.head(20)


# In[208]:

'''The Following graph shows the distribution of batters based on their salaries.
The Player names are distributed along the X-axis and the average salaries on the y-axis. 
The graph also indicates the runs scored by the batters in each season. As the graph moves from green to red in color, the 
runs scored by the player also increases. Thus the players having more red areas with their name have scored more runs.

Only players with OBP > 0.345 have been included. 
These players are the ones that are to be targeted when the transfer window opens up.


This Graph helps in making selections during the transfer season. The players can be chosen based on their 
market value and performances.
'''


# In[207]:

get_ipython().run_cell_magic('HTML', '', "\n\n<script type='text/javascript' src='https://us-east-1.online.tableau.com/javascripts/api/viz_v1.js'></script><div class='tableauPlaceholder' style='width: 1523px; height: 678px;'><object class='tableauViz' width='1523' height='678' style='display:none;'><param name='host_url' value='https%3A%2F%2Fus-east-1.online.tableau.com%2F' /> <param name='site_root' value='&#47;t&#47;prajval' /><param name='name' value='Baseball-batting&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='showAppBanner' value='false' /><param name='showShareOptions' value='true' /></object></div>")


# In[209]:

'''The Following graph shows the distribution of pitchers based on their salaries.
The Player names are distributed along the X-axis and the average salaries on the y-axis. 
The graph also indicates the Strike Outs of pitchers. As the graph moves from green to red in color, the 
Strike Outs also increases. Thus the pitchers having a color more closer to red have more Strike outs.

The players are ranked based on their ERA. The graph shows the percentile of ERA of each player. 
The percentile of ERA of a Pitcher closer to 50  indicates that he is statistically stronger than the pitchers who are below 
him in the percentile score.

This Graph helps in making selections during the transfer season. The players can be chosen based on their 
market value and performances.

'''



# In[211]:

get_ipython().run_cell_magic('HTML', '', "\n<script type='text/javascript' src='https://us-east-1.online.tableau.com/javascripts/api/viz_v1.js'></script><div class='tableauPlaceholder' style='width: 1523px; height: 678px;'><object class='tableauViz' width='1523' height='678' style='display:none;'><param name='host_url' value='https%3A%2F%2Fus-east-1.online.tableau.com%2F' /> <param name='site_root' value='&#47;t&#47;prajval' /><param name='name' value='Baseball-pitcher&#47;Sheet5' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='showAppBanner' value='false' /><param name='showShareOptions' value='true' /></object></div>")


# In[ ]:

'''Hence the data has been analysed and conclusions are drawn. The data can be put to good use during the transfer season
where the managers will be hunting for upcoming talents who put great value to the team. '''


# In[ ]:



