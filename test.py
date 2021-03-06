import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error
df =pd.read_csv('XXH2017_YRBS_Data.dat')#former use data that could be appended if wanted for more security of response
df2 =pd.read_csv('yrbs2015.dat')#data Im actually using

##q2: ,q66 ,q67 ,q32 ,q42

def get_perc(d):
    #grabs percentages from a dictionary inefficent at large sizes
    sm=0
    perc=[]
    for i in d.keys():
        sm+=d[i]
    for i in d.keys():
        temp=(d[i]/sm)*100
        perc.append(round(temp,1))
    return perc

def corrPlot(dat):
    #plots a pairplot
    sns.pairplot(dat)
    plt.show()
    pass

def piePlot(vals,labels, leg=None, save=False, titles=None):
    #plots up to 4 pie charts with various conditions regarding them
    fig=plt.figure(figsize=(8,8))
    ax1=fig.add_subplot(2,2,1)
    ax2=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(2,2,3)
    ax4=fig.add_subplot(2,2,4)
    axes=[ax1,ax2,ax3,ax4]
    for i in range(len(vals)):
        #axes[i].set_title(f'{titles[i]}') #still in works/tempermental
        axes[i].pie(vals[i],labels=labels[i])
    if leg:
        plt.legend(leg, loc=1)
    if save:
        plt.savefig(f'{title}.png')
    plt.show()
    pass

def plot_roc(test_fact, true, model):
    #plots roc curve from test values
    probs = model.predict_proba(test_fact)
    preds = probs[:,1]
    fpr, tpr, thresholesbian_drinking_rate =roc_curve(true, preds)
    roc_score=round(roc_auc_score(true,preds),3)
    meanline=np.arange(0,1.1,.1)

    plt.plot(meanline,meanline, color='black')
    plt.plot(fpr,tpr, label=f'Area under Curve: {roc_score}')
    plt.title('ROC Curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.savefig('ROC.png')
    plt.legend()
    plt.show()
    pass

sex=[]
alc=[]
cig=[]
gen=[]
age=[]
wc=[]
for i in range(len(df2['Students'])):
    #a: 1 0 days where drunk alchohol per month, 2 1-2 days per month, 3 3-5 days
    #4 6-9 5 10-19
    #loop assigning data to easily acessible variables. makes everything else a bit easier,
    temp=df2['Students'].loc[i][97]#sexuality
    Alc=df2['Students'].loc[i][73]#Alcohol usage in last month
    Cig=df2['Students'].loc[i][62]#igarette usage past month
    Age=df2['Students'].loc[i][16]#age
    Gen=df2['Students'].loc[i][17]#gender
    wC=df2['Students'].loc[i][56]#wildcard, currently depression screener
    if temp==' ' or Gen==' ' or Alc==' 'or Age==' ': #essential components, cant be imputed
        df2.drop([i],inplace=True)
    else:
        sex.append(int(temp))
        alc.append(int(Alc))
        if Cig==' ':#cvan be imputed, cigarette
            cig.append(0)
        else:
            cig.append(int(Cig))
        gen.append(int(Gen))
        age.append(int(Age))
        if wC==' ':#wildcard, can be imputed
            wc.append(0)
        else:
            wc.append(int(wC))

corrTest=pd.DataFrame({'Sexuality':sex,'Alcohol Usage':age,'Smoking':cig, 'Age':age, 'Gender': gen, 'Wildcard':wc})

#bad assignment variables

bi_drinking_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
gay_drinking_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
straight_drinking_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
questioning_drinking_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
lesbian_drinking_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
# ADVICE: get method dict comprehension list comprehension
factors=[]#features
y_vals=[]#these are my y values
for i in range(len(s)):
    #worst loop in here
    if alc[i]>1:#(you been drinking kiddo?)
        y_vals.append(1)#Yes
    else:
        y_vals.append(0)#no
    if sex[i]==2 and gen[i]==2:#s==2 checks for answer (Gay/Lesbian) g==2 checks if male
        gay_drinking_dict[alc[i]]+=(1)#adds to dictionary containg values for each category
    elif sex[i]==2 and gen[i]==1:#same as above but checks for female
        lesbian_drinking_dict[alc[i]]+=(1)
    elif sex[i]==1:#straight
        straight_drinking_dict[alc[i]]+=(1)
    elif sex[i]==3:#bisexual
        bi_drinking_dict[alc[i]]+=(1)
    elif sex[i]==4:#questioning/not sure
        questioning_drinking_dict[a[i]]+=(1)
    factors.append([sex[i],age[i],gen[i],wc[i],cig[i]]) #adds corresponding features for student

#percentages for pie chart labels
pg=get_perc(gay_drinking_dict)
ps=get_perc(straight_drinking_dict)
pl=get_perc(lesbian_drinking_dict)
pb=get_perc(bi_drinking_dict)
pq=get_perc(questioning_drinking_dict)

#MODELING
X=np.asarray(factors)
y=np.asarray(y_vals)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model=LogisticRegressionCV(cv=10,class_weight='balanced',random_state=42)
model.fit(X_train,y_train)
#how accurate?
pred=model.predict(X_test)
acc=accuracy_score(y_test, pred)
print(acc)#not very :(

#for pie chart
alclab=['0 days',' 1-2 days','3-5 days','6-9 days','10-19 days','20-29 days','30 days']# for legend for pie chart
drink_vals=([bi_drinking_dict.values(),straight_drinking_dict.values(),gay_drinking_dict.values(),
lesbian_drinking_dict.values()])
drink_labs=([pb,ps,pg,pl])
