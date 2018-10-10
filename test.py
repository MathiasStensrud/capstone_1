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
df =pd.read_csv('XXH2017_YRBS_Data.dat')
df2 =pd.read_csv('yrbs2015.dat')
# df.info()
##q2: ,q66 ,q67 ,q32 ,q42
print('updated')

s=[]
a=[]
c=[]
g=[]
ag=[]
w=[]
for i in range(len(df2['Students'])):
    temp=df2['Students'].loc[i][97]
    alc=df2['Students'].loc[i][73]
    cig=df2['Students'].loc[i][62]
    age=df2['Students'].loc[i][16]
    gen=df2['Students'].loc[i][17]
    wC=df2['Students'].loc[i][74]
    if temp==' ' or gen==' ' or alc==' 'or age==' ':
        df2.drop([i],inplace=True)
    else:
        s.append(int(temp))
        a.append(int(alc))
        if cig==' ':
            c.append(0)
        else:
            c.append(int(cig))
        g.append(int(gen))
        ag.append(int(age))
        if wC==' ':
            w.append(0)
        else:
            w.append(int(wC))

corrTest=pd.DataFrame({'Sexuality':s,'Alcohol Usage':a,'Smoking':c, 'Age':ag, 'Gender': g, 'Wildcard':w})
# print(corrTest.corr())
# sns.pairplot(corrTest)
# plt.show()
# gS=[]
# gA=[]
# sS=[]
# sA=[]

# for i in range(len(s)):
#     if s[i]>1 and s[i]<4:
#         if a[i]>1:
#             gA.append(1)
#         else:
#             gA.append(0)
#     else:
#         if a[i]>1:
#             sA.append(1)
#         else:
#             sA.append(0)
bdl={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
gdl={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
sdl={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
qdl={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
bd=[]
gd=[]
sd=[]
qd=[]
factors=[]
st=[]
for i in range(len(s)):
    if a[i]>1:
        st.append(1)
    else:
        st.append(0)
    if s[i]==2:
        gdl[a[i]]+=(1)
        if a[i]>1:
            gd.append(1)
        else:
            gd.append(0)
    elif s[i]==1:
        sdl[a[i]]+=(1)
        if a[i]>1:
            sd.append(1)
        else:
            sd.append(0)
    elif s[i]==3:
        bdl[a[i]]+=(1)
        if a[i]>1:
            bd.append(1)
        else:
            bd.append(0)
    elif s[i]==4:
        qdl[a[i]]+=(1)
        if a[i]>1:
            qd.append(1)
        else:
            qd.append(0)
    factors.append([g[i],s[i],ag[i]])
gavg=0
savg=0
bavg=0
for i in range(2, len(sdl)):
    savg+=sdl[i]
    gavg+=gdl[i]
    bavg+=bdl[i]
gavg=np.mean(gavg)
savg=np.mean(savg)
bavg=np.mean(bavg)

X=np.asarray(factors)
y=np.asarray(st)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model=LogisticRegressionCV(cv=10,class_weight='balanced',random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)
acc=accuracy_score(y_test, pred)
print(acc)
# rmse=np.sqrt(mean_squared_error(y_test, pred))
# print(rmse)
# probs = model.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold =roc_curve(y_test, preds)
# roc_score=round(roc_auc_score(y_test,preds),2)
# meanline=np.arange(0,1.1,.1)
# plt.plot(meanline,meanline, color='black')
# plt.plot(fpr,tpr, label=f'Area under Curve: {roc_score}')
# plt.title('ROC Curve')
# plt.xlabel('False Positive')
# plt.ylabel('True Positive')
# plt.savefig('ROC.png')
# plt.legend()
# plt.show()
fig=plt.figure(figsize=(8,8))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)



ax1.pie(bdl.values(),labels=None)
ax1.set_title('Bi drinking rates')
ax2.pie(sdl.values(),labels=None)
ax2.set_title('Straight drinking rates')
ax3.pie(gdl.values(),labels=None)
ax3.set_title('Gay/Lesbian drinking rates')
plt.savefig('drinking.png')
plt.show()
# gsdr=sum(gsd)/(len(gsd)-sum(gsd))
# ssdr=sum(ssd)/(len(ssd)-sum(ssd))
