"""OvaTrace analysis pipeline.
Cleans the PCOS dataset, runs stats/PCA/LDA/models, and writes
  artifacts/results.json     (all dashboard data)
  artifacts/early_rf.joblib  (Random Forest used by /api/predict)
Run:  python analysis.py
"""
import json
from pathlib import Path
import joblib
import pandas as pd, numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import *

ROOT=Path(__file__).parent
DATA=ROOT/'data'/'PCOS_data_without_infertility.xlsx'
OUT=ROOT/'artifacts'
raw=pd.read_excel(DATA,sheet_name='Full_new')
raw.columns=[c.strip() for c in raw.columns]
miss_before=int(raw.isna().sum().sum()); dup=int(raw.duplicated().sum())
d=raw.drop(columns=['Unnamed: 44']).copy()
amh_bad=int(pd.to_numeric(d['AMH(ng/mL)'],errors='coerce').isna().sum())
d['AMH(ng/mL)']=pd.to_numeric(d['AMH(ng/mL)'],errors='coerce')
for c in ['AMH(ng/mL)','Marraige Status (Yrs)']: d[c]=d[c].fillna(d[c].median())
d['Fast food (Y/N)']=d['Fast food (Y/N)'].fillna(d['Fast food (Y/N)'].mode()[0])
d['Cycle(R/I)']=(d['Cycle(R/I)']==4).astype(int)
d['BMI']=d['Weight (Kg)']/(d['Height(Cm)']/100)**2
d['Follicles']=d['Follicle No. (L)']+d['Follicle No. (R)']
S=['Weight gain(Y/N)','hair growth(Y/N)','Skin darkening (Y/N)','Hair loss(Y/N)','Pimples(Y/N)']
d['Symptom_Count']=d[S].sum(axis=1)
y=d['PCOS (Y/N)']
nice={'Age (yrs)':'Age','BMI':'BMI','Weight (Kg)':'Weight','Waist:Hip Ratio':'Waist:Hip','Cycle length(days)':'Cycle length','AMH(ng/mL)':'AMH','Follicles':'Total Follicles','FSH/LH':'FSH/LH'}
NUM=list(nice)
o={'n':len(d),'cols':raw.shape[1],'miss':miss_before,'dup':dup,'amh_bad':amh_bad,'pos':int(y.sum()),'nice':nice}
o['summary']=[[nice[c],round(d[c].mean(),2),round(d[c].std(),2),round(d[c].min(),2),round(d[c].quantile(.25),2),round(d[c].median(),2),round(d[c].quantile(.75),2),round(d[c].max(),2)] for c in NUM]
z=StandardScaler().fit_transform(d[NUM]);mm=(d[NUM]-d[NUM].min())/(d[NUM].max()-d[NUM].min())
o['scale']=[[nice[c],round(d[c].mean(),2),round(d[c].std(),2),round(z[:,i].mean(),2),round(z[:,i].std(),2),round(mm[c].min(),1),round(mm[c].max(),1)] for i,c in enumerate(NUM)]
o['disp']=[[nice[c],round(d[c].var(),2),round(d[c].std(),2),round(d[c].std()/d[c].mean()*100,1),round(d[c].quantile(.75)-d[c].quantile(.25),2)] for c in NUM]
o['skew']=[[nice[c],round(float(stats.skew(d[c])),2),round(float(stats.kurtosis(d[c])),2)] for c in NUM]
cor=d[NUM].corr().round(2);o['corr']=cor.values.tolist();o['cov']=d[NUM].cov().round(1).values.tolist()
# histograms
def hist(c,bins):
    h={}
    for k in (0,1): h[k]=np.histogram(d[y==k][c],bins=bins)[0].tolist()
    return {'bins':[f'{bins[i]:.0f}-{bins[i+1]:.0f}' if bins[1]-bins[0]>=1 else f'{bins[i]:.1f}' for i in range(len(bins)-1)],'neg':h[0],'pos':h[1]}
o['hist']={'BMI':hist('BMI',np.arange(15,42,3)),'Age':hist('Age (yrs)',np.arange(18,50,4)),'Cycle length (days)':hist('Cycle length(days)',np.arange(1,13,2)),'Total follicles':hist('Follicles',np.arange(0,45,5))}
o['scatter']=[[round(a,1),round(b,1),int(c)] for a,b,c in zip(d['Age (yrs)'],d['BMI'],y)]
# tests
tt=[]
for c in NUM+['Endometrium (mm)','TSH (mIU/L)','PRL(ng/mL)','Vit D3 (ng/mL)','RBS(mg/dl)']:
    t,p=stats.ttest_ind(d[y==1][c],d[y==0][c],equal_var=False);tt.append([nice.get(c,c),round(d[y==1][c].mean(),2),round(d[y==0][c].mean(),2),round(float(t),2),float(p)])
o['ttest']=tt
B=['Cycle(R/I)']+S+['Fast food (Y/N)','Reg.Exercise(Y/N)']
o['chi']=[[c.replace('(Y/N)','').replace('(R/I)','irregular').strip(),round(float(stats.chi2_contingency(pd.crosstab(d[c],y))[0]),2),float(stats.chi2_contingency(pd.crosstab(d[c],y))[1]),round(float(np.sqrt(stats.chi2_contingency(pd.crosstab(d[c],y),correction=False)[0]/len(d))),3)] for c in B]
# PCA
pca=PCA().fit(z);pc=pca.transform(z)
o['pca']={'var':[round(v*100,1) for v in pca.explained_variance_ratio_],'ev':[round(v,2) for v in pca.explained_variance_],'load':[[nice[c],round(pca.components_[0][i],2),round(pca.components_[1][i],2)] for i,c in enumerate(NUM)],'pts':[[round(a,2),round(b,2),int(c)] for a,b,c in zip(pc[:,0],pc[:,1],y)]}
lda=LDA().fit(z,y);l1=lda.transform(z)[:,0]
bins=np.linspace(l1.min(),l1.max(),13)
o['lda']={'bins':[round((bins[i]+bins[i+1])/2,1) for i in range(12)],'neg':np.histogram(l1[y==0],bins)[0].tolist(),'pos':np.histogram(l1[y==1],bins)[0].tolist(),'coef':[[nice[c],round(float(lda.scalings_[i,0]),2)] for i,c in enumerate(NUM)],'acc':round(lda.score(z,y),3),'means':[[nice[c],round(d[y==0][c].mean(),2),round(d[y==1][c].mean(),2)] for c in NUM]}
# RF
FE=['Age (yrs)','BMI','Waist:Hip Ratio','Cycle(R/I)','Cycle length(days)']+S+['Fast food (Y/N)','Reg.Exercise(Y/N)','Follicle No. (L)','Follicle No. (R)','AMH(ng/mL)','FSH/LH','TSH (mIU/L)','PRL(ng/mL)','Vit D3 (ng/mL)','RBS(mg/dl)']
Xtr,Xte,ytr,yte=train_test_split(d[FE],y,test_size=.25,stratify=y,random_state=42)
rf=RandomForestClassifier(300,min_samples_leaf=2,random_state=42,class_weight='balanced').fit(Xtr,ytr)
pr=rf.predict_proba(Xte)[:,1];pd_=(pr>=.5).astype(int);fpr,tpr,_=roc_curve(yte,pr);ix=np.linspace(0,len(fpr)-1,35).astype(int)
tn,fp,fn,tp=confusion_matrix(yte,pd_).ravel()
cv=cross_val_score(rf,d[FE],y,cv=StratifiedKFold(5,shuffle=True,random_state=1),scoring='roc_auc')
o['rf']={'acc':round(accuracy_score(yte,pd_),3),'prec':round(precision_score(yte,pd_),3),'rec':round(recall_score(yte,pd_),3),'f1':round(f1_score(yte,pd_),3),'auc':round(roc_auc_score(yte,pr),3),'cvauc':round(cv.mean(),3),'cvsd':round(cv.std(),3),'cm':[int(tn),int(fp),int(fn),int(tp)],'roc':[[round(fpr[i],3),round(tpr[i],3)] for i in ix],'imp':sorted([[c.replace('(Y/N)','').replace('(mIU/mL)','').strip(),round(float(v),3)] for c,v in zip(FE,rf.feature_importances_)],key=lambda t:-t[1])[:12],'ntr':len(Xtr),'nte':len(Xte)}
# insights
o['symrate']=[[int(k),int(len(g)),round(g['PCOS (Y/N)'].mean()*100,1)] for k,g in d.groupby('Symptom_Count')]
def rate(m):return round(y[m].mean()*100,1),int(m.sum())
o['combo']={'irr+hg':rate((d['Cycle(R/I)']==1)&(d['hair growth(Y/N)']==1)),'irr+sd+wg':rate((d['Cycle(R/I)']==1)&(d['Skin darkening (Y/N)']==1)&(d['Weight gain(Y/N)']==1)),'fol>=20':rate(d['Follicles']>=20),'fol<10':rate(d['Follicles']<10),'amh>=5':rate(d['AMH(ng/mL)']>=5),'amh<2':rate(d['AMH(ng/mL)']<2),'none':rate((d['Cycle(R/I)']==0)&(d[S].sum(axis=1)==0)),'ff+noex':rate((d['Fast food (Y/N)']==1)&(d['Reg.Exercise(Y/N)']==0)),'ff+ex':rate((d['Fast food (Y/N)']==1)&(d['Reg.Exercise(Y/N)']==1))}
o['pca_sep']=round(float(abs(np.corrcoef(pc[:,0],y)[0,1])),2);o['pca_sep2']=round(float(abs(np.corrcoef(pc[:,1],y)[0,1])),2)
# calculator: logistic on early features
E=['Age (yrs)','BMI','Waist:Hip Ratio','Cycle(R/I)','Cycle length(days)']+S+['Fast food (Y/N)','Reg.Exercise(Y/N)']
sc=StandardScaler().fit(d[E]);lr=LogisticRegression(C=.5,max_iter=2000).fit(sc.transform(d[E]),y)
w=lr.coef_[0]/sc.scale_;b=lr.intercept_[0]-np.sum(lr.coef_[0]*sc.mean_/sc.scale_)
o['calc']={'f':E,'w':[round(float(v),5) for v in w],'b':round(float(b),4),'auc':round(float(cross_val_score(lr.__class__(C=.5,max_iter=2000),sc.transform(d[E]),y,cv=5,scoring='roc_auc').mean()),3)}
o['kpi']={'prev':round(y.mean()*100,1),'age':round(d['Age (yrs)'].mean(),1),'bmi':round(d['BMI'].mean(),1),'irr':round(d['Cycle(R/I)'].mean()*100,1)}

# ---- lifestyle statistics -------------------------------------------------
ff,ex=d['Fast food (Y/N)'],d['Reg.Exercise(Y/N)']
life={'combo':[],'bmiex':[]}
for f in (0,1):
    for e in (0,1):
        m=(ff==f)&(ex==e)
        life['combo'].append([('Fast food' if f else 'No fast food')+' + '+('exercise' if e else 'no exercise'),int(m.sum()),round(float(y[m].mean()*100),1)])
band=pd.cut(d['BMI'],[0,25,30,100],labels=['BMI <25','BMI 25–30','BMI ≥30'])
for b in ['BMI <25','BMI 25–30','BMI ≥30']:
    r=[b]
    for e in (0,1):
        m=(band==b)&(ex==e); r+=[int(m.sum()),round(float(y[m].mean()*100),1)]
    life['bmiex'].append(r)
o['life']=life

# ---- server-side model: Random Forest on the 12 non-invasive features -----
mkrf=lambda:RandomForestClassifier(400,min_samples_leaf=3,random_state=42,class_weight='balanced')
early_rf=mkrf().fit(d[E],y)
o['calc']['rf_auc']=round(float(cross_val_score(mkrf(),d[E],y,cv=StratifiedKFold(5,shuffle=True,random_state=1),scoring='roc_auc').mean()),3)
OUT.mkdir(exist_ok=True)
joblib.dump({'model':early_rf,'features':E},OUT/'early_rf.joblib')

class Enc(json.JSONEncoder):
    def default(s,x): return x.item() if hasattr(x,'item') else str(x)
(OUT/'results.json').write_text(json.dumps(o,separators=(',',':'),cls=Enc))
print('Wrote results.json and early_rf.joblib | RF test AUC',o['rf']['auc'],'| early RF CV AUC',o['calc']['rf_auc'])
