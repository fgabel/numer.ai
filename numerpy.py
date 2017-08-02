import pandas as pd
import numpy as np
import xgboost as xgb
#from sklearn.model_selection import train_test_split
#import xgboost as xgb
#from sklearn import cross_validation as CV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLars
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
print("Loading data...")
path = 'C:/Users/Frank/Desktop/Kaggle/numer.ai/'
# Load the data from the CSV files
X = pd.read_csv(path+'numerai_training_data.csv', header=0)
holdout = pd.read_csv(path+'numerai_tournament_data.csv', header=0)



def trainy(X, data_type = None, cv = None, resemble = False, final_pred = False):
    """
    Method for splitting the dataset into train and validation set.
    
    vars
    data_type: use when importing the tournament data to split val, test, live
    cv: use for setting up a era-subset validation set. cv is the size of the train set.
    resemble: restrict eras to specific eras with certain target. deprecated.
    final_pred: if we are aiming for a final prediction, we want to take all eras
    """
    if data_type:
        X = X.loc[X["data_type"]==data_type,:]
    if cv:
        rnd = set(np.random.choice(X.era.unique(), cv, replace=False))
        alt = set(X.era.unique())-rnd
        traina = X.loc[X["era"].isin(rnd),:] # era subsetting
        train_val = X.loc[X["era"].isin(alt),:]
        # only include eras that resemble the val.
        if resemble:
            ab = {val: ke for (ke,val) in dict(traina.groupby(by=['era']).target.mean()).items()}
            hehe = train_val.groupby(by=['era']).target.mean()
            newrnd = set([ab[key] for key in ab if abs(key-np.asarray(hehe))<0.015])
            #for final validation:
            
            print("We´re here")
            traina = traina.loc[traina["era"].isin(newrnd),:]
        if final_pred: 
            ab = {val: ke for (ke,val) in dict(traina.groupby(by=['era']).target.mean()).items()}
            hehe = train_val.groupby(by=['era']).target.mean()
            newrnd = set([ab[key] for key in ab if 0.49<key<0.51])
            traina = traina.loc[traina["era"].isin(newrnd),:]
        y_train = traina["target"]
        print("Train mean:{}, train max:{}".format(y_train.mean(), max(traina.groupby(by=['era']).target.mean())))
        traina.drop(["target"], axis=1, inplace = True)   
        y_val = train_val["target"]
        train_val.drop(["target"], axis=1, inplace = True)
        print("train set: {}".format(sorted(traina.era.unique())))
        print("test set: {}".format(sorted(list(alt))))
        
        return traina, y_train, train_val, y_val
    y = X["target"]
    X.drop(["target"], axis=1, inplace = True)  
    return X, y


def GridSearch(estimator, train, y_train):
    parameters = {'n_estimators':(500, 1000, 1500), 'max_depth':[6, 8, 10]}
    ftwo_scorer = make_scorer(log_loss)
    
    clf = GridSearchCV(estimator, parameters, scoring = ftwo_scorer)
    clf.fit(a,b)
    print("Best parameters are: {} with score {}".format(clf.best_params_, clf.best_score_))
    return clf

from sklearn.manifold import TSNE

def new_feature(data):
    
#    tsvd = TruncatedSVD(n_components=2, random_state=420)
#    tsvd_results_train = tsvd.fit_transform(data)
    
    # PCA
    pca = PCA(n_components=5, random_state=420)
    pca2_results_train = pca.fit_transform(data)
    
    # ICA
#    ica = FastICA(n_components=2, random_state=420)
#    ica2_results_train = ica.fit_transform(data)
##    
#    # GRP
#    grp = GaussianRandomProjection(n_components=2, eps=0.1, random_state=420)
#    grp_results_train = grp.fit_transform(data)
#    
#    # SRP
#    srp = SparseRandomProjection(n_components=2, dense_output=True, random_state=420)
#    srp_results_train = srp.fit_transform(data)
#    
    for i in range(1, 2 + 1):
        data['pca_' + str(i)] = pca2_results_train[:, i - 1]
    
#        data['ica_' + str(i)] = ica2_results_train[:, i - 1]
#    
#        data['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
#    
#        data['grp_' + str(i)] = grp_results_train[:, i - 1]
#    
#        data['srp_' + str(i)] = srp_results_train[:, i - 1]
    return data
#extract strings that contain "feature"
#def feature_engin(X):
#    features = [f for f in list(train) if "feature" in f]
#    
    
# super important: feature 10 & feature 17 multiplied    
    
print("Unique number of eras is {}".format(X.era.unique().shape[0]))
a, b, c, d = trainy(X, cv=80, resemble=False, final_pred = True)
d.mean()


train, y_train = trainy(X)
val, y_val = trainy(holdout, 'validation')
test, _ = trainy(holdout, 'test')
live, _ = trainy(holdout, 'live')

final_pred, _ = trainy(holdout) # in the end, we predict all holdout data
ids = final_pred.id

val_ids = pd.DataFrame({'era': val.era})
valc_ids = pd.DataFrame({'era': c.era})

features = [f for f in list(train) if "feature" in f]


#poly = PolynomialFeatures(2)
#aa = poly.fit_transform(a)
"""kick non-feature columns"""
final_pred = final_pred[features]
a = a[features]
c = c[features]
val = val[features]

"""append PCA / ICA"""

final_pred = new_feature(final_pred)
a = new_feature(a)
c = new_feature(c)
val = new_feature(val)



#dtrain = xgb.DMatrix(a, b)
#dtest = xgb.DMatrix(c, d)
print("Running XGB model...")

    

xgbo = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.002, subsample=0.7, colsample_bytree=0.85, objective='binary:logistic', silent=False)
xgbo.fit(a, b)


rfc = RandomForestRegressor(n_estimators=300, max_features='auto', max_depth=3, verbose = 1)
rfc.fit(a, b)

#print ("Logloss : %f" % log_loss(y_val, fpred_rfc))

lr = LogisticRegression(C=0.3)
lr.fit(a, b)



fpred_xgbo = xgbo.predict(final_pred)
fpred_xgbo_val = xgbo.predict(val)
fpred_xgbo_valc = xgbo.predict(c) # artificial validation set

fpred_rfc = rfc.predict(final_pred)
fpred_rfc_val = rfc.predict(val)
fpred_rfc_valc = rfc.predict(c)

fpred_lr = lr.predict_proba(final_pred)[:,1]
fpred_lr_val = lr.predict_proba(val)[:,1]
fpred_lr_valc = lr.predict_proba(c)[:,1]


#from scipy.stats.mstats import gmean
#fpred = gmean([fpred_xgbo, fpred_rfc, fpred_lr], axis = 0)

def combine(*preds, weights = []):
    fpred = [weight*pred for weight, pred in zip(weights, preds)]
    #0.1*fpred_xgbo+0.7*fpred_rfc+0.2* fpred_lr
    fpred = np.sum(fpred, axis=0)
    return fpred


#feat = list(zip(poly.get_feature_names(), xgbo.feature_importances_))
#feat = sorted(feat, key = lambda x: x[1])

fpred = combine(fpred_xgbo_val, fpred_rfc_val, fpred_lr_val, weights = (0.1, 0.2, 0.7))
fpred = combine(fpred_xgbo, fpred_rfc, fpred_lr, weights = (0.1, 0.2, 0.7))

#0.4, 0.5, 0.1
print ("Logloss : %f" % log_loss(y_val, xgbo.predict(val)))

print ("Logloss : %f" % log_loss(y_val, rfc.predict(val)))

print ("Logloss : %f" % log_loss(y_val, lr.predict_proba(val)))

#print ("Logloss : %f" % log_loss(y_val, svr.predict(val)))

#check consistency, validation dataset
fpred = combine(fpred_xgbo_val, fpred_rfc_val, fpred_lr_val, weights = (0.01, 0.4, 0.59))

df = pd.DataFrame(fpred)
abc = val_ids.join(df.set_index(val_ids.index[:len(df)])).join(y_val)
abc.columns = ["era", "pred","target"]
s=[]
for i  in abc.era.unique():
    subset = abc.loc[abc.era == i]
    s.append(log_loss(subset.target.values, subset.pred.values)+np.log(0.5))
    
eval_results =[]
import itertools as it
heh = it.permutations([a/10 for a in range(10)], 3)
for weights in [a for a in heh if sum(a)==1]:
    fpred = combine(fpred_xgbo_valc, fpred_rfc_valc, fpred_lr_valc, weights = weights)
    df = pd.DataFrame(fpred)
    abc = valc_ids.join(df.set_index(valc_ids.index[:len(df)])).join(d)
    s=[]
    a=[]
    abc.columns = ["era","pred","target"]
    l_l = log_loss(abc.target.values, abc.pred.values)
    for i in abc.era.unique():
        subset = abc.loc[abc.era == i]
        s.append(log_loss(subset.target.values, subset.pred.values)+np.log(0.5))
    eval_results.append((len([s for s in s if s<0]), l_l, s, weights))
    
#feat = list(zip(poly.get_feature_names(), xgbo.feature_importances_))
#print("Running RandomForest model...")
##clf = LassoLars(alpha=0.01)
#rfc = RandomForestRegressor(n_estimators=200, max_features='auto', max_depth=6, verbose = 1)
#rfc.fit(a, b)
##clf.fit(a, b)
#
#print("Predicting...")
#    # Your trained model is now used to make predictions on the numerai_tournament_data
#pred = rfc.predict(c)

#print ("Logloss : %f" % log_loss(d, pred))
##
##print("Running Support Vector Regression...")
##
##svr = svm.SVR()
##svr.fit(a, b)
##print("Predicting...")
##pred = rfc.predict(c)
##print ("Logloss : %f" % log_loss(d, pred))
#
#    # Now you can upload these predictions on numer.ai
##
##if __name__ == '__main__':
##    main()
#
## 
#
#from sklearn.linear_model import LogisticRegression
#
#abc = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='binary:logistic', silent=False)
#svr_rbf = svm.SVR(kernel='rbf')
#
#lr = LogisticRegression()
#
#regressors = [xgbo, rfc]
#stregr = StackingRegressor(regressors=regressors, 
#                           meta_regressor=abc)
#
#stregr.fit(a, b)
#preds = stregr.predict(c)
#print ("Logloss : %f" % log_loss(d, preds))

#ll=[]
##GridSearch(xgbo, a, b)
#for i in range(20):
#    def trainy(X, data_type = None, cv = None, resemble = False):
#        """
#        Method for splitting the dataset into train and validation set.
#        
#        vars
#        data_type: use when importing the tournament data to split val, test, live
#        cv: use for setting up a era-subset validation set. cv is the size of the train set.
#        """
#        if data_type:
#            X = X.loc[X["data_type"]==data_type,:]
#        if cv:
#            rnd = set(np.random.choice(X.era.unique(), cv, replace=False))
#            alt = set(X.era.unique())-rnd
#            traina = X.loc[X["era"].isin(rnd),:] # era subsetting
#            train_val = X.loc[X["era"].isin(alt),:]
#            # only include eras that resemble the val.
#            if resemble:
#                ab = {val: ke for (ke,val) in dict(traina.groupby(by=['era']).target.mean()).items()}
#                hehe = train_val.groupby(by=['era']).target.mean()
#                newrnd = set([ab[key] for key in ab if abs(key-np.asarray(hehe))<0.01])
#                print("We´re here")
#                traina = traina.loc[traina["era"].isin(newrnd),:]
#            
#            y_train = traina["target"]
#            traina.drop(["target"], axis=1, inplace = True)   
#            y_val = train_val["target"]
#            train_val.drop(["target"], axis=1, inplace = True)
#            print("train set: {}".format(sorted(traina.era.unique())))
#            print("test set: {}".format(sorted(list(alt))))
#            return traina, y_train, train_val, y_val
#        
#        y = X["target"]
#        X.drop(["target"], axis=1, inplace = True)  
#        return X, y
#    a, b, c, d = trainy(X, cv=95, resemble=True)
#    d.mean()
#    #kill all rows with low target:
#        
#    #poly = PolynomialFeatures(2)
#    #aa = poly.fit_transform(a)
#    #cc = poly.fit_transform(c)
#    #aa = aa[:,1:]
#    #cc = cc[:,1:]
#    
#    features = [f for f in list(train) if "feature" in f]
#    
#    a = a[features]
#    c = c[features]
#    a["new"] = a["feature10"]*a["feature17"]
#    c["new"] = c["feature10"]*c["feature17"]
#
#    #dtrain = xgb.DMatrix(a, b)
#    #dtest = xgb.DMatrix(c, d)
#    print("Running XGB model...")
#    
#    
#    xgbo = xgb.XGBRegressor(n_estimators=2000, max_depth=5, learning_rate=0.0005, subsample=0.9, colsample_bytree=0.85, objective='binary:logistic', silent=False)
#    xgbo.fit(a, b)
#    pred = xgbo.predict(c)
#    ll.append(log_loss(d, pred))
#    print("{}".format(i))



#results_df = pd.DataFrame(data={'probability':fpred})
#joined = pd.DataFrame(ids).join(results_df)
##
#print("Writing predictions to predictions.csv")
#    # Save the predictions out to a CSV file
#joined.to_csv('C:/Users/Frank/Desktop/kaggle/NUMER.AI/predictions.csv', index=False)