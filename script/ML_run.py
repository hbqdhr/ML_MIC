import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
import math
import os;
import pickle
cwd = os.getcwd()



X = pd.read_csv(cwd + "/example/new_data.txt", header=0, delim_whitespace=True)
mymap = {'A':1, 'T':2, 'G':3, 'C':4}
X_new = X.applymap(lambda s: mymap.get(s) if s in mymap else s)

X_new = X_new.ix[0,:].values
X_new.astype(int)

for i in range(len(X_new)):
    if not isinstance(X_new[i],np.int64) :
            print(i)
            print(X_new[i])
            X_new[i]=5




# load the model from disk
pen_model = pickle.load(open(cwd + '/database/pen_model.sav', 'rb'))
pen_pred = pen_model.predict(X_new.reshape(1,-1))

amo_model = pickle.load(open(cwd + '/database/amo_model.sav', 'rb'))
amo_pred = amo_model.predict(X_new.reshape(1,-1))

mer_model = pickle.load(open(cwd + '/database/mer_model.sav', 'rb'))
mer_pred = mer_model.predict(X_new.reshape(1,-1))

tax_model = pickle.load(open(cwd + '/database/tax_model.sav', 'rb'))
tax_pred = tax_model.predict(X_new.reshape(1,-1))

cft_model = pickle.load(open(cwd + '/database/cft_model.sav', 'rb'))
cft_pred = cft_model.predict(X_new.reshape(1,-1))

cfx_model = pickle.load(open(cwd + '/database/cfx_model.sav', 'rb'))
cfx_pred = cfx_model.predict(X_new.reshape(1,-1))



# save result
value = [math.pow(2,np.round(pen_pred)), math.pow(2,np.round(amo_pred)), math.pow(2,np.round(mer_pred)), math.pow(2,np.round(tax_pred)), math.pow(2,np.round(cft_pred)), math.pow(2,np.round(cfx_pred))]

valuemap = {0.125:0.12, 0.0625:0.06, 0.03125:0.03}
f = lambda x: valuemap.get(x) if x in valuemap else x
value_new = list(map(f, value))

res = [['pen','amo','mer','tax','cft','cfx'], value_new]

print(res)
with open(cwd + '/example/prediction_res.txt', 'w') as file:
    file.writelines('\t'.join(str(j) for j in i) + '\n' for i in res)





