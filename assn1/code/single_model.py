import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import json 

import pickle

train = np.loadtxt( "train.dat" )

def createFeatures( X ):
    return X

# data_trn, data_tst = train_test_split( train, train_size = 10000 )
X_trn = train[:, :64]
p = train[:,64:68]
q = train[:,68:72]
Y_trn = train[:,-1]

model = LinearSVC( loss = "hinge" )
# models = [[LogisticRegression()]*16]*16

train_dict = {}

# for a in range(16):
#     for b in range(16):
#         if a==b:
#             continue
#         train_mat = np.array([])
#         for i in range(train.shape[0]):
#             p_dec = 8*int(p[i,0]) + 4*int(p[i,1]) + 2*int(p[i,2]) + int(p[i,3])
#             q_dec = 8*int(q[i,0]) + 4*int(q[i,1]) + 2*int(q[i,2]) + int(q[i,3])
#             if p_dec == a and q_dec == b:

#                 crp  = (np.append(X_trn[i],np.array([Y_trn[i]])))
#                 crp = crp.reshape(1,-1)
#                 if train_mat.size == 0:
#                     train_mat = crp
#                 else:
#                     train_mat = np.append(train_mat, crp, axis=0)
        
#         train_dict[(a,b)] = train_mat


# # dicti = dict([("%d,%d" % k, str(v.tolist())) for k, v in train_dict.items()])
# # with open("sample.json", "w") as outfile:
# #     json.dump(dicti, outfile, indent = 4)
# with open('dict_file.pickle', 'wb') as dict_file:
#     pickle.dump(train_dict, dict_file)

with open('dict_file.pickle', 'rb') as handle:
    train_dict = pickle.load(handle)
print('train_dict= ',train_dict)

idx1 = 11
idx2 = 15

train_data = train_dict[(idx1,idx2)]
model.fit(createFeatures(train_data[:,:64]), train_data[:,-1])

test_data = np.loadtxt("test.dat")
X_tst = test_data[:, :64]
p = test_data[:,64:68]
q = test_data[:,68:72]
Y_tst = test_data[:,-1]

pred = np.array([])
Y_actual = np.array([])
for i in range(test_data.shape[0]):
    p_dec = 8*int(p[i,0]) + 4*int(p[i,1]) + 2*int(p[i,2]) + int(p[i,3])
    q_dec = 8*int(q[i,0]) + 4*int(q[i,1]) + 2*int(q[i,2]) + int(q[i,3])
    if p_dec == idx1 and q_dec == idx2:
        prediction = model.predict(createFeatures(X_tst[i].reshape(1,-1)))
        pred = np.append(pred, prediction)
        Y_actual = np.append(Y_actual, Y_tst[i])
    
print(np.average(pred == Y_actual))
