import numpy as np
import os
import sklearn
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

def load_and_process():
    with open("bank-additional-full.csv",'r') as file:
        a = file.readlines()
        ret = np.zeros((41189,20),dtype='U25')
        labels = np.zeros((41189,),dtype='U25')
        for ind,i in enumerate(a):
            if ind != 0:
                line = i.split(';')
                for j in range(len(line)):
                    if line[j][0] == '"':
                        line[j] = line[j][1:-1]
                ret[ind] = np.asarray(line[:-1])
                labels[ind] = np.asarray(line[-1])
    return ret[1:41186],labels[1:41186]
total,labels = load_and_process()
numerical = total[:,[0,10,11,12,13,15,16,17,18,19]].astype(np.float64)
enc = OneHotEncoder()
db = np.concatenate([enc.fit_transform(total[:,[1,2,3,4,5,6,7,8,9,14]]).toarray(),numerical],axis = 1)
features = enc.get_feature_names().tolist()+['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
for ind,i in enumerate(features):
    if i.startswith("x0"):
        features[ind] = 'job'+i[2:]
    if i.startswith("x1"):
        features[ind] = 'marital'+i[2:]
    if i.startswith("x2"):
        features[ind] = 'education'+i[2:]
    if i.startswith("x3"):
        features[ind] = 'default'+i[2:]
    if i.startswith("x4"):
        features[ind] = 'housing'+i[2:]
    if i.startswith("x5"):
        features[ind] = 'loan'+i[2:]
    if i.startswith("x6"):
        features[ind] = 'contact'+i[2:]
    if i.startswith("x7"):
        features[ind] = 'month'+i[2:]
    if i.startswith("x8"):
        features[ind] = 'day_of_week'+i[2:]
    if i.startswith("x9"):
        features[ind] = 'poutcome'+i[2:]
indexes = [i for i in range(63) if i not in [features.index(i) for i in features if i.endswith("no")]]
features = np.asarray(features)
db = db[:,indexes]

def crossvalFolds(global_set,labels,folds=5,argument=30):
    foldSize = int(labels.shape[0]/folds)
    shuffler = np.arange(global_set.shape[0])
    #np.random.shuffle(shuffler)
    new = global_set[shuffler,:]
    newl = labels[shuffler]
    results = []
    confusion = [0,0,0,0]
    for i in range(folds):
        test_i = new[i*foldSize:(i+1)*foldSize,:]
        test_label_i = newl[i*foldSize:(i+1)*foldSize]
        if i == 0:
            train_i = new[(i+1)*foldSize:,:]
            train_label_i = newl[(i+1)*foldSize:]
        elif i == folds:
            train_i = new[:i*foldSize,:]
            train_label_i = newl[:i*foldSize]
        else:
            train_i = np.concatenate((new[(i+1)*foldSize:,:],new[:i*foldSize,:]))
            train_label_i = np.concatenate((newl[(i+1)*foldSize:],newl[:i*foldSize]))
        clf = tree.DecisionTreeClassifier(max_depth =argument,class_weight={'no"':1.126956,'yes"':8.8767724})
        clf.fit(train_i,train_label_i)
        result = clf.predict(test_i)
        accuracy = 0
        total = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for ind,sample in enumerate(test_label_i):
            if sample == 'yes"':
                total+=8.876724
                if sample == result[ind]:
                    accuracy+=8.876724
                    TP += 1
                else:
                    FP += 1
            if sample == 'no"':
                total+=1.126956
                if sample == result[ind]:
                    accuracy+=1.126956
                    TN +=1
                else:
                    FN += 1
        #results.append(clf.score(test_i,test_label_i))
        results.append(accuracy/total)
        confusion[0] += TP
        confusion[1] += FP
        confusion[2] += TN
        confusion[3] += FN
        #print("Progress : "+str(i+1)+"/"+str(folds)+".")
    return results,confusion

results_glob = []
for comp in [2,3,4,5,6]:
    pca = PCA(n_components=comp)
    new_db = pca.fit_transform(db)
    for arg in [2,3,4,5,6,7,8,9]:
        results,confusion = crossvalFolds(new_db,labels,argument =arg)
        results_glob.append(sum(results)/len(results))
print(results_glob)
print(results_glob.index(max(results_glob)))
#PCA 3 MAX DEPTH 3
    print("")
    print("MEAN ACCURACY OVER FOLDS : "+str(sum(results)/len(results))+"% for maximum depth = "+str(arg)+".")
    print("Accuracy for positives : "+str(confusion[0]/(confusion[0]+confusion[3]))+"% for maximum depth = "+str(arg)+".")
    print("Accuracy for negatives : "+str(confusion[2]/(confusion[2]+confusion[1]))+"% for maximum depth = "+str(arg)+".")
    #print(str(confusion[0])+" "+str(confusion[1]))
    #print(str(confusion[2])+" "+str(confusion[3]))

pca = PCA(n_components=3)
new_db = pca.fit_transform(db)
print(pca.explained_variance_ratio_)
tests = pca.components_.tolist()
for ind,i in enumerate(pca.components_):
    for j in range(len(i)):
        if i[j]<0:
            tests[ind][j] = (-i[j],features[j])
        else:
            tests[ind][j] = (i[j],features[j])
    tests[ind].sort(key=lambda x:-x[0])
    print(tests[ind][:4])
estimator = tree.DecisionTreeClassifier(max_depth =3,class_weight={'no"':1.126956,'yes"':8.8767724})
estimator.fit(new_db,labels)
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
print(feature)
threshold = estimator.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if node_depth[i]<5:
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print(node_depth[i] * "\t"+str(feature[i])+"<="+str(threshold[i]))


