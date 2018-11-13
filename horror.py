import numpy as np
import os
import sklearn
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

def load_and_process(name):
    with open(name,'rb') as file:
        a = file.readlines()
        ret = []
        labels = []
        for ind,i in enumerate(a):
            if ind != 0:
                count = 0
                start = 0
                stop = 0
                labels.append(str(i)[-7:-4])
                for j in range(len(i)):
                    if str(i)[j] == '"' and count == 2:
                        start = j+1
                        count += 1
                    elif str(i)[j] == '"' and count == 3:
                        stop = j-2
                    elif str(i)[j] == '"':
                        count+=1
                ret.append(str(i)[start:stop].lower())
    return ret,labels

def find_words(vect):
    new_vect = []
    to_suppr = []
    for ind1,string in enumerate(vect):
        words = np.zeros((300,))
        oui = 'qwertyuiopasdfghjklzxcvbnm'
        i = 0
        number = 0
        to_add = ''
        print(str(ind1)+"/"+str(len(vect)))
        while i != len(string):
            if string[i] in oui:
                to_add += string[i]
            else:
                if to_add != '' and to_add in voc:
                    for k in range(300):
                        words[k]+=model[to_add][k]
                        number+=1
                to_add = ''
            i+=1
        if number != 0:
            new_vect.append(words/float(number))
        else:
            to_suppr.append(ind1)
    return new_vect

def make_db():
    filename = 'GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    voc = list(model.wv.vocab)

    train, train_labels = load_and_process("train.csv")
    train,suppr = find_words(train)
    new_label = []
    for i in range(len(train_labels)):
        if i not in suppr:
            new_label.append(train_labels[i])
    train_labels = new_label
    file = open("new_db.csv",'w')
    for ind,i in enumerate(train):
        a=''
        for j in i:
            a+=str(j)+','
        a+=train_labels[ind]+"\n"
        file.write(a)
    file.close()

def read_db():
    db = np.zeros((19575,300))
    labels = np.zeros((19575,),dtype='U25')
    with open("new_db.csv",'r') as file:
        a = file.readlines()
        for ind,i in enumerate(a[:19575]):
            line = i.split(',')
            for ind2,j in enumerate(line):
                if ind2 == 300:
                    labels[ind] = j
                else:
                    db[ind,ind2] = float(j)
    return db,labels

#make_db()
db, labels = read_db()

def crossvalFolds(global_set,labels,folds=5,argument=30):
    foldSize = int(labels.shape[0]/folds)
    shuffler = np.arange(global_set.shape[0])
    np.random.shuffle(shuffler)
    new = global_set[shuffler,:]
    newl = labels[shuffler]
    results = []
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
        clf = tree.DecisionTreeClassifier(max_depth =argument)
        clf.fit(train_i,train_label_i)
        results.append(clf.score(test_i,test_label_i))
        print("Progress : "+str(i+1)+"/"+str(folds)+".")
    return results

def gridsearch(array):
    accuracies = []
    for i in array:
        acc = crossvalFolds(db, labels, argument = i)
        acc = sum(acc)/float(len(acc))
        accuracies.append(acc)
    print(accuracies)
    print("Best accuracy : "+str(accuracies.index(max(accuracies)))+" (accuracy : "+str(max(accuracies))+").")
    print("Achieved for max depth = "+str(array[accuracies.index(max(accuracies))]))
    return array[accuracies.index(max(accuracies))]

optimal = gridsearch([3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,75,100])

def analyse(optimal):
    estimator = tree.DecisionTreeClassifier(max_depth =optimal)
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

analyse(optimal)
