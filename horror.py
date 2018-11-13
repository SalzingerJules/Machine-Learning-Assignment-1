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

def find_words(vect, voc, model):
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
    train,suppr = find_words(train, voc, model)
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
    db = np.zeros((19365,300))
    labels = np.zeros((19365,),dtype='U25')
    with open("new_db.csv",'r') as file:
        a = file.readlines()
        for ind,i in enumerate(a[:19365]):
            line = i.split(',')
            for ind2,j in enumerate(line):
                if ind2 == 300:
                    labels[ind] = j[:-1]
                else:
                    db[ind,ind2] = float(j)
    return db,labels

#make_db()
db, labels = read_db()
strings, labels_ = load_and_process("train.csv")
strings = np.asarray(strings)[:19365]
labels_ = np.asarray(labels_)[:19365]
print(np.unique(labels))
for i in range(len(labels)):
    if labels[i] == '':
        print(i)

def private_dictionnaries(data, labels):
    dictionnary = [[],[],[]]
    count = [[],[],[]]
    authors = ['HPL', 'EAP', 'MWS']
    for ind1,string in enumerate(data):
        words = np.zeros((300,))
        oui = 'qwertyuiopasdfghjklzxcvbnm'
        i = 0
        number = 0
        to_add = ''
        while i != len(string):
            if string[i] in oui:
                to_add += string[i]
            else:
                if to_add != '' and to_add not in dictionnary[authors.index(labels[ind1])]:
                    dictionnary[authors.index(labels[ind1])].append(to_add)
                    count[authors.index(labels[ind1])].append(0)
                elif to_add != '':
                    count[authors.index(labels[ind1])][dictionnary[authors.index(labels[ind1])].index(to_add)] = 1
                to_add = ''
            i+=1
    print("Dictonnaries computed !")
    for i in range(3):
        while count[i].count(0):
            ind = count[i].index(0)
            count[i].pop(ind)
            dictionnary[i].pop(ind)
    print("Dictionnaries purified")
    to_suppr = [[],[],[]]
    for ind,i in enumerate(dictionnary[0]):
        found = 0
        if i in dictionnary[1]:
            to_suppr[1].append(dictionnary[1].index(i))
            found = 1
        if i in dictionnary[2]:
            to_suppr[2].append(dictionnary[2].index(i))
            found = 1
        if found:
            to_suppr[0].append(ind)
    for ind,i in enumerate(dictionnary[1]):
        if i in dictionnary[2] and ind not in to_suppr[1]:
            to_suppr[2].append(dictionnary[2].index(i))
            to_suppr[1].append(ind)
    print("Dictionnaries analysed !")
    for i in range(3):
        to_suppr[i].sort()
        for j in to_suppr[i]:
            dictionnary[i].pop(j)
            for k in range(len(to_suppr[i])):
                to_suppr[i][k] -= 1
    print("Dictionnaries pruned !")
    return dictionnary

def dictionnary_voting(dictionnary,string):
    oui = 'qwertyuiopasdfghjklzxcvbnm'
    i = 0
    votes = [0,0,0]
    to_add = ''
    while i != len(string):
        if string[i] in oui:
            to_add += string[i]
        else:
            if to_add in dictionnary[0]:
                votes[0]+=1
            if to_add in dictionnary[1]:
                votes[1]+=1
            if to_add in dictionnary[2]:
                votes[2]+=1
            to_add = ''
        i+=1
    return votes

class ensemble():
    def __init__(self,number=100, argument=7, components = 5,use_dict = True):
        self.number = number
        self.argument = argument
        self.models = []
        self.weights = []
        self.components = components
        self.use_dict = use_dict
    def fit(self,data, labels):
        for i in range(self.number):
            print(i)
            choice = np.arange(300)
            self.choice = np.random.choice(choice, 250)
            pca = PCA(n_components=self.components)
            ndata = pca.fit_transform(data[:,self.choice])
            esti = tree.DecisionTreeClassifier(max_depth =self.argument)
            esti.fit(ndata,labels)
            self.weights.append(esti.score(ndata,labels))
            self.models.append(esti)
    def score(self,data,strings,labels,dictionnary):
        prediction = []
        score = 0
        total = 0
        totalvotes = 0
        pca = PCA(n_components=self.components)
        ndata = pca.fit_transform(data[:,self.choice])
        for ind,sample in enumerate(ndata):
            votes = [0,0,0]
            authors = ['HPL','EAP','MWS']
            for i in range(self.number):
                predic = self.models[i].predict([sample])[0]
                if predic in authors:
                    votes[authors.index(predic)]+=(self.weights[i]*10)**3
            if self.use_dict:
                vote_mult = dictionnary_voting(dictionnary,strings[ind])
                for i in range(3):
                    votes[i] = votes[i]*(vote_mult[i]+1)
            if votes != [0,0,0]:
                if authors[votes.index(max(votes))]==labels[ind]:
                    score += 1
                total+=1
        return float(score)/float(total)

def crossvalFolds(global_set,labels,strings,labels_,folds=5,argument=30, components = 7, number = 10):
    foldSize = int(labels.shape[0]/folds)
    shuffler = np.arange(global_set.shape[0])
    np.random.shuffle(shuffler)
    new = global_set[shuffler,:]
    newl = labels[shuffler]
    news = strings[shuffler]
    newls = labels_[shuffler]
    results = []
    for i in range(folds):
        test_i = new[i*foldSize:(i+1)*foldSize,:]
        test_label_i = newl[i*foldSize:(i+1)*foldSize]
        test_strings_i = news[i*foldSize:(i+1)*foldSize]
        test_labels_strings_i = newls[i*foldSize:(i+1)*foldSize]
        if i == 0:
            train_i = new[(i+1)*foldSize:,:]
            train_label_i = newl[(i+1)*foldSize:]
            train_strings_i = news[(i+1)*foldSize:]
            train_labels_strings_i = newls[(i+1)*foldSize:]
        elif i == folds:
            train_i = new[:i*foldSize,:]
            train_label_i = newl[:i*foldSize]
            train_strings_i = news[:i*foldSize]
            train_labels_strings_i = newls[:i*foldSize]
        else:
            train_i = np.concatenate((new[(i+1)*foldSize:,:],new[:i*foldSize,:]))
            train_label_i = np.concatenate((newl[(i+1)*foldSize:],newl[:i*foldSize]))
            train_strings_i = np.concatenate((news[(i+1)*foldSize:],news[:i*foldSize]))
            train_labels_strings_i = np.concatenate((newls[(i+1)*foldSize:],newls[:i*foldSize]))
        #clf = tree.DecisionTreeClassifier(max_depth =argument)
        dictionnary = private_dictionnaries(train_strings_i, train_labels_strings_i)
        clf = ensemble(number =number, argument = argument, components = components)
        clf.fit(train_i,train_label_i)
        results.append(clf.score(test_i, test_strings_i,test_label_i ,dictionnary))
        print("Progress : "+str(i+1)+"/"+str(folds)+".")
    return results

def gridsearch(array1, array2):
    accuracies_t = []
    for i in array1:
        accuracies = []
        for j in array2:
            acc = crossvalFolds(db, labels, argument = i, components = j, number = 10)
            acc = sum(acc)/float(len(acc))
            accuracies.append(acc)
        accuracies_t.append(accuracies)
    for i in accuracies_t:
        print(i)
    print("Best accuracy : "+str(max([max(i) for i in accuracies_t]))+").")
    print("Achieved for max depth = "+str(array1[accuracies_t.index(max(accuracies_t, key=lambda x:max(x)))]))
    print("Achieved for components = "+str(array2[max(accuracies_t, key=lambda x:max(x)).index(max(max(accuracies_t, key=lambda x:max(x))))]))
    return array1[accuracies_t.index(max(accuracies_t, key=lambda x:max(x)))],array2[max(accuracies_t, key=lambda x:max(x)).index(max(max(accuracies_t, key=lambda x:max(x))))]

#optimal,components = gridsearch([3,4,5,6,7,8,9], [3,4,5,6,7,8,9])
optimal = 4
components = 3
results = crossvalFolds(db, labels, strings, labels_, argument = optimal, components = components)
print("Mean accuracy :"+str(sum(results)/len(results))+"%.")

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

#analyse(optimal)

