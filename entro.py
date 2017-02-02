from collections import Counter
from math import log
from china import Classifier,DataSet,DataInstance,InstanceType,run_test,run_crossfold_test
from datasets import CarDataSet,IrisDataSet,LensesDataSet,VotingDataSet,discretize

def calcEntropyBranch(targets):
    target_counter = Counter(targets)
    total = sum(target_counter.values())
    entropy_sum = 0
    #print(total)
    for target in target_counter.keys():
    #    print("{} -> {count}".format(target,count=target_counter[target]))
        count = target_counter[target]
        entropy_sum -= count*log(float(count)/total,2)
    return entropy_sum

def calcEntropyForFeature(data,feature_index:int):
    target_values = [datum.target for datum in data]
    feature_values = [datum.feature[feature_index] for datum in data]
    feature_set = set(feature_values)
    feature_dict = {x:[] for x in feature_set}
    entropy = 0.0
   # print(feature_dict)
    for datum in data:
        feature_dict[datum.feature[feature_index]].append(datum)
    for branch in feature_dict.values():
        entropy += calcEntropyBranch([x.target for x in branch])
    return (entropy/len(data),feature_dict)

def build_tree(data,possible_features):
    """ Recursive tree building
    """
    #Check if it is already complete
    target_data = [d.target for d in data]
    min_entropy = calcEntropyBranch(target_data)
    if min_entropy == 0:
        return -1,data[0].target,None
    elif len(possible_features) == 0:
        return -1, Counter(target_data).most_common(1)[0][0],None # Returns most common target
    chosen_feature = -1
    chosen_tree = None
    for x in possible_features:
        entropy, tree = calcEntropyForFeature(data,x)
        if chosen_feature == -1 or entropy < min_entropy:
            min_entropy = entropy
            chosen_tree = tree
            chosen_feature = x
    ret_tree = {}
   # print(possible_features)
   # print(chosen_feature)
    possible_features = list(possible_features)
    possible_features.remove(chosen_feature)
    for x in chosen_tree.keys():
        ret_tree[x] = build_tree(chosen_tree[x],possible_features)
    return chosen_feature,ret_tree, Counter(target_data).most_common(1)[0][0]
    
class ID3Classifier(Classifier):
    def __init__(self):
       # super().__init__(filename)
        self.tree = {}
        self.data = []
        self.root_feature = -1
        self.default = ""

    def fit(self,dataset):
        self.data.extend(dataset.data)
        self.root_feature, self.tree,self.default = build_tree(
            self.data,list(range(len(self.data[0].feature))))
        pass

    def predict_instance(self,instance):
        feat = self.root_feature
        node = self.tree
        default = self.default
        while feat != -1:
            feat,node,default = node.get(instance.feature[feat],(-1,default,None))
        return node

    def print_tree(self):
        print_tree(0,self.root_feature,self.tree,self.data[0].feature_names)

def print_tree(level,feature,node,feature_names):
    if feature == -1:
        print("[{}]".format(node))
    else:
        print("({})".format(feature_names[feature]))
        level += 1
        for k,v in sorted(node.items()):
            print("  "*level + "-{:<6} ".format("{}:".format(k)),end="")
            print_tree(level,v[0],v[1],feature_names)

if "__main__" == __name__:
    # data = CarDataSet("datasets/car.data")
    # classifier, acc_matrix = run_crossfold_test(data,ID3Classifier)
    # classifier.print_tree()
    iris_data = IrisDataSet("datasets/iris.data")
    d_iris_data = discretize(iris_data)
    lenses_data = LensesDataSet("datasets/lenses.data")
    voting_data = VotingDataSet("datasets/house-votes-84.data")
    print("\nRunning n-fold validation on Iris data")
    d_iris_classifer,d_iris_matrix = run_crossfold_test(d_iris_data,ID3Classifier)
    print("\nRunning n-fold validation on Lenses data")
    lenses_classifier,lenses_matrix = run_crossfold_test(lenses_data,ID3Classifier)
    print("\nRunning n-fold validation on Voting data")
    voting_classifier,voting_matrix = run_crossfold_test(voting_data,ID3Classifier)
    print("\nIris decision tree")
    d_iris_classifer.print_tree()
    # print(iris_data.data[0].data)
    # print(d_iris_data.data[0].data)
    print("\nLenses decision tree(See documentation for number interpretation)")
    lenses_classifier.print_tree()
    print("\nVoting decision tree")
    voting_classifier.print_tree()
