from collections import Counter
from math import log
from china import Classifier,DataSet,DataInstance,InstanceType,run_test

class CarDataInstance(DataInstance):
    buying = {'vhigh':3,'high':6,'med':9,'low':12}
    maint = buying
    doors = {'2':3,'3':6,'4':9,'5more':12}
    persons = {'2':4,'4':8,'more':12}
    lug_boot = {'small':4,'med':8,'big':12}
    safety = {'low':4,'med':8,'high':12}
    attr = [buying,maint,doors,persons,lug_boot,safety]
    feature_names = ["buying","maint","doors","persons","lug_boot","safety"] 

    def __init__(self,line):
        super().__init__(line)
        self.feature_names = CarDataInstance.feature_names

    def compare(self,other):
        d = 0.0
        for i in range(6):
            d += (CarDataInstance.attr[i][self.data[i]] - CarDataInstance.attr[i][other.data[i]])**6


@InstanceType(CarDataInstance)
class CarDataSet(DataSet):
    pass



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
        return -1,data[0].target
    elif len(possible_features) == 0:
        return -1, Counter(target_data).most_common(1)[0][0] # Returns most common target
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
    return chosen_feature,ret_tree
    
class ID3Classifier(Classifier):
    def __init__(self):
       # super().__init__(filename)
        self.tree = {}
        self.data = []
        self.root_feature = -1

    def fit(self,dataset):
        self.data.extend(dataset.data)
        self.root_feature, self.tree = build_tree(
            self.data,list(range(len(self.data[0].feature))))
        
        #//Build tree
        pass

    def predict_instance(self,instance):
        feat = self.root_feature
        node = self.tree
        while feat != -1:
            print(feat)
            print(node.keys())
            feat,node = node[instance.feature[feat]]
        return node

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
#    print(calcEntropyBranch(["A","A","A","B","A","B","C"]))
#    print(calcEntropyBranch(["A","A"]))
#    print(calcEntropyBranch(["A","C"]))
    data = CarDataSet("datasets/car.data")
#    entropy, tree = calcEntropyForFeature(data.data,1)
#    print(entropy)
#    entropy, tree = calcEntropyForFeature(next(iter(tree.values())),1)
#    print(entropy)
    root_feature, tree = build_tree(data.data,list(range(len(data.data[0].feature))))
#    print(tree)
    print_tree(0,root_feature,tree,data.data[0].feature_names)
    #1728 data points
    data2 = data.split(480)
    acc = run_test(ID3Classifier(),[data,data2],[1])
    print("acc: {}".format(acc))
