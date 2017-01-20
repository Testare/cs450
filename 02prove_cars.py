from china import DataSet,DataInstance,InstanceType,run_crossfold_test,Classifier
import random
from collections import Counter
class CarDataInstance(DataInstance):
    buying = {'vhigh':3,'high':6,'med':9,'low':12}
    maint = buying
    doors = {'2':3,'3':6,'4':9,'5more':12}
    persons = {'2':4,'4':8,'more':12}
    lug_boot = {'small':4,'med':8,'big':12}
    safety = {'low':4,'med':8,'high':12}
    attr = [buying,maint,doors,persons,lug_boot,safety]
    def compare(self,other):
        d = 0.0
        for i in range(6):
            d += (CarDataInstance.attr[i][self.data[i]] - CarDataInstance.attr[i][other.data[i]])**6


@InstanceType(CarDataInstance)
class CarDataSet(DataSet):
    pass

class KNNClassifier(Classifier):
    def __init__(self,kNeighbors = 1):
        #super().__init()
        self.kNeighbors = kNeighbors
        self.data = []

    def fit(self,dataset):
        self.data.extend(dataset.data)

    def predict_instance(self,instance):
        disttargets = {}
        for datainstance in self.data:
            disttargets[instance.compare(datainstance)] = datainstance
        common = sorted(disttargets.keys())[:self.kNeighbors]
        counter = Counter([disttargets[x].target for x in common])
        return counter.most_common(1)[0][0]


if __name__ == "__main__":
    data = CarDataSet("datasets/car.data")
    classifier = KNNClassifier(3)
    run_crossfold_test(classifier,data) 
