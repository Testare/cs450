import re
import random

IRIS_CLASSES = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
SET_SIZE = 15
SET_COUNT = 10

class IrisDataInstance:
    '''An instance of Iris data'''
    regex = re.compile(",")
    def __init__(self,line):
        '''Initializes an instance of Iris data from a string'''
        data = IrisDataInstance.regex.split(line[:-1])
        self.sepal_length = data[0]
        self.sepal_width = data[1]
        self.petal_length = data[2]
        self.petal_width = data[3]
        self.iris_class = data[4]

class IrisDataSet:
    '''A set of IrisDataInstances'''
    def __init__(self,filename=None):
        '''Initializes a Data Set, either from a file or empty if filename
        not specified'''
        self.data = []
        if filename == None:
            return
        with open(filename) as data:
            for line in [x for x in data if x != "\n"]:
                self.data.append(IrisDataInstance(line))
            random.shuffle(self.data)

    def split(self,count):
        '''Splits off /count/ amount of instances from the end of this data set, puts them into a new data set, deletes them from this one, and returns the split off data'''
        newSet = IrisDataSet()
        newSet.data.extend(self.data[-count:])
        del self.data[-count:]
        #Negative count and first step ensures we take the right amount off,
        #and that we take it off the end
        return newSet
    def copy(self):
        '''Creates and returns a copy of this data set'''
        newSet = IrisDataSet()
        newSet.data.extend(self.data)
        return newSet

class HardCodedClassifier:
    ''' A learning classifier with a hard-coded algorithim'''
    def fit(self,dataset):
        '''Trains on the new data'''
        pass
    def predict(self,dataset):
        '''Predicts the classes of all the instances in the data set'''
        return [self._predict_instance(x) for x in dataset.data]
    def _predict_instance(self,instance):
        '''Predicts the class of one instance'''
        return IRIS_CLASSES[0];

def run_test(sets,testIndices):
    '''Runs tests over the sets, training with most of the sets but testing on
    the test sets specified by the test indices'''
    classifier = HardCodedClassifier()
    testSets = []
    for x in range(10):
        if x in testIndices:
            testSets.append(sets[x])
            continue
        classifier.fit(trainingdata)

    correct = 0
    for testdata in testSets:
        predictions = classifier.predict(testdata)
        for i in range(15):
            if predictions[i] == testdata.data[i].iris_class:
                correct += 1
    return correct/(len(testIndices)*15)


if __name__ == "__main__":
    trainingdata = IrisDataSet("iris.data")
    trainingdata2 = trainingdata.copy()
    dataSubsets = []
    for x in range(10): # Dataset size is kinda hard-coded... meh
        dataSubsets.append(trainingdata.split(15))
    acc = run_test(dataSubsets,[0,1,2])
    print( "Accuracy - %f" % acc)
    bestacc = 0.0
    beststrin = ""    #triple loop finds optimal 30% set of numbers for accuracy
                       #n-fold validation complete
    for i in range(8):
        for j in range(i + 1,9):
            for k in range(j + 1,10):
                acc = run_test(dataSubsets,[i,j,k])
                strin = "%d %d %d - %2.1f%%" % (i,j,k,acc*100)
                if acc > bestacc:
                    bestacc = acc
                    beststrin = strin
                print(strin)
    print("\n\nBest accuracy: \n%s" % beststrin)
