import re
import random
import math

def TargetColumn(num):
    '''A decorator that specifies which column of a data instance line is the target. Default is -1'''
    def change_num(cls):
        cls.target_column = num
        return cls
    return change_num

def InstanceType(instance_class):
    '''A decorate for the DataSet class that specifies the class of the DataInstances it contains'''
    def change_class(cls):
        cls.instance_class = instance_class
        return cls
    return change_class

class DataInstance:
    '''An instance of data'''
    regex = re.compile(",")
    target_column = -1
    def __init__(self,line):
        '''Initializes an instance of Iris data from a string'''
        self.data = self.regex.split(line[:-1])
        self.target = self.data[self.target_column]
        del self.data[self.target_column]

    def compare(self,other):
        '''Returns the distance between two data instances. Can and probably should be overwritten'''
        return sum(map(lambda x,y:(float(x)- float(y))**2,self.data,other.data))


class DataSet:
    '''A set of DataInstances'''

    instance_class = DataInstance

    def __init__(self,filename=None):
        '''Initializes a Data Set, either from a file or empty if filename
        not specified'''
        self.data = []
        if filename == None:
            return
        with open(filename) as data:
            for line in [x for x in data if x != "\n"]:
                self.data.append(self.instance_class(line))
            random.shuffle(self.data)

    def split(self,count):
        '''Splits off /count/ amount of instances from the end of this data set, puts them into a new data set, deletes them from this one, and returns the split off data'''
        newSet = self.__class__()
        newSet.data.extend(self.data[-count:])
        del self.data[-count:]
        return newSet

    def copy(self):
        '''Creates and returns a copy of this data set'''
        newSet = self.__class__() 
        newSet.data.extend(self.data)
        return newSet

class Classifier:
    ''' A learning classifier with a hard-coded algorithim'''
    def fit(self,dataset):
        '''Trains on the new data'''
        pass
    def predict(self,dataset):
        '''Predicts the classes of all the instances in the data set'''
        return [self.predict_instance(x) for x in dataset.data]
    def predict_instance(self,instance):
        '''Predicts the class of one instance'''
        return 'Iris-setosa';

class CrossMatrix:
    '''tbcompleted'''

def run_test(classifier,sets,testIndices):
    '''Runs tests over the sets, training with most of the sets but testing on the test sets specified by the test indices'''
    testSets = []
    for x in range(len(sets)):
        if x in testIndices:
            testSets.append(sets[x])
            continue
        classifier.fit(sets[x])
    correct = 0
    attempts = 0
    for testdata in testSets:
        predictions = classifier.predict(testdata)
        for i in range(len(testdata.data)):
            if predictions[i] == testdata.data[i].target:
                correct += 1
            attempts += 1
    return correct/attempts



def run_crossfold_test(classifier,trainingdata):
    dataSubsets = []
    increment = len(trainingdata.data)/float(10)
    baseAmount = 0.0;
    for x in range(9): # Dataset size is kinda hard-coded... meh
        baseAmount += increment
        dataSubsets.append(trainingdata.split(math.floor(baseAmount)))
        baseAmount -= math.floor(baseAmount)
    dataSubsets.append(trainingdata)
    avgacc = 0
    iters = 0
    bestacc = 0.0
    beststrin = ""    #triple loop finds optimal 30% set of numbers for accuracy
                       #n-fold validation complete
    for i in range(8):
        for j in range(i + 1,9):
            for k in range(j + 1,10):
                acc = run_test(classifier,dataSubsets,[i,j,k])
                strin = "%d %d %d - %2.1f%%" % (i,j,k,acc*100)
                if acc > bestacc:
                    bestacc = acc
                    beststrin = strin
                avgacc += acc
                iters += 1
                print(strin)
    print("\n\nBest accuracy: \n%s" % beststrin)
    print("\nAverage accuracy: \n%2.1f%%" % (float(100*avgacc)/iters))

if __name__ == "__main__":
    trainingdata = DataSet("iris.data")
    classifier = Classifier()
    run_crossfold_test(classifier,trainingdata)
