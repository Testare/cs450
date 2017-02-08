import re
import random
import math
from functools import reduce


#Decorators

def TargetColumn(num):
    '''A decorator that specifies which column of a data instance line is the target. Default is -1'''
    def change_num(cls):
        cls.target_column = num
        return cls
    return change_num

def IgnoreColumns(*cols):
    def change_num(cls):
        cls.ignore_columns = cols
        return cls
    return change_num

def InstanceType(instance_class):
    '''A decorate for the DataSet class that specifies the class of the DataInstances it contains'''
    def change_class(cls):
        cls.instance_class = instance_class
        return cls
    return change_class

#Classes

class DataInstance:
    '''An instance of data'''
    regex = re.compile(",")
    target_column = -1
    ignore_columns = []
    feature_names = None
    def __init__(self,line=None,data=None,target=None):
        '''Initializes an instance of Iris data from a string'''
        if line != None:
            self.data = self.regex.split(line[:-1])
            self.target = self.data[self.target_column]
            del self.data[self.target_column]
            for x in reversed(sorted(self.ignore_columns)):
                del self.data[x]
        elif data != None and target != None:
            self.data = data
            self.target = target
        self.feature = self.data
        self.feature_names = self.__class__.feature_names
        if self.feature_names == None:
            self.feature_names = ["unnamed"] * (len(self.feature)+1)

    def compare(self,other):
        '''Returns the distance between two data instances. Can and probably should be overwritten'''
        return sum(map(lambda x,y:(float(x)- float(y))**2,self.data,other.data))
    def discretize(self):
        pass

    def copy(self):
        return self.__class__(data=self.data,target=self.target)

class DataSet:
    '''A set of DataInstances'''

    instance_class = DataInstance

    def __init__(self,filename=None,data_set=[]):
        '''Initializes a Data Set, either from a file or empty if filename
        not specified'''
        self.data = data_set.copy()
        if filename == None:
            return
        with open(filename) as data:
            for line in [x for x in data if x != "\n"]:
                self.data.append(self.instance_class(line))
            random.shuffle(self.data)

    def __iter__(self):
        class DataIterator:
            def __init__(self,iterate):
                self.i = -1
                self.it = iterate
                self.end = len(iterate.data)

            def __next__(self) -> DataInstance:
                self.i += 1
                if self.i == self.end:
                    raise StopIteration
                return self.it.data[self.i]
        return DataIterator(self)

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

    def discretize(self):
        for d in self.data:
            d.discretize()


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
    def __init__(self):
        self.att_list = []
        self.matrix = {}
        self.size = 0
        self._acc = -1

    def update(self,prediction,target):
        if prediction not in self.att_list:
            self._new_att(prediction)
        if target not in self.att_list:
            self._new_att(target)
        self.matrix[target][prediction] += 1
        self.size += 1
        self._acc = -1

    def _new_att(self,att):
        self.att_list.append(att)
        for row in self.matrix.values():
            row[att] = 0
        self.matrix[att] = {row:0 for row in self.att_list}

    @property
    def accuracy(self):
        if (self.size == 0):
            return 0.0
        if (self._acc == -1):
            self._acc = float(sum([self.matrix[x][x] for x in self.att_list])) \
            / float(self.size)
        return self._acc

    def __str__(self):
        self.att_list = sorted(self.att_list)
        retstr = "Rows:Targets\nCols:Prediction\n[{}]\n".format(reduce(lambda x,y: x + ", " + y,self.att_list))
        for x in self.att_list:
            for y in self.att_list:
                retstr += "|{:^5}".format(self.matrix[x][y])
            retstr += "| T:{}\n".format(x)
        return retstr
        return "{}\n{}\n".format(str(self.att_list),str(self.matrix))
class NormalizedDataInstance(DataInstance):
    def __init__(self,*args,**options):
        super().__init__(*args,**options)
        # self.feature_names = class_.feature_names
    pass

@InstanceType(NormalizedDataInstance)
class NormalizedDataSet(DataSet):
    def __init__(self,data_instance_class,*args,**options):
        super().__init__(*args,**options)
        self.data_instance_class = data_instance_class
        self.instance_class = data_instance_class


def run_test(classifier,sets,testIndices):
    '''Runs tests over the sets, training with most of the sets but testing on the test sets specified by the test indices'''
    test_set = []
    training_set = []
    for x in range(len(sets)):
        if x in testIndices:
            test_set.extend(sets[x].data)
        else:
            training_set.extend(sets[x].data)

    classifier.fit(DataSet(data_set=training_set))
    correct = 0
    attempts = 0
    predictions = classifier.predict(DataSet(data_set=test_set))
    matrix = CrossMatrix()
    for i in range(len(test_set)):
        if predictions[i] == test_set[i].target:
            correct += 1
        matrix.update(predictions[i],test_set[i].target)
        attempts += 1
    return matrix


def run_crossfold_test(trainingdata,classifier_class,*classifier_args):
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
    bestclassifier = None
    bestacc = CrossMatrix()
    beststrin = ""    #triple loop finds optimal 30% set of numbers for accuracy
                       #n-fold validation complete
    for i in range(8):
        for j in range(i + 1,9):
            for k in range(j + 1,10):
                classifier = classifier_class(*classifier_args)
                acc = run_test(classifier,dataSubsets,[i,j,k])
                strin = "%d %d %d - %2.1f%%" % (i,j,k,acc.accuracy*100)
                if acc.accuracy > bestacc.accuracy:
                    bestacc = acc
                    beststrin = strin
                    bestclassifier = classifier
                avgacc += acc.accuracy
                iters += 1
                print(strin)
    print("\n\nBest accuracy: \n%s" % beststrin)
    print("\nAverage accuracy: \n%2.1f%%" % (float(100*avgacc)/iters))
    print("\nBest matrix: \n{}".format(bestacc))
    return bestclassifier,bestacc


# Functions


def normalized(dataset:DataSet):
    x_min = dataset.data[0].data
    x_max = x_min
    for y in dataset.data[1:]:
        x_min = map(min,x_min,y.data)
        x_max = map(max,x_max,y.data)
    x_min = list(x_min)
    x_max = [xax - xin for xin,xax in zip(x_min,x_max)]
    # targets = [datum.target for datum in dataset]
    data_values = [NormalizedDataInstance(
                    data=[((val - xin)/x_norm) for val, xin, x_norm
                     in zip(x.data, x_min, x_max)],
                    target=x.target)
                    for x in dataset]

    d_set = NormalizedDataSet(data_instance_class=dataset.instance_class,
                              data_set=data_values)
    return d_set

if __name__ == "__main__":
   #  trainingdata = DataSet("iris.data")
   #  classifier = Classifier()
    # run_crossfold_test(classifier,trainingdata)
    print("k")
