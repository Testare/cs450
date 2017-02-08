from functools import reduce
from china import DataSet,DataInstance,InstanceType,TargetColumn,IgnoreColumns
from math import inf
import re

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

class IrisDataInstance(DataInstance):
    '''An instance of Iris data'''
    min = [50]*4
    max = [0]*4
    feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
    def __init__(self,*args,**options):
        '''Initializes an instance of Iris data from a string'''
        super().__init__(*args,**options)
        self.data = list(map(float,self.data))
        self.sepal_length = float(self.data[0])
        self.sepal_width =  float(self.data[1])
        self.petal_length = float(self.data[2])
        self.petal_width =  float(self.data[3])
        self.sdata = []

        for i in range(len(self.data)):
            IrisDataInstance.max[i] = max(IrisDataInstance.max[i],self.data[i])
            IrisDataInstance.min[i] = min(IrisDataInstance.min[i],self.data[i])
    def compare(self,other):
        return sum(map(lambda x,y,mn,mx:((float(x)- float(y))/(mx - mn))**2,
                       self.data,
                       other.data,IrisDataInstance.max,IrisDataInstance.min))

@TargetColumn(0)
class VotingDataInstance(DataInstance):
    feature_names = [
        "handicapped-infants","water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze","el-salvador-aid","religious-groups-in-schools",
        "anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile",
        "immigration","synfuels-corporation-cutback","education-spending",
        "superfund-right-to-sue","crime","duty-free-exports",
        "export-administration-act-south-africa"
    ]
    pass

@IgnoreColumns(0)
class LensesDataInstance(DataInstance):
    regex = re.compile("\s+")
    feature_names = ["Age","Perscription","Astigmatic","Tear production rate"]
    pass

class PimaIndianInstance(DataInstance):
    feature_names = ["Pregnancies","Plasma Glucose Concentration",
                     "Blood press","Tricep thickness","serum insulin","BMI",
                     "Pedigree Function","Age"]

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.data = list(map(float, self.data))

@InstanceType(CarDataInstance)
class CarDataSet(DataSet):
    pass


class DiscretizedDataInstance(DataInstance):
    # TODO work on this more
    def __init__(self,line=None,data=None,target=None):
        super().__init__(data=data,target=target)

@InstanceType(DiscretizedDataInstance)
class DiscretizedDataSet(DataSet):
    pass

# TODO 1.0 Working on this currently
# TODO 1.1 Generalize this to all datasets
def discretize(data_set,bins=10):
    min_data = data_set.data[0].data
    max_data = min_data
    for data_inst in data_set.data:
        min_data = map(min,min_data,data_inst.data)
        max_data = map(max,max_data,data_inst.data)
    dividers = bins - 1
    data_bounds = []
    if(dividers < 0):
        raise Exception("Cannot discretize with <1 bins")
    for min_, max_ in zip(min_data,max_data):
        step = (max_ - min_)/bins
        lowbounds = {}
        for i in range(bins):
            lowbound = -inf if i == 0 else (min_ + step*(i))
            highbound = inf if i == dividers else (min_ + step*(i+1))
            target_str = "{:.2f}->{:.2f}".format(lowbound,highbound)
            lowbounds[lowbound] = target_str
        data_bounds.append(lowbounds)
    def apply_labels(data_instance):
        ret_data = []
        for bounds,data in zip(data_bounds,data_instance.data):
            biggest_key = -inf
            for key in bounds.keys():
                if data >= key:
                    biggest_key = key
                else: break
            ret_data.append(bounds[biggest_key])
        new_data_instance = DiscretizedDataInstance(
            data=ret_data,target=data_instance.target)
        new_data_instance.feature_names = data_instance.__class__.feature_names 
        return new_data_instance

        # Uncomment this to prove it works
        # print(self.data[0].data)
        # print(apply_labels(self.data[0]))
    return DiscretizedDataSet(data_set=list(map(apply_labels,data_set.data)))

@InstanceType(IrisDataInstance)
class IrisDataSet(DataSet):
    pass 

@InstanceType(LensesDataInstance)
class LensesDataSet(DataSet):
    pass

@InstanceType(VotingDataInstance)
class VotingDataSet(DataSet):
    pass

@InstanceType(PimaIndianInstance)
class PimaIndianSet(DataSet):
    pass
