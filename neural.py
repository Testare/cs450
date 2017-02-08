import random
from itertools import chain
from china import Classifier,DataInstance,normalized
from data_sets import IrisDataSet,PimaIndianSet


class Neuron:
    def __init__(self, inputs_):
        self.weights = [random.randrange(-100,100,)/float(100)
                        for _ in range(inputs_ + 1)]

    def __getitem__(self, inputs_):
        return self._g(sum(map(lambda x, y: x*y,
                               self.weights, chain([-1],inputs_))))

    @staticmethod
    def _g(sum_):
        if sum_ > 0:
            return 1
        return 0


class NeuronLayer:
    def __init__(self, inputs, nodes):
        self.nodes = [Neuron(inputs) for _ in range(nodes)]
        for x in self.nodes:
            print(x.weights)

    def __getitem__(self, inputs_):
        return [self.nodes[i][inputs_] for i in range(len(self.nodes))]


class NNClassifier(Classifier):
    def __init__(self, outputs):
        self.neurons = None
        self.outputs = outputs

    def fit(self, dataset):
        if self.neurons is None:
            inputs_ = len(dataset.instance_class.feature_names)
            self.neurons = NeuronLayer(inputs_,self.outputs)

    def outputs_for_instance(self, instance: DataInstance):
        if self.neurons is None:
            inputs_ = len(instance.feature_names)
            self.neurons = NeuronLayer(inputs_,self.outputs)
        return self.neurons[instance.data]

    def predict_instance(self,instance):
        return 0


if "__main__" == __name__:
    iris_data = normalized(IrisDataSet("datasets/iris.data"))
    pima_data = normalized(
        PimaIndianSet("datasets/pima-indians-diabetes.data"))
    classifier = NNClassifier(7)
    classifier.fit(pima_data)
    for k in pima_data:
        print(classifier.outputs_for_instance(k))