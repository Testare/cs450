import random
from itertools import chain
from china import Classifier,DataInstance,normalized,run_crossfold_test
from data_sets import IrisDataSet,PimaIndianSet
import math
import logging

verification_logger = logging.getLogger(__name__)

class Neuron:
    G_CONST = -1 #In case we want to try changing this constant later

    def __init__(self, inputs_):
        self.weights = [random.randrange(-100,100,)/float(100)
                        for _ in range(inputs_ + 1)]

    def __getitem__(self, inputs_):
        return self._g(sum(map(lambda x, y: x*y,
                               self.weights, chain([-1],inputs_))))

    @staticmethod
    def _g(sum_):
        # to be -> g(h) = 1/(1 + e^(-h))
        # if sum_ > 0:
        #     return 1
        # return 0
        return 1/(1 + math.exp(Neuron.G_CONST*sum_))

    @staticmethod
    def _g_prime(sum_=None,a=None):
        if a is None:
            a = Neuron._g(sum_)
        return a*(1-a)

class NeuronLayer:
    def __init__(self, inputs, nodes):
        self.nodes = [Neuron(inputs) for _ in range(nodes)]
        verification_logger.info("INIT-NEURON-LAYER")
        for x in self.nodes:
            verification_logger.info(x.weights)

    def __getitem__(self, inputs_):
        return [self.nodes[i][inputs_] for i in range(len(self.nodes))]


class NNClassifier(Classifier):
    def __init__(self, input_count, *layers_, target_set=None):
        """
        Creates a classifiers with nodes according to inputs

        :param input_count: The amount of inputs
        :param layers_: The number of nodes in the hidden and output layers
        """
        self.neurons = None
        layers = zip(chain([input_count],layers_),layers_)
        # print(list(layers))
        self.layers = [NeuronLayer(x,y) for (x,y) in layers]
        self.target_set = target_set

    def get_activations(self,inputs):
        def activation_for_layer(layers,inputs_):
            if not layers:
                return []
            outputs = layers[0][inputs_]
            verification_logger.info(str(layers[0]) + "->" + str(outputs))
            return [outputs] + activation_for_layer(layers[1:],outputs)

        verification_logger.info("GET_ACTIVATIONS_NNCLASSIFIER")
        return activation_for_layer(self.layers,inputs)

    def fit(self, dataset):
        self.target_set = sorted(
            {data_instance.target for data_instance in dataset})
        # Dun't do much rn
        # I might need a better way to specify classes later, but this is what
        # I got for now...

    def predict_instance(self,instance):
        final_activations = self.get_activations(instance.data)[-1]
        list_ = list(enumerate(final_activations))
        verification_logger.info(list(map(lambda x:("%d -> %s" % (x[0],self.target_set[x[0]]),x[1]),list_)))
        return self.target_set[max(list_,key=lambda x: x[1])[0]]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    # Comment out the following line to see some of the inner workings
    verification_logger.setLevel(logging.CRITICAL)
    iris_data = normalized(IrisDataSet("datasets/iris.data"))
    pima_data = normalized(
        PimaIndianSet("datasets/pima-indians-diabetes.data"))

    pima_classifier = NNClassifier(8,2,2)
    pima_classifier.fit(pima_data)
    example_indian = DataInstance(data=[0,1,2,3,4,5,6,7],target=0)
    verification_logger.info("PIMA PREDICTION: %s" % str(pima_classifier.predict_instance(example_indian)))

    iris_classifier = NNClassifier(4,6,3)
    iris_classifier.fit(iris_data)
    example_iris = DataInstance(data=[1.0,1.0,1.0,1.0],target="Iris-virginica")
    verification_logger.info("IRIS PREDICTION: %s" % str(iris_classifier.predict_instance(example_iris)))

    #Run crossfold test on iris_data
    run_crossfold_test(iris_data,NNClassifier,4,6,3)
    run_crossfold_test(pima_data,NNClassifier,8,8,2)
    # classifier.fit(pima_data)
    # for k in pima_data:
    #     print(classifier.outputs_for_instance(k))
