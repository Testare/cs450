import random
from itertools import chain
from collections import deque
import operator
import math
import logging
from functools import reduce

from china import Classifier,DataInstance,normalized,\
    run_crossfold_test,run_simple_test,run_pretrained_test
from data_sets import IrisDataSet,PimaIndianSet, DataSet

from neural2 import weight_net,full_update_net,get_activations


verification_logger = logging.getLogger(__name__)

class NNClassifier(Classifier):
    def __init__(self, *layers_, learning_rate=1,target_set=None):
        """
        Creates a classifiers with nodes according to inputs

        :param input_count: The amount of inputs
        :param layers_: The number of nodes in the hidden and output layers
        """

        self.neurons = None
        # layers = zip(chain([input_count],layers_),layers_)
        self.weight_net = weight_net(*layers_)
        self.target_set = target_set
        self.learning_rate=learning_rate

    def fit(self, dataset, valid_set=None,epoch=0):
        if dataset.data:
            if not valid_set:
                training_set = dataset.copy()
                valid_set = training_set.split((len(dataset)*3 + 1)//10)
            else:
                training_set = dataset
            lst_ = deque()
            for epoch in range(1000):
                self.target_set = sorted(
                    {data_instance.target for data_instance in dataset})
                self.weight_net = reduce(lambda x,y:full_update_net(net=x,inputs_=y.data,target=self.target_set.index(y.target),learning_rate=self.learning_rate),training_set.data,self.weight_net)
                results = run_pretrained_test(self,valid_set)
                print("[{}] -> {:.2f}%".format(epoch,100*results.accuracy))
                print(results)
                lst_.append(results.accuracy)
                if(results.accuracy > 0.99):
                    return list(lst_)
            return list(lst_)

    def predict_instance(self,instance):
        final_activations = get_activations(net=self.weight_net,input_=instance.data)[-1]
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

    # print(len(pima_data))
    # test_data = pima_data.split(154)
    # pima_classifier = NNClassifier(8, 22, 2)

    # test_classifier = NNClassifier(3,2,2)
    # test_data_set = DataSet(data_set=[
    #     DataInstance(data=[1,2,3],target="m"),
    #     DataInstance(data=[1,3,5],target="l"),
    #     DataInstance(data=[3,3,1],target="m")
    # ])
    # #test_classifier.fit(test_data_set)
    # activ_ = test_classifier.get_activations([1,1,1])
    # for x in test_classifier.layers:
    #     print("* %s" % x)
    #     for node in x.nodes:
    #         print(node.weights)
    #
    # print(activ_)
    # print(test_classifier.get_error(activ_,"m"))
    # pima_classifier.fit(pima_data)
    # print(run_simple_test(pima_classifier,pima_data,test_data))

    # iris_classifier = NNClassifier(4,6,6,3,learning_rate=1)
    # print(iris_classifier.fit(iris_data))
    # print(run_pretrained_test(iris_classifier,iris_data))


    pima_classifier = NNClassifier(8,8,8,8,2,learning_rate=1)
    print(pima_classifier.fit(pima_data))
    print(run_pretrained_test(pima_classifier,iris_data))