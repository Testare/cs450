import random
import math
from itertools import chain

G_CONST = -1


def g(sum_):
    return 1 / (1 + math.exp(G_CONST * sum_))


def g_prime(a):
    return a * (1 - a)


def weight_net(
        *layer_nodes,
        func=lambda l, i, j: random.randrange(-100, 100, ) / float(100)):
    """
    Returns a neural net in the shape
    [layer][i][j]
    where layer is 0 indexed
    :param layer_nodes:
    :return:
    """
    return [[[func(layer, input_, output_) for output_ in
              range(layer_nodes[layer + 1])]
             for input_ in range(layer_nodes[layer] + 1)]
            for layer in range(len(layer_nodes) - 1)]


def activate_layer(input_, layer):
    return [
        g(sum(layer[i][j] * in_ for i, in_ in
              zip(range(len(layer)), chain(input_,[-1]))))
        for j in range(len(layer[0]))
        ]


def get_activations(input_, net):
    if not net:
        return [input_]
    return [input_] + get_activations(activate_layer(input_, net[0]), net[1:])


def get_error_layer(error_before, net, activation):
    if not activation:
        return []
    error_layer = [g_prime(a=activation[-1][j])
                   * sum(net[-1][j][k] * error_before[k]
                         for k in range(len(error_before)))
                   for j in range(len(activation[-1]))]
    return get_error_layer(error_layer, net[:-1], activation[:-1]) \
           + [error_layer]


def get_error(target_node, net, activation):
    output_error = [g_prime(a=activation[-1][i])
                    * (activation[-1][i] - (0 if i != target_node else 1))
                    for i in range(len(activation[-1]))]
    return get_error_layer(output_error, net[1:], activation[1:-1]) + [
        output_error]


def update_net(net,activations,error,learning_rate):
    return [[[net[l][i][j] - (
        learning_rate * error[l][j] * (
            -1 if i == len(activations[l]) else activations[l][i]))
              for j in range(len(net[l][i]))]
             for i in range(len(net[l]))]
            for l in range(len(net))]


def full_update_net(net,inputs_,target,learning_rate):
    activations = get_activations(input_=inputs_, net=net)
    error = get_error(target_node=target, net=net, activation=activations)
    return update_net(net=net, activations=activations,
                      error=error, learning_rate=learning_rate)

# UTILITY FUNCTIONS

def report_ret(x,tag="",):
    print("{}:{}".format(tag,x))
    return x


def report_val(report,val):
    print(report)
    return val

if "__main__" == __name__:
    print(weight_net(1, 1, 1, 1))
    print(g(4))
    # get(get_error(1,weight_net(4,3,2)))
