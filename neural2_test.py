
import neural2

def test_battery():
    weight_net_test()
    activation_error_test()

def weight_net_test():
    assert str(neural2.weight_net(1,2,func=lambda l,i,j:1)) == "[[[1, 1], [1, 1]]]"
    assert str(neural2.weight_net(1,1,1,1,func=lambda l,i,j:1)) == "[[[1], [1]], [[1], " \
                                                     "[1]], [[1], [1]]]"
    assert str(neural2.weight_net(3,4,2,func=lambda l,i,j:2)) \
           == "[[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]], " \
              "[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]]"

def activation_error_test():
    net = neural2.weight_net(5,2,func=lambda l,i,j:2)
    net2 = neural2.weight_net(2,2,2,func=lambda l,i,j:1)

    activ = neural2.get_activations([1,1,2,1,1],net)
    activ2 = neural2.get_activations([1,1],net2)
    activ2_a1 = neural2.g(1)
    activ2_a2 = neural2.g(activ2_a1*2 - 1)
    # These are all calculated since we can redefine the outputs of g and g_prime
    activ2_e1 = neural2.g_prime(a=activ2_a2)*(activ2_a2)
    activ2_e2 = neural2.g_prime(a=activ2_a2)*(activ2_a2 - 1)
    activ2_e3 = neural2.g_prime(a=activ2_a1)*(activ2_e1 + activ2_e2)
    error = neural2.get_error(0,net,activ)
    error2 = neural2.get_error(1,net2,activ2)

    net2_lr = 0.5
    net2_w1 = 1 - net2_lr*activ2_a1*activ2_e1
    net2_w2 = 1 - net2_lr*activ2_a1*activ2_e2
    net2_w3 = 1 + net2_lr*activ2_e1
    net2_w4 = 1 + net2_lr*activ2_e2
    net2_w5 = 1 - net2_lr*activ2_e3
    net2_w6 = 1 + net2_lr*activ2_e3
    expected_updated_net2 = str([
        [[net2_w5, net2_w5], [net2_w5, net2_w5], [net2_w6, net2_w6]],
        [[net2_w1, net2_w2], [net2_w1, net2_w2], [net2_w3, net2_w4]]
    ])

    updated_net = neural2.update_net(net2,activ2,error2,net2_lr)

    assert neural2.activate_layer([1,1,2,1,1],net[0])[0] == neural2.g(10)

    assert str(activ) == "[[1, 1, 2, 1, 1], [{}, {}]]".format(neural2.g(10),neural2.g(10))
    assert str(activ2) == "[[1, 1], [{}, {}], [{}, {}]]".format(
        activ2_a1, activ2_a1,
        activ2_a2, activ2_a2
    )
    assert str(error) == "[[{}, {}]]".format(
            neural2.g_prime(neural2.g(10)) * (neural2.g(10) - 1),
            neural2.g_prime(neural2.g(10)) * neural2.g(10))
    assert str(error2) == "[[{}, {}], [{}, {}]]".format(
        activ2_e3,activ2_e3,
        activ2_e1,activ2_e2
    )
    assert str(updated_net) == str([
        [[net2_w5, net2_w5], [net2_w5, net2_w5], [net2_w6, net2_w6]],
        [[net2_w1, net2_w2], [net2_w1, net2_w2], [net2_w3, net2_w4]]
    ])

if "__main__" == __name__:
    test_battery()