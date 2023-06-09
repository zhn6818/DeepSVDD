from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .mydata_LeNet import mydata_LeNet, mydata_LeNet_Autoencoder
from .pa_LeNet import pa_LeNet, pa_LeNet_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet',
                            'cifar10_LeNet_ELU', 'mydata_LeNet', 'pa_LeNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'mydata_LeNet':
        net = mydata_LeNet()
    if net_name == 'pa_LeNet':
        net = pa_LeNet()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet',
                            'cifar10_LeNet_ELU', 'mydata_LeNet', 'pa_LeNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'mydata_LeNet':
        ae_net = mydata_LeNet_Autoencoder()

    if net_name == 'pa_LeNet':
        ae_net = pa_LeNet_Autoencoder()

    return ae_net
