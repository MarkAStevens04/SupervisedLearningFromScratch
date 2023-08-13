"""
= Future Upgrades =
    - different activation functions (sigmoid, relu, etc)
    - different starting weights & biases
    - randomize order of adjusting weights & biases
    - try adjusting all weights & biases at different times
        right now, we update the activations of the network as the weights & biases change
        try adjusting as if no other weights & biases have been affected
    - autosave & load networks
        - try getting a network pretty high in accuracy, then save
        - then start like 100 new networks from that high accuracy network,
        - save the network when it gets a higher accuracy, & repeat
    - click a neuron to change its value
    - when cost is high, steps should be big. When cost is low, steps should be small
    -
    - GPU optimizations:
            Network doesn't change until after change direction of all neurons & connections is determined.
            Therefor, we can get the current weights & activations of all neurons in the GPU first
            Then, distribute the work of re-running the network for each connection & neuron to
                different parts of the GPU.
"""

# pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu118/torch_stable.html
from os import path
import sys
sys.path.append(path.abspath('C:/Users/doyle/OneDrive/Desktop/Programming/Python/personalUtils'))
from utilities import *

import pygame
import time
import ctypes
import numpy as np
import torch
import random
history = open('tweaks.txt', 'r+')
ctypes.windll.user32.SetProcessDPIAware()

# Technical
SCREEN_X = 3840 / 2
SCREEN_Y = 2160 / 2
FPS = 2000
ACCURACY_SAMPLING = 100

# Aesthetic
BACKGROUND = (0, 0, 0)
LINE_WIDTH = 3
DEFAULT_LINE_COLOR  = (255, 255, 255)
DEFAULT_NEURON_COLOR = (200, 255, 200)
CIRCLE_SIZE = 50
BUFFER = 200
# tries to keep min_spacing for all neurons
MIN_SPACING = 140
NEURON_COLOR_LIST = [(255, 0, 0), (50, 50, 50), (0, 255, 0)]
LINE_COLOR_LIST = [(255, 0, 0), (50, 50, 50), (0, 255, 0)]
DEFAULT_TEXT_COLOR = (255, 255, 255)



# Create Screen
screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y), pygame.HWSURFACE|pygame.DOUBLEBUF)
pygame.display.set_caption('SLFS')
pygame.font.init()
font = pygame.font.Font('freesansbold.ttf', 32)


# Default Network Attributes
# 0.001, 0.005, 0.01, 0.0001
# DEFAULT_STEP_SIZE = 0.001
DEFAULT_STEP_SIZE = 0.03
# modifies step size depending on how much it affects the network
SLOW_APPROACH = True

# minimum difference in costs to implement a change
MIN_COST_THRESHHOLD = 0.0000000001

# NEW_STEP_SIZE is the step size when SLOW_APPROACH is True
# *note* DEFAULT_STEP_SIZE is the size of steps used to find the correct direction to move
# while NEW_STEP_SIZE is how much to scale the difference between the previous cost and the better cost
# essentially, DEFAULT finds the direction, NEW_STEP finds the strength
NEW_STEP_SIZE = 0.03

# VARIABLE_STEP_SIZE modifies the step size of the entire network
# based on the accuracy of the network as a whole
# step size constant is how quickly we should decrease the step size based on the accuracy
VARIABLE_STEP_SIZE = False
stepSizeConstant = 2

# Maximum difference between desired result and actual result to be considered a correct identification.
# ex: 0.1 means that an actual result of 0.95 with desired 1 is correct, while a: 0.89 e: 1 is incorrect
# note: value shouldn't be above 0.5. a: 0.500000...1, e:1 should be lowest accuracy considered correct
# ACCURACY_THRESHHOLD = 0.1
ACCURACY_THRESHHOLD = 0.001

# NOTE about DEBUG: holding mouse down is fast, pressing space is slow
debug = DebugObject()
debug.toggle_print(False)
debug.add_print_categories('u')
debug.add_print_categories('GPU')
DEBUG = True
COST_PRECISION = 12
CONNECTION_WEIGHT_SCALE_FACTOR = 1
NEURON_WEIGHT_SCALE_FACTOR = 1

# Affirm GPU acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('using', device, 'device')

# compare times!
tCPU = Timer()
tGPU = Timer()
tMisc = Timer()
tCPU.changeSampleSize(9)
tGPU.changeSampleSize(9)
tMisc.changeSampleSize(9)

def findColor(color_list, index) -> tuple[int, int, int]:
    """
    takes a list of colors, along with an index, and returns the correct color
    = Pre-conditions =
    index is float between 0 and 1
    """
    spacing = 1/len(color_list)
    pos = int(index / spacing)
    distance = index % spacing * len(color_list)

    startC = color_list[pos]
    if pos == len(color_list) - 1:
        endC = color_list[0]
    else:
        endC = color_list[pos + 1]

    outList = [0, 0, 0]
    for c in range(3):
        outList[c] = (endC[c] - startC[c]) * distance + startC[c]

    outC = tuple(outList)

    return outC

def findAccColor(color_list, index) -> tuple[int, int, int]:
    """
    takes a list of colors, along with an index, and returns the correct color
    this is adjusted so that top & bottom are the first & last colors
    = Pre-conditions =
    index is float between 0 and 1
    at least 2 colors in color_list
    """
    spacing = 1/(len(color_list)-1)
    pos = int(index / spacing)
    distance = index % spacing * (len(color_list) - 1)

    # hot fix to get rid of error when index = 1
    if index == 1:
        pos -= 1
        distance = 0.999999

    startC = color_list[pos]
    endC = color_list[pos + 1]

    outList = [0, 0, 0]
    for c in range(3):
        outList[c] = (endC[c] - startC[c]) * distance + startC[c]

    outC = tuple(outList)

    return outC

def activationFunction(activation: float) -> float:
    """l
    this module acts as our squisher! it wil take any flaot, and return a float.
    For right now, we're going to use a ReLU function.
    (if the activation is negative, return 0. Otherwise, return the original value)
    """
    # to_return = sigmoid(activation*10)
    to_return = sigmoid(activation)
    # print('orig: ', round(activation, 2), 'sigm: ', round(to_return, 2))
    # return min(max(0, activation), 1)
    return to_return

def sigmoid(x: float) -> float:
    """
    this is just to put the sigmoid function in one place.
    This takes any number, and returns a number between 0 and 1.
    Similar to our activationFunction, but this will only produce between a 0 and 1.
    """
    return 1 / (1 + np.exp(-x))

def varyStepSize(accuracy) -> float:
    """
    takes an accuracy, and returns an appropriate step size
    """
    # a = 1 - (1 / (1 + np.exp(-5 * 0.5)))
    # b = 1 - (1 / (1 + np.exp(-5 * accuracy)))
    # a = (b / a) * DEFAULT_STEP_SIZE

    # a = ((1 - accuracy) ** 2) * DEFAULT_STEP_SIZE * 4
    # a = np.exp(-1 * stepSizeConstant * accuracy) * DEFAULT_STEP_SIZE

    a = (1 - accuracy) * DEFAULT_STEP_SIZE
    debug.p(f'new step size: {round(a, 3)}', 'u')
    debug.p(f'accuracy: {round(accuracy, 2)}')
    return a

class Network:
    """
    A neural network, created from scratch
    == Attributes ==
    input_width: num input neurons
    output_width: num output neurons
    num_hidden_layers: num hidden layers
    hidden_width: width of every hidden layer
    step_size: float describing how much the network should move

    _layers: list of layers of neurons. Each list layer is populated with correct num of neurons
        [[input_neuron_1, i_n_2, ..., i_n_n], [hidden_neuron_1, ..., hidden_neuron_n], ..., ]]

    _spacing: private attribute to enable consistent spacing

    _accuracyHistory: list of past correct / failed identifications
                            max length is based on ACCURACY_SAMPLING
    _accuracy: float of the current accuracy

    _highAcc: highest accuracy achieved

    _numSteps: number of steps ever taken by network

    = Pre-conditions =
    screen must already be initialized
    colors must be defined
    """
    def __init__(self, input_width: int, output_width: int, num_hidden_layers: int, hidden_width) -> None:
        self.input_width = input_width
        self.output_width = output_width
        self.num_hidden_layers = num_hidden_layers
        self.hidden_width = hidden_width
        self.step_size = DEFAULT_STEP_SIZE
        self._layers = [[], []]
        self._spacing = 0
        self._accuracyHistory = []
        self._accuracy = 0.5
        self._highAcc = 0
        self._steps = 0
        self._costThreshhold = MIN_COST_THRESHHOLD
        self._biasTensorRepr = torch.zeros((num_hidden_layers + 2), max(input_width, output_width, hidden_width))

        # for activations, better to have activations in one layer occupy a single row. Makes matrix mult easier
        self._activationTensorRepr = torch.zeros((num_hidden_layers + 2), max(input_width, output_width, hidden_width))

        # outline each layer of the network
        for i in range(self.num_hidden_layers):
            self._layers.append([])

        # populate input layer
        for i in range(self.input_width):
            new = Neuron()
            # input layer should have no biases
            new._bias = 0
            self._layers[0].append(new)
        self._calcNeuronPosition(0)

        # populate hidden layers
        for layer_index in range(self.num_hidden_layers):
            for i in range(hidden_width):
                new = Neuron()
                self._layers[layer_index + 1].append(new)
            self._calcNeuronPosition(layer_index + 1)

        # populate output layer
        for i in range(self.output_width):
            new = Neuron()
            # new._bias = 1
            self._layers[-1].append(new)
        self._calcNeuronPosition(len(self._layers) - 1)

        # make connections
        for i in range(len(self._layers) - 1):
            for sNeuron in self._layers[i]:
                for eNeuron in self._layers[i+1]:
                    conA = Connection(sNeuron, eNeuron)
                    # sNeuron.addPostConnection(conA)
                    # eNeuron.addPreConnection(conA)

        # update tensor representations of neurons
        for i in range(len(self._layers) - 1):
            for sNeuron in self._layers[i+1]:
                sNeuron.buildTensor()

    def _calcNeuronPosition(self, layer_index) -> None:
        """
        adjust the position of all neurons in layer

        = Pre-conditions =
            input layer must be calculated FIRST
        """
        if self.num_hidden_layers == 0:
            print('ERROR! No hidden layers')
            dx = (SCREEN_X - 2 * BUFFER) / (self.num_hidden_layers + 1)
            x = BUFFER + dx * layer_index
        else:
            dx = (SCREEN_X - 2 * BUFFER) / (self.num_hidden_layers + 1)
            x = BUFFER + dx * layer_index

        # formats the network to have consistent neuron spacing
        # the hidden layers are defined to match the spacing of the input layer
        if layer_index == 0 or layer_index == len(self._layers) - 1:
            buff = BUFFER
        else:
            if self._spacing < MIN_SPACING:
                buff = (SCREEN_Y + BUFFER - (MIN_SPACING * len(self._layers[layer_index]))) / 2
            else:
                buff = (SCREEN_Y + BUFFER - (self._spacing * len(self._layers[layer_index]))) / 2

            if buff < BUFFER:
                buff = (1/2) * BUFFER

        # first, give all neurons proper layer labels
        for n in self._layers[layer_index]:
            n.changeLayer(layer_index)

        # checks the number of neurons for outliers
        if len(self._layers[layer_index]) == 1:
            self._layers[layer_index][0].changePos((x, SCREEN_Y / 2))
            dy = 0
        elif len(self._layers[layer_index]) == 2:
            self._layers[layer_index][0].changePos((x, BUFFER))
            self._layers[layer_index][-1].changePos((x, SCREEN_Y - BUFFER))
            dy = (SCREEN_Y - (2 * buff)) / 2
        elif len(self._layers[layer_index]) > 2:
                dy = (SCREEN_Y - (2 * buff)) / (len(self._layers[layer_index]) - 1)
                # we go through the length of the layer except 2 because we already set the first and last neuron
                # those are special cases, & should be treated as such
                for neuron_index in range(len(self._layers[layer_index])):
                    yPos = buff + (dy * neuron_index)
                    self._layers[layer_index][neuron_index].changePos((x, yPos))

        if layer_index == 0:
            self._spacing = dy

    def _input(self, in1: list[int]) -> None:
        """
        Take in1 as a list of data
        Assign each value from in1 to the activations of the input neurons
        """
        for data_index in range(len(in1)):
            self._layers[0][data_index]._activation = in1[data_index]

        self._buildTensorRepr()

    def _runNetwork(self) -> None:
        """
        updates activations of all neurons in all layers
        dependent on neuron's biases & weights of connections

        = Pre-conditions =
        input has already been called
            if not, this function will just produce meaningless results

        For this function, sigmoid on all neurons.
        This lets the weights have high values, while our output will be between 0 and 1.
        """
        tCPU.start()

        for layer_index in range(1, len(self._layers)):
            for neuron in self._layers[layer_index]:

                act = 0
                for connect in neuron.preConnections:
                    act += connect._weight * connect.start._activation

                act += neuron._bias
                # print('pre: ', round(act, 3), 'post: ', round(activationFunction(act), 3))
                act = activationFunction(act)

                # act += neuron._bias


                neuron.changeActivation(act)

        tCPU.end()
        # self._runNetworkTensor()
        #     print('went inside')
        #
        # print('going through')

    def _calcCost(self, label: list[int]) -> float:
        """
        determine the cost for the current activations
        label is a set describing the correct output of each of the neurons

        *smaller cost is better*

        = Pre-conditions =
        runNetwork has been called
        length of label is
        """
        cost = 0
        debug.p(f'--Evaluating--', 'c')
        for label_index in range(len(label)):
            # want small changes to be meaninful when accuracy is very off
            # want small changes to be insignificant when accuracy is close
            actual = self._layers[-1][label_index]._activation
            expected = label[label_index]

            debug.p(f'prevCost: {round(cost, 2)}', 'c')
            debug.p(f'a: {round(actual, 2)}', 'c')
            debug.p(f'e: {round(expected, 2)}', 'c')
            cost += np.square(actual - expected)
            debug.p(f'NewCost: {round(cost, 10)}', 'c')
            debug.p('', 'c')
        return cost

    def _updateNetwork(self) -> None:
        """
        Updates all neurons and connections to their future values
        :return:
        """
        for layer_index in range(len(self._layers) - 1, -1, -1):
            # first, we adjust future weights of connections
            if layer_index != 0:
                for neuron_index in range(len(self._layers[layer_index])):
                    for connection in self._layers[layer_index][neuron_index].preConnections:
                        connection.updateWeight()

            # then adjust future biases
                for neuron_index in range(len(self._layers[layer_index])):
                    neuron = self._layers[layer_index][neuron_index]
                    neuron.updateBias()

    def _buildTensorRepr(self) -> None:
        """
        Creates the accurate tensor representation of the network.

        = Pre-condition =
        Input method must already have been called
        :return:
        """

        for layIndex in range(len(self._layers)):
            for neurIndex in range(len(self._layers[layIndex])):
                neuron = self._layers[layIndex][neurIndex]
                self._activationTensorRepr[layIndex][neurIndex] = neuron._activation
                self._biasTensorRepr[layIndex][neurIndex] = neuron._bias

    def _updateBiasRepr(self) -> None:
        """
        updates the bias representation of the network
        :return:
        """
        for layIndex in range(len(self._layers)):
            for neurIndex in range(len(self._layers[layIndex])):
                neuron = self._layers[layIndex][neurIndex]
                # note that activation and bias have opposite dimensions, and are thus called accordingly
                self._biasTensorRepr[layIndex][neurIndex] = neuron._bias

    def _runNetworkTensor(self) -> None:
        """
        Updates the tensor representation of the network
        For every layer, starting on the layer after input layer,
        calculate the activations of the neurons by performing a matrix multiplication
        of the previous layer's activations with the corresponding connections.

        Activations first, weights second.

        Activations are a 1xn matrix.
        Weights are an nxm matrix.

        n is number of neurons in previous layer. m is number of neurons in current layer.
        :return:
        """
        # start at index 1 because we do not update the activations of the input layer.
        # must update activations from furthest left to furthest right.
        torch.cuda.synchronize()
        tGPU.start()

        debug.p('going ', 'GPU')
        self._updateBiasRepr()
        gpu_bias = self._biasTensorRepr.to(device)
        for layIndex in range(1, len(self._layers)):
            numPrevNeurons = len(self._layers[layIndex - 1])
            weights = self._retrieveTensorWeights(layIndex)
            activations = self._activationTensorRepr[layIndex - 1][0:numPrevNeurons]
            # note that activation and bias have opposite dimensions, and are thus called accordingly

            # must update the activations with sigmoid and bias in every layer
            a_gpu = activations.to(device)
            w_gpu = weights.to(device)

            next_activations = torch.matmul(a_gpu, w_gpu)

            # next_activations = next_activations + self._biasTensorRepr[layIndex]
            next_activations = next_activations + gpu_bias[layIndex]
            next_activations = torch.sigmoid(next_activations)

            self._activationTensorRepr[layIndex] = next_activations

        debug.p('activation in tensors: ', 'GPU')
        debug.p(f'{self._activationTensorRepr}', 'GPU')

        tGPU.end()
        torch.cuda.synchronize()


    def _retrieveTensorWeights(self, layerIndex: int) -> torch.tensor:
        """
        Takes a layer index, then creates a matrix representation of all weights of all connections
        between this layer and previous layer.

        Each column represents a single neuron's weights.
        = pre-conditions =
        CANNOT be called on input layer, as it has no previous connections

        :param layerIndex:
        :return:
        """
        numConnections = len(self._layers[layerIndex - 1])
        numNeurons = len(self._layers[layerIndex])
        returnTensor = torch.zeros(numConnections, numNeurons)
        # print('---')
        # print('num connections: ', numConnections)
        # print('num neurons: ', numNeurons)
        # print('return Tensor: ')
        for neurIndex in range(numNeurons):
            weightTensor = self._layers[layerIndex][neurIndex]._tensorReprWeights
            # print(weightTensor)
            # print(returnTensor[:, neurIndex])
            # print(returnTensor)
            # print()
            returnTensor[:, neurIndex] = weightTensor
        # print('row of connections: ')
        # print(returnTensor)
        return returnTensor

    # def _calcOneLayerActivation(self, activations: torch.tensor, weights: torch.tensor, bias: torch.tensor) -> torch.tensor:
    #     # Fully calculates the activation of a single neuron, given the previous layer's activations,
    #     # weights of connections, and neuron's bias.
    #     #
    #     # = pre-condition =
    #     #   - activations of previous layer are accurate
    #     #
    #     # :param activations: one row matrix, each entry describing corresponding activation of neuron.
    #     # :param weights: one column matrix, each entry describing corresponding activation of neuron.
    #     # :return: 1x1 tensor with the new activation of the neuron


    def _step(self, label: list[int]):
        """
        adjusts the weights & biases to reduce the cost function

        = Pre-conditions =
        length of list matches number of output neurons
        """
        tMisc.start()
        self._steps += 1

        # layer_index goes from last layer to first
        # starts by first adjusting the connections
        # might be helpful to randomize the order the weights & biases are adjusted
        debug.p(f'****************************** start ******************************')
        debug.p(f'step size: {round(self.step_size, 3)}')
        NEW_STEP_SIZE = self.step_size

        for layer_index in range(len(self._layers) - 1, -1, -1):
            # first, we adjust future weights of connections
            if layer_index != 0:
                debug.p(f'--- connections ---')

                for neuron_index in range(len(self._layers[layer_index])):
                    debug.p(f'--new neuron--')

                    for connection in self._layers[layer_index][neuron_index].preConnections:

                        debug.p(f'-connection-')
                        debug.p(connection)

                        first_cost = self._calcCost(label)
                        connection._weight += self.step_size
                        self._runNetwork()
                        cost_plus = self._calcCost(label)
                        connection._weight -= 2 * self.step_size
                        self._runNetwork()
                        cost_minus = self._calcCost(label)
                        connection._weight += self.step_size

                        if first_cost <= (cost_plus + self._costThreshhold) and \
                                first_cost <= (cost_minus + self._costThreshhold):
                            # the best option is to do nothing

                            debug.p(f'startCost    : {round(first_cost, COST_PRECISION)} *')
                            debug.p(f'addStepCost  : {round(cost_plus, COST_PRECISION)}')
                            debug.p(f'minusStepCost: {round(cost_minus, COST_PRECISION)}')
                            debug.p()
                            pass
                        elif cost_plus <= cost_minus:
                            # cost plus is better
                            # connection._weight += self.step_size

                            if SLOW_APPROACH:
                                # diff = first_cost - cost_plus
                                diff = cost_plus / first_cost
                                connection.nextWeight(connection._weight + (diff * NEW_STEP_SIZE))
                                debug.p(f'd: {diff}')
                            else:
                                connection.nextWeight(connection._weight + self.step_size)

                            debug.p(f'startCost    : {round(first_cost, COST_PRECISION)}')
                            debug.p(f'addStepCost  : {round(cost_plus, COST_PRECISION)} *')
                            debug.p(f'minusStepCost: {round(cost_minus, COST_PRECISION)}')
                            debug.p()
                        else:
                            # cost minus is better

                            if SLOW_APPROACH:
                                # diff = first_cost - cost_minus
                                diff = cost_minus / first_cost
                                connection.nextWeight(connection._weight - (diff * NEW_STEP_SIZE))
                                debug.p(f'd: {diff}')
                            else:
                                connection.nextWeight(connection._weight - self.step_size)

                            # connection._weight -= self.step_size

                            debug.p(f'startCost    : {round(first_cost, COST_PRECISION)}')
                            debug.p(f'addStepCost  : {round(cost_plus, COST_PRECISION)}')
                            debug.p(f'minusStepCost: {round(cost_minus, COST_PRECISION)} *')
                            debug.p()
                        debug.p('no change to current value, change to future')
                        debug.p(connection)
                    debug.p()

                # to not adjust bias of output layers:
                # if layer_index != len(self._layers) - 1:

                if layer_index != len(self._layers):
                # then adjust future biases
                    debug.p('--- neurons ---')

                    for neuron_index in range(len(self._layers[layer_index])):
                        first_cost = self._calcCost(label)
                        neuron = self._layers[layer_index][neuron_index]

                        debug.p('-neuron-')
                        debug.p(neuron)

                        neuron._bias += self.step_size
                        self._runNetwork()
                        cost_plus = self._calcCost(label)
                        neuron._bias -= 2 * self.step_size
                        self._runNetwork()
                        cost_minus = self._calcCost(label)
                        neuron._bias += self.step_size

                        if first_cost <= (cost_plus + self._costThreshhold) and \
                                first_cost <= (cost_minus + self._costThreshhold):
                            # the best option is to do nothing

                            debug.p(f'startCost    : {round(first_cost, COST_PRECISION)} *')
                            debug.p(f'addStepCost  : {round(cost_plus, COST_PRECISION)}')
                            debug.p(f'minusStepCost: {round(cost_minus, COST_PRECISION)}')
                            debug.p()

                        elif cost_plus <= cost_minus:
                            # best to add to the bias
                            if SLOW_APPROACH:
                                # diff = first_cost - cost_plus
                                diff = cost_plus / first_cost
                                neuron.nextBias(neuron._bias + (diff * NEW_STEP_SIZE))
                                debug.p(f'd: {diff}')
                            else:
                                neuron.nextBias(neuron._bias + self.step_size)
                            # neuron._bias += self.step_size

                            debug.p(f'startCost    : {round(first_cost, COST_PRECISION)}')
                            debug.p(f'addStepCost  : {round(cost_plus, COST_PRECISION)} *')
                            debug.p(f'minusStepCost: {round(cost_minus, COST_PRECISION)}')
                            debug.p()
                        else:
                            # best to subtract from the bias
                            if SLOW_APPROACH:
                                # diff = first_cost - cost_minus
                                diff = cost_minus / first_cost
                                neuron.nextBias(neuron._bias - (diff * NEW_STEP_SIZE))

                                debug.p(f'd: {diff}')
                            else:
                                neuron.nextBias(neuron._bias - self.step_size)

                            debug.p(f'startCost    : {round(first_cost, COST_PRECISION)}')
                            debug.p(f'addStepCost  : {round(cost_plus, COST_PRECISION)}')
                            debug.p(f'minusStepCost: {round(cost_minus, COST_PRECISION)} *')
                            debug.p()
                        neuron.updateTensor()

                debug.p()
                        # neuron._bias -= self.step_size
        #         print(neuron)
        #         print()
        # print()
        # print()
        # print('*****')

        # determine if network is accurate or not
        tMisc.end()
        self._updateNetwork()
        self._runNetworkTensor()


        results = self._findAccuracy(label, ACCURACY_THRESHHOLD)
        if 0 in results:
            self._updateAccuracy(0)
        else:
            self._updateAccuracy(1)


    def _findAccuracy(self, label: list[int], threshhold: float):
        """
        Returns a 0 or 1 based on whether the specific neuron is within the correct
        threshhold of the label.
        :param label:
        :param threshhold:
        :return:
        """
        self._runNetwork()
        results = []

        # all output neurons must be within accuracy threshhold for the identification to
        # be considered correct.
        for label_index in range(len(label)):
            actual = self._layers[-1][label_index]._activation
            expected = label[label_index]
            if (expected - threshhold <= actual <= expected + threshhold):
                results.append(1)
            else:
                results.append(0)

        return results

    def _updateAccuracy(self, result: int):
        """
        takes a result (either a correct or failed identification) and adds it to the
        Network's history.
        """
        self._accuracyHistory.insert(0, result)
        if len(self._accuracyHistory) > ACCURACY_SAMPLING:
            self._accuracyHistory.pop()
        self._accuracy = sum(self._accuracyHistory) / len(self._accuracyHistory)

        if self._accuracy > self._highAcc and self._steps > ACCURACY_SAMPLING:
            self._highAcc = self._accuracy

        if VARIABLE_STEP_SIZE:
            self.step_size = varyStepSize(self._accuracy)

    def _changeStepSize(self):
        """
        modifies the current step size based on the accuracy of the network
        """
        self.step_size = DEFAULT_STEP_SIZE

    def drawOnScreen(self, sc, index, AorB='b'):
        """
        draws all neurons & connections on screen
        AorB determines displaying a neuron's activation or bias in its color. default is bias
        """
        for layer_index in range(len(self._layers)):
            for neuron_index in range(len(self._layers[layer_index])):
                neuron = self._layers[layer_index][neuron_index]
                # for the neurons, change it to white and black?
                # print(neuron._activation)

                if AorB == 'a':
                    c = findAccColor(NEURON_COLOR_LIST, min(max(neuron._activation, 0), 1))
                    neuron.changeColor(c)
                    neuron.drawOnScreen(sc, 'a')
                else:
                    c = findAccColor(NEURON_COLOR_LIST, sigmoid(neuron._bias))
                    neuron.changeColor(c)
                    neuron.drawOnScreen(sc)


                if neuron.postConnections != []:
                    for con in neuron.postConnections:
                        w = sigmoid(con._weight)
                        c = findAccColor(LINE_COLOR_LIST, w)
                        con.changeColor(c)
                        con.drawOnScreen(screen)


class Connection:
    """
    A connection between multiple neurons
    == Attributes ==
    start: beginning neuron
    end: ending neuron
    _color: color of the segment (purely visual)
    _xy1: starting position, dependent on pos. of start neuron (float, float)
    _xy2: ending position, dependent on pos. of end neuron (float, float)
    _weight: significance of connection
                can be any real number
    """
    def __init__(self, start, end) -> None:
        """
        creates a connection between two neurons
        adds self to list of preConnection & postConnection for corresponding neurons
        = Pre-conditions =
        start & end neuron must already exist
        start & end neuron must have valid positions
        connection must not already exist in pre or postConnection in start&end neurons
        """
        self.start = start
        self.end = end
        self.color = DEFAULT_LINE_COLOR
        self._xy1 = start.findPosition()
        self._xy2 = end.findPosition()

        # random activation of weights, dependent on some scaling factor
        self._weight = (random.random() - 0.5) * 2 * CONNECTION_WEIGHT_SCALE_FACTOR

        self._futureWeight = None
        start.addPostConnection(self)
        end.addPreConnection(self)

    def changeColor(self, color: tuple[int, int, int]) -> None:
        """
        alters the color of the connection
        """
        self.color = color

    def nextWeight(self, weight) -> None:
        """
        Updates the future value for the weight of this connection
        :param weight:
        :return:
        """
        self._futureWeight = weight

    def updateWeight(self) -> None:
        """
        Updates the current weight to the value of the future weight

        MUST BE CALLED BEFORE UPDATE BIAS IN NEURONS!
        The weights must all be updated first.
        :return:
        """
        if self._futureWeight is not None:
            self._weight = self._futureWeight
            debug.p('oooooooooooooooooooooooooooo updated weight ooooooooooooooooooooooooo', 'u')

    def drawOnScreen(self, sc):
        """
        sc is screen being drawn on
        draws the connection on the screen
        """
        pygame.draw.line(sc, self.color, self.start.findPosition(), self.end.findPosition(), LINE_WIDTH)
        if DEBUG:
            offset = 0.25
            textX = (self._xy1[0] - self._xy2[0]) * offset + self._xy2[0]
            textY = (self._xy1[1] - self._xy2[1]) * offset + self._xy2[1]
            text = font.render(str(round(self._weight, 3)), True, self.color, BACKGROUND)
            textRect = text.get_rect()
            textRect.center = (textX, textY)
            sc.blit(text, textRect)

    def __str__(self):
        """
        return stats of the connection
        """
        return f"between layers {self.start._layer} and {self.end._layer}\n" \
               f"current Weight: {self._weight}\n" \
               f"future  Weight: {self._futureWeight}\n"


class Neuron:
    """
    a single neuron in a neural network
    == Attributes ==
    x: x position
    y: y position
    _layer: int for which layer the neuron is in
           0 is input layer, 1 through n are hidden layers, n + 1 is output layer
    preConnections: list of Connection objects between this neuron and neurons in previous layer
            if this neuron is in input layer, this list is empty
    postConnections: list of Connection objects between this neuron and neurons in next layer
            if this neuron is in output layer, this list is empty

    _color: color of the neuron
    _bias: amount required to have neuron become significantly activated
                can be any real number
    _activation: how turned on this neuron is ;)
    _tensorReprWeights: 1xN tensor with each entry corresponding to the weight of the pre-connection neuron
    """
    def __init__(self):
        self.x = 0
        self.y = 0
        self._layer = 0
        self.preConnections = []
        self.postConnections = []
        self.color = DEFAULT_NEURON_COLOR

        # random bias between -1 and 1 multiplied by some scaling factor
        self._bias = (random.random() - 0.5) * 2 * NEURON_WEIGHT_SCALE_FACTOR
        self._futureBias = None
        # self._activation = 0.75
        self._activation = 0

    def buildTensor(self) -> None:
        """
        Creates a tensor representation of all weights connected to this neuron
        :return:
        """
        self._tensorReprWeights = torch.zeros(1, len(self.preConnections))

    def updateTensor(self) -> None:
        """
        reEvaluates the tensor and updates all values
        :return:
        """
        for connectionIndex in range(len(self.preConnections)):
            currCon = self.preConnections[connectionIndex]
            self._tensorReprWeights[0, connectionIndex] = currCon._weight

    def addPreConnection(self, c) -> None:
        """adds the connection, c, to this neurons list of preConnections

        = Pre-conditions =
        c is a connection object
        c does not already exist in the list
        """
        self.preConnections.append(c)

    def addPostConnection(self, c) -> None:
        """adds the connection, c, to this neurons list of postConnections

        = Pre-conditions =
        c is a connection object
        c does not already exist in the list
        """
        self.postConnections.append(c)

    def setPos(self, pos: tuple[int]) -> None:
        """
        takes x,y coords & set the position of the neuron
        purely visual, has no effect on functionality

        = Pre-conditions =
        x,y must exist on screen
        """
        self.x = pos[0]
        self.y = pos[1]

    def changePos(self, pos: tuple[int]) -> None:
        """
        takes x,y coords & adjusts the position of the neuron
        purely visual, has no effect on functionality

        = Pre-conditions =
        x,y must exist on screen
        """
        self.x += pos[0]
        self.y += pos[1]

    def changeColor(self, color: tuple[int, int, int]) -> None:
        """
        alters the color of the neuron
        """
        self.color = color

    def drawOnScreen(self, sc, AorB='b'):
        """
        sc is screen being drawn on
        draws this neuron on the screen

        AorB means draw the activation or bias of the neuron. default is bias
        """
        pygame.draw.circle(sc, self.color, (self.x, self.y), CIRCLE_SIZE)
        if DEBUG:
            text = font.render(str(round(self._bias, 3)), True, DEFAULT_TEXT_COLOR)
            textRect = text.get_rect()
            textRect.center = (self.x, self.y - 80)
            sc.blit(text, textRect)

            if AorB == 'a':
                text = font.render(str(round(self._activation, 3)), True, DEFAULT_TEXT_COLOR)
                textRect = text.get_rect()
                textRect.center = (self.x, self.y + 80)
                sc.blit(text, textRect)

    def changeActivation(self, activation):
        """
        updates the activation of the neuron
        """
        self._activation = activation

    def findPosition(self) -> tuple[int, int]:
        return self.x, self.y

    def changeLayer(self, layer) -> None:
        """
        adjust the label of this neuron's layer
        :return:
        """
        self._layer = layer

    def nextBias(self, bias) -> None:
        """
        updates the future bias to some value
        :param bias:
        :return:
        """
        self._futureBias = bias

    def updateBias(self) -> None:
        """
        Updates the bias of the neuron to the future bias value.
        Must be called AFTER all weights have been updated to their future values
        :return:
        """
        if self._futureBias is not None:
            self._bias = self._futureBias
            self.updateTensor()

    def __str__(self):
        """
        return stats of the neuron
        """
        return f"layer: {self._layer}, x: {self.x}, y: {self.y}\n" \
               f"activation: {self._activation}, bias: {self._bias} \n"
               # f"postCon: {self.postConnections}\n" \
               # f"preCon: {self.preConnections} \n"


c = Network(2, 1, 3, 40)
i = 1

data1 = [[[0, 0], [0]], [[1, 0], [1]], [[0, 1], [1]], [[1, 1], [0]]]
# xOR
# [[[data1], [label]], [[data2], [label]], ....]


def xOR_solvedNet() -> Network:
    """
    returns a neural network that has solved the xOR problem
    :return:
    """
    c = Network(2, 1, 1, 2)
    c._layers[0][0]._bias = 0
    c._layers[0][1]._bias = 0
    c._layers[1][0]._bias = -1
    c._layers[1][1]._bias = 3
    c._layers[2][0]._bias = -3

    c._layers[0][0].postConnections[0]._weight = 2
    c._layers[0][0].postConnections[1]._weight = -2
    c._layers[0][1].postConnections[0]._weight = 2
    c._layers[0][1].postConnections[1]._weight = -2

    c._layers[1][0].postConnections[0]._weight = 2
    c._layers[1][1].postConnections[0]._weight = 2
    return c

def xOR_primedNet() -> Network:
    """
    Returns a neural network primed to solve the xOR problem
    :return:
    """
    strength = 0.302
    c = Network(2, 1, 1, 2)
    c._layers[0][0]._bias = 0
    c._layers[0][1]._bias = 0
    c._layers[1][0]._bias = -1 * strength
    c._layers[1][1]._bias = 3 * strength
    c._layers[2][0]._bias = -3 * strength

    c._layers[0][0].postConnections[0]._weight = 2 * strength
    c._layers[0][0].postConnections[1]._weight = -2 * strength
    c._layers[0][1].postConnections[0]._weight = 2 * strength
    c._layers[0][1].postConnections[1]._weight = -2 * strength

    c._layers[1][0].postConnections[0]._weight = 2 * strength
    c._layers[1][1].postConnections[0]._weight = 2 * strength
    return c

def recurseData(difficulty: int):
    """
    Recursively creates all possible output data for <difficulty> number of neurons.
    Values can only be 0 or 1
    :return:
    """
    if difficulty == 1:
        return [[0], [1]]
    else:
        to_return = []
        from_recursion = recurseData(difficulty - 1)
        for a in from_recursion:

            to_return.append(a + [0])
            to_return.append(a + [1])
        return to_return

def straightAcross() -> (Network, list[list[int, int]], list[list[int, int]]):
    """
    Network that has data that should only work straight across
    :return:
    """
    difficulty = 7
    # proportion_training = 1
    proportion_training = 0.8
    c = Network(difficulty, difficulty, 1, difficulty)
    # c._layers[0][0]._bias = 0
    # c._layers[0][1]._bias = 0
    # c._layers[1][0]._bias = -1
    # c._layers[1][1]._bias = -1
    #
    # c._layers[0][0].postConnections[0]._weight = 2
    # c._layers[0][0].postConnections[1]._weight = 0
    # c._layers[0][1].postConnections[0]._weight = 0
    # c._layers[0][1].postConnections[1]._weight = 2
    # data1 = [[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 1], [0, 1]], [[1, 0], [1, 0]]]

    all_possible = recurseData(difficulty)
    all_data = []
    for a in all_possible:
        pair = [a, a]
        all_data.append(pair)

    random.shuffle(all_data)
    cutoff = len(all_data) * proportion_training
    cutoff = int(cutoff)
    training = all_data[:cutoff]
    test = all_data[cutoff:]

    return c, training, test

def superSimple() -> (Network, list[list[int, int]]):
    """
    As simple as it gets. One input, one output, one connection
    :return:
    """
    c = Network(1, 1, 0, 1)
    # c._layers[0][0]._bias = 0
    # c._layers[1][0]._bias = 0
    #
    # c._layers[0][0].postConnections[0]._weight = 1

    data1 = [[[0], [0]], [[1], [1]]]
    return c, data1

def superSimpleInverse() -> (Network, list[list[int, int]]):
    """
    As simple as it gets. One input, one output, one connection
    :return:
    """
    c = Network(1, 1, 0, 1)
    # c._layers[0][0]._bias = 0
    # c._layers[1][0]._bias = 1
    #
    # c._layers[0][0].postConnections[0]._weight = -2

    data1 = [[[0], [1]], [[1], [0]]]
    return c, data1




# c = xOR_primedNet()
# c = xOR_solvedNet()
# c, data1, data2 = straightAcross()
# c, data1 = superSimple()
# c, data1 = superSimpleInverse()

dataIndex = 0

acc_list = []
fps = 0

# for line in history:
#     print(line[:-1])
#
# networks = []
# for nLayers in range(1, 4):
#     for nWidths in range(1, 6):
#         networks.append([2, 1, nLayers, nWidths])
#
# netNumber = 0
# currentNetwork = networks[0]
# c = Network(networks[0][0], networks[0][1], networks[0][2], networks[0][3])
# trainingTime = 120
# dataSet = 'XOR'
# history.write('\n \n')
# history.write('*** ' + str(trainingTime) + 'sec train time - ' + str(len(networks)) + ' networks - ' + dataSet + ' ***\n')


# start_trial = time.time()


# this makes it so that the main algorithm can be tested
if __name__ == '__main__':
    running = True

else:
    running = False

# FPS helper
start = 0
end = 0

# track whether mouse is down or up
down = False
# track whether spacebar has been clicked
space = False
tempPrintOut = debug.print_out
while running:
    start = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # track if user clicks or presses space
        if event.type == pygame.MOUSEBUTTONDOWN:
            down = True
        if event.type == pygame.MOUSEBUTTONUP:
            down = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            space = True
            tempPrintOut = debug.print_out
            debug.toggle_print(True)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            tempPrintOut = debug.print_out
            debug.toggle_print(True)
            debug.p(f'test case')
            # display activations in a test case
            # randomizes the data


            c._input(data1[dataIndex][0])
            c._runNetwork()
            c._runNetworkTensor()


            debug.p(c._accuracy)

            accText = font.render('accuracy: ' + str(round(c._accuracy, 2)), True, (255, 255, 255), BACKGROUND)
            accTextRect = accText.get_rect()
            accTextRect.bottomleft = (50, SCREEN_Y - 50)

            dataIndex += 1
            if dataIndex == len(data1):
                dataIndex = 0
                random.shuffle(data1)

            screen.fill(BACKGROUND)
            c.drawOnScreen(screen, i, 'a')
            screen.blit(text, textRect)
            screen.blit(accText, accTextRect)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
            # updates the accuracy based on the networks test data!
            all_results = []
            for label in data2:
                c._input(label[0])
                res1 = c._findAccuracy(label[1], 0.5)
                if 0 in res1:
                    all_results.append(0)
                else:
                    all_results.append(1)
            test_accuracy = sum(all_results) / len(all_results)
            print(round(test_accuracy, 4))


    # only proceed if we're not in debug mode, or if we received a click or space from the user
    if ((not DEBUG) or (down)) or (space):

        # randomizes the data
        dataIndex += 1
        if dataIndex == len(data1):
            dataIndex = 0
            random.shuffle(data1)

        c._input(data1[dataIndex][0])
        c._runNetwork()
        c._step(data1[dataIndex][1])


        for out in c._layers[-1]:
            # print(out._activation)
            pass


        i += 0.1
        if i >= 1.0:
            i -= 1
            debug.p(c._accuracy)
            accText = font.render('accuracy: ' + str(round(c._accuracy, 2)), True, (255, 255, 255), BACKGROUND)
            accTextRect = accText.get_rect()
            accTextRect.bottomleft = (50, SCREEN_Y - 50)
            # print what the FPS max COULD be
            text = font.render('FPP: ' + str(round(fps)), True, (255, 255, 255), BACKGROUND)
            textRect = text.get_rect()
            textRect.center = (SCREEN_X - 100, SCREEN_Y - 50)

            debug.toggle_print(True)
            print(f'CPU time: {tCPU.avgDuration()}')
            print(f'GPU time: {tGPU.avgDuration()}')
            print(f'misc time: {tMisc.avgDuration()}')
            debug.toggle_print(False)

        space = False

        screen.fill(BACKGROUND)
        # c.drawOnScreen(screen, i)
        # screen.blit(text, textRect)
        # screen.blit(accText, accTextRect)

    debug.toggle_print(tempPrintOut)


    # end_trial = time.time()

    # if end_trial - start_trial >= trainingTime:
    #     start_trial = time.time()
    #     acc = c._highAcc
    #     curr = c._accuracy
    #     steps = c._steps
    #     history.write('H: '+ "%.2f" % acc + ' | C:' + "%.2f" % curr + " | S: {:4.0f}".format(steps) + ' - Width: ' + str(networks[netNumber][2]) + ' - Num Neurons: ' + str(networks[netNumber][3]) + '\n')
    #     if netNumber < len(networks) - 1:
    #         netNumber += 1
    #     else:
    #         running = False
    #     c = Network(networks[netNumber][0], networks[netNumber][1], networks[netNumber][2], networks[netNumber][3])


    # FPS Adjust
    end = time.time()
    diff = end - start
    if diff != 0:
        fps = 1 / diff
    waitTime = 1/FPS - diff

    if waitTime > 0:
        time.sleep(waitTime)
    else:
        pygame.draw.line(screen, (255, 0, 0), (SCREEN_X - 100, SCREEN_Y - 100), (SCREEN_X, SCREEN_Y), 10)
    pygame.display.flip()

