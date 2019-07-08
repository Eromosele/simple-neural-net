import numpy as np


def sigmoid(x, der=False):
    if der is True:
        return sigmoid(x)*(1-sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


# number of hidden nodes
n_layer_nodes = 8
N_OUTPUT_NODES = 1

# the input is limited to just 1 example and 1 output
x = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
y = np.array([[0], [1], [1], [0]])

derived_error = 0

# weights for the layer between the input and hidden layer
weights = np.random.random((x.shape[1], n_layer_nodes))

# weights between the hidden layer and the output layer
weights_output = np.random.random((n_layer_nodes, N_OUTPUT_NODES))

# train
for i in range(500):

    # value of the weighted sums for the hidden nodes
    z1 = np.dot(x, weights)
    # squash z1
    a1 = sigmoid(z1)
    derived_a1 = sigmoid(z1, True)

    # value of the weighted sums for the output layer
    z2 = np.dot(a1, weights_output)
    # squashed outputs
    output = sigmoid(z2)  # return a single output

    # calculate errors
    error = (1/2)*(y-output)**2
    derived_error = y-output

    # backpropagate and update the weights
    scalar = derived_error * sigmoid(z2, True)

    # update weights with the derived function
    weights_output += np.dot(scalar.T,a1).T
    weights += (np.dot((np.dot(scalar,weights_output.T) * derived_a1).T, x)).T
    print('error: %s' % str(np.mean(np.abs(error))))
print('prediction: %s' % output)


# test
input = np.array([[1, 0]])
# value of the weighted sums for the hidden nodes
z1 = np.dot(input, weights)
# squash z1
a1 = sigmoid(z1)
derived_a1 = sigmoid(z1, True)

# value of the weighted sums for the output layer
z2 = np.dot(a1, weights_output)
# squashed outputs
predoutput = sigmoid(z2)  # return a single output

print(predoutput)
