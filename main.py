import numpy as np

learning_speed = 15
epoch_count = 1024
set_count = 1
gradient_moment = 1
layer_count = 3
layers_node_count = [2, 2, 1]
displacement_nodes = [False] * 2


def sigmoid(np_matrix, derivative=False):
    if derivative:
        return (1 - np_matrix) * np_matrix
    else:
        return 1 / (1 + np.exp(-np_matrix))


# init synapses
synapses = []
for i in range(layer_count - 1):
    synapses.append(4*np.random.random((layers_node_count[i],
                                        layers_node_count[i + 1]))-2)

learning_input = open("learning_input", "r")

for epoch in range(epoch_count):
    current_error = 0

    for current_set in range(set_count):
        nodes = []
        input_data = [int(i) for i in learning_input.readline().split()]

        nodes.append(np.array(input_data[:layers_node_count[0]]))

        out_ideal = np.array(input_data[layers_node_count[0]:])

        for current_layer in range(1, layer_count):
            nodes.append(sigmoid(np.dot(nodes[current_layer - 1],
                                        synapses[current_layer - 1])))

        current_error = sum((np.array(nodes[-1]) - np.array(out_ideal)) ** 2)
        current_error /= layers_node_count[-1]

        delta = [(out_ideal - nodes[-1])*sigmoid(nodes[-1], True)]
        for current_delta in range(layer_count-2, -1, -1):
            delta.append(sigmoid(nodes[current_delta], True) *
                         np.dot(delta[-1], synapses[current_delta].T))

            a_shape, = nodes[current_delta].shape
            b_shape, = delta[-2].shape
            synapses[current_delta] += learning_speed*np.dot(nodes[current_delta].reshape((a_shape,
                                                                                          1)),
                                                             delta[-2].reshape((1, b_shape)))

        print(nodes[0])
        print(synapses)
        print(nodes[-1])
        print(current_error)

    current_error /= set_count


learning_input.close()
