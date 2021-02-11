import numpy as np
import tensorflow as tf

def __num_elems(shape):
    '''Returns the number of elements in the given shape

    Args:
        shape: TensorShape
    
    Return:
        tot_elems: int
    '''
    tot_elems = 1
    for s in shape:
        tot_elems *= int(s)
    return tot_elems

def graph_size(graph):
    '''Returns the size of the given graph in bytes

    The size of the graph is calculated by summing up the sizes of each
    trainable variable. The sizes of variables are calculated by multiplying
    the number of bytes in their dtype with their number of elements, captured
    in their shape attribute

    Args:
        graph: TF graph
    Return:
        integer representing size of graph (in bytes)
    '''
    tot_size = 0
    with graph.as_default():
        vs = tf.trainable_variables()
        for v in vs:
            tot_elems = __num_elems(v.shape)
            dtype_size = int(v.dtype.size)
            var_size = tot_elems * dtype_size
            tot_size += var_size
    return tot_size


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0] # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array

    return np.sum(np.square(client_grads))