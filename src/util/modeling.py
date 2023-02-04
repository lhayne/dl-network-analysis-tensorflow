import tensorflow as tf
import numpy as np
import networkx
from re import split
import itertools
import pydot
import math

class DiGraphModel(tf.keras.Model):
    """
    Model which inherits from tf.keras.Model and computes
    directed graph of units in network.
    """
    def __init__(self,*args,**kwargs):
        super(DiGraphModel, self).__init__(*args,**kwargs)
        dot_graph = tf.keras.utils.model_to_dot(self,show_shapes=True)
        self.networkx_model_graph = networkx.drawing.nx_pydot.from_pydot(dot_graph)
        self.unit_graph = None
        
    def get_networkx_graph(self,abs=False):
        model_node_attributes = networkx.get_node_attributes(self.networkx_model_graph,'label')
        model_node_attributes = {node_id:label.replace('\n','') for node_id,label in model_node_attributes.items()}
        model_node_attributes = {node_id:label.replace(' ','') for node_id,label in model_node_attributes.items()}
        model_node_attributes = {node_id:label.replace('|','') for node_id,label in model_node_attributes.items()}
        self.model_node_attributes = model_node_attributes
        
        self.unit_graph = networkx.DiGraph()
        layer_types = {node_id:split('\:|\{|\}',label)[1] for node_id,label in model_node_attributes.items()}
        if 'InputLayer' in layer_types.values():
            node_id = list(layer_types.keys())[list(layer_types.values()).index('InputLayer')]
            self._construct_unit_graph(node_id,abs=abs)
        else:
            raise Exception('No InputLayer in Model')

        return self.unit_graph
    
    def _construct_unit_graph(self,input_node_id,abs=abs):
        for source,target in networkx.bfs_edges(self.networkx_model_graph,input_node_id):
            source_label = self.model_node_attributes[source]
            source_layer_name = split('\:|\{|\}',source_label)[0]
            source_output_shape_string = split('\:|\{|\}',source_label)[9]
            source_output_shape = self._parse_shape_string(source_output_shape_string)
            
            target_label = self.model_node_attributes[target]
            target_layer_name = split('\:|\{|\}',target_label)[0]
            target_layer_type = split('\:|\{|\}',target_label)[1]
            target_output_shape_string = split('\:|\{|\}',target_label)[9]
            target_output_shape = self._parse_shape_string(target_output_shape_string)
            
            if target_layer_type == 'Dense':
                self._add_Dense_to_graph(source_layer_name,source_output_shape,
                                         target_layer_name,target_output_shape,
                                         abs=abs)
            else:
                raise Exception(target_layer_type,'not currently supported')
    
    def _add_Dense_to_graph(self,source_layer_name,source_output_shape,
                                 target_layer_name,target_output_shape,
                                 abs=abs):
        source_nodes = [source_layer_name+'.'+str(i) for i in range(source_output_shape[-1])]
        target_nodes = [target_layer_name+'.'+str(i) for i in range(target_output_shape[-1])]
        edges = list(itertools.product(source_nodes,target_nodes))
        
        weights = np.ravel(self.get_layer(target_layer_name).weights[0])
        
        if abs:
            weights = np.abs(weights)
        
        edge_weights = {k:weights[i] for i,k in enumerate(edges)}
        
        self.unit_graph.add_edges_from(edges)
        networkx.set_edge_attributes(self.unit_graph,values=edge_weights,name='weight')
        
    def _parse_shape_string(self,shape_string):
        elements = split('\[|\]|\(|\)|\,|\ ',shape_string)
        shape = []
        for e in elements:
            if e == 'None':
                shape.append(None)
            elif e.isdigit():
                shape.append(int(e))
        return shape


class KernelMaskConstraint(tf.keras.constraints.Constraint):
    """
    Kernel constraint which allows for pruning kernel weights using
    binary mask of zeros and ones.
    """
    def __init__(self,mask):
        self.mask = tf.convert_to_tensor(mask,dtype=tf.float32)

    def __call__(self, w):
        return w * self.mask


class UnitMaskLayer(tf.keras.layers.Layer):
    """
    Mask layer which applies binary mask to post activations in a network.
    """
    def __init__(self, **kwargs):
        super(UnitMaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer=tf.keras.initializers.Ones(),
                                      trainable=False)
        super(UnitMaskLayer, self).build(input_shape)

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
        
        
def percent_highest_mask(array,keep_percent,abs=True):
    """
    Generates a mask of same shape as 'array' with 'keep_percent' 
    percent highest values in array set to one and rest set to zero.

    Parameters:
    ----------
    array (numpy array):
        N-d array to generate mask from
    keep_percent (float):
        The percentage of elements to keep
    abs (bool):
        Whether to take the absolute value of array elements before
        computing mask

    Returns:
    -------
    mask (numpy array):
        Binary N-d array of same shape as array with elements to keep
        set to one and everything else set to zero
    """
    # calculate number of elements to keep (K)
    assert (keep_percent >= 0 and keep_percent <= 1)
    K = math.ceil(array.size * keep_percent)

    # partition the indices of the elements and select the K largest
    if abs:
        kept_indices = np.argpartition(np.abs(array),-K,axis=None)[-K:]
    else:
        kept_indices = np.argpartition(array,-K,axis=None)[-K:]

    # create array of zeros with kept_indices set to 1
    mask = np.zeros((array.size,))
    mask[kept_indices] = 1
    mask = np.reshape(mask,array.shape)

    return mask
