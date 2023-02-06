from util import modeling
import tensorflow as tf
import networkx
import pandas as pd
import math
import numpy as np
import json
import gc
from collections import Counter
import pickle

def minmax(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def main():
    BATCH_SIZE = 60

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape[0],-1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

    stats = pd.DataFrame([],columns=['iteration','method','keep_percent','num_parameters','num_units','epochs','val_loss','val_accuracy'])

    for iteration in range(10):
        for keep_exponent in range(1,25):
            percent_to_keep = (1/(5/4)**keep_exponent)

            init_model = tf.keras.models.load_model('../models/initialized/lenet_iteration_'+str(iteration))

            init_model = tf.keras.models.load_model('../models/initialized/lenet_iteration_'+str(iteration))

            # Katz unit pruning
            # calculate katz centrality
            digraph_model = modeling.DiGraphModel(inputs=init_model.inputs,outputs=init_model.outputs)
            unit_graph = digraph_model.get_networkx_graph(abs=True) # We want the absolute valued weights
            edge_weights = networkx.get_edge_attributes(unit_graph,'weight')
            assert np.alltrue(np.asarray([edge_weights[k] for k in edge_weights.keys()]) > 0)
            katz_centrality = networkx.katz_centrality(unit_graph,weight='weight')

            dense_300_centrality = np.asarray([katz_centrality['dense_300.'+str(i)] for i in range(300)])
            dense_300_centrality = minmax(dense_300_centrality)
            dense_100_centrality = np.asarray([katz_centrality['dense_100.'+str(i)] for i in range(100)])
            dense_100_centrality = minmax(dense_100_centrality)

            dense_300_activations = modeling.get_activations(init_model,'dense_300',x_train) # N x 300
            print(dense_300_activations.shape)
            dense_300_selectivities = modeling.selectivity(dense_300_activations,y_train_one_hot)
            class_occurances = Counter(np.argmax(dense_300_selectivities,axis=0))
            print(class_occurances)
            pickle.dump(class_occurances,open('../masks/kept_classes_dense_300_iteration_'+str(iteration)+'_exponent_'+str(keep_exponent)+'.pkl','wb'))

            dense_300_selectivities = np.max(dense_300_selectivities,axis=0) # maximally selective class
            dense_300_selectivities = minmax(dense_300_selectivities)
            print(np.min(dense_300_selectivities),np.max(dense_300_selectivities),dense_300_selectivities.shape)

            dense_100_activations = modeling.get_activations(init_model,'dense_100',x_train) # N x 100
            print(dense_100_activations.shape)
            dense_100_selectivities = modeling.selectivity(dense_100_activations,y_train_one_hot) 
            class_occurances = Counter(np.argmax(dense_100_selectivities,axis=0))
            print(class_occurances)
            pickle.dump(class_occurances,open('../masks/kept_classes_dense_100_iteration_'+str(iteration)+'_exponent_'+str(keep_exponent)+'.pkl','wb'))

            dense_100_selectivities = np.max(dense_100_selectivities,axis=0) # maximally selective class
            dense_100_selectivities = minmax(dense_100_selectivities)
            print(np.min(dense_100_selectivities),np.max(dense_100_selectivities))

            num_kept_mask_300 = math.ceil(300 * percent_to_keep)
            num_kept_mask_100 = math.ceil(100 * percent_to_keep)

            # Selectivity + Centrality based unit pruning
            dense_300_mask = modeling.percent_highest_mask(dense_300_centrality +
                                                           dense_300_selectivities,percent_to_keep)
            dense_300_mask = np.reshape(dense_300_mask,(1,300))

            dense_100_mask = modeling.percent_highest_mask(dense_100_centrality +
                                                           dense_100_selectivities,percent_to_keep)
            dense_100_mask = np.reshape(dense_100_mask,(1,100))

            # create model with applied unit mask
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(784,)),
                tf.keras.layers.Dense(300,activation='relu',name='dense_300'),
                modeling.UnitMaskLayer(name='mask_300'),
                tf.keras.layers.Dense(100,activation='relu',name='dense_100'),
                modeling.UnitMaskLayer(name='mask_100'),
                tf.keras.layers.Dense(10,activation='softmax',name='dense_10')
            ])
            model.get_layer('mask_300').set_weights([dense_300_mask])
            model.get_layer('mask_100').set_weights([dense_100_mask])
            
            # load initial weights
            model.get_layer('dense_300').set_weights(init_model.get_layer('dense_300').get_weights())
            model.get_layer('dense_100').set_weights(init_model.get_layer('dense_100').get_weights())
            model.get_layer('dense_10').set_weights(init_model.get_layer('dense_10').get_weights())
            # train and save model
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012), metrics=['accuracy'])
            history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=500,steps_per_epoch=100,shuffle=True,callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            mode='min',
                                            min_delta=0,
                                            patience=200,
                                            restore_best_weights=True,
                    )])
            history = history.history
            model.save('../models/trained/katz_selectivity_percent_'+str(keep_exponent)+'_iteration_'+str(iteration))
            json.dump(history,open('../histories/katz_selectivity_percent_'+str(keep_exponent)+'_iteration_'+str(iteration)+'.json','w'))

            best_epoch = np.argmin(history['val_loss'])
            stats.loc[len(stats)] = [iteration,'selectivity',percent_to_keep,None,(num_kept_mask_300,num_kept_mask_100),best_epoch,history['val_loss'][best_epoch],history['val_accuracy'][best_epoch]]
            stats.to_csv('../summary_stats/lenet_katz_selectivity.csv')

            del model
            del init_model
            gc.collect()

if __name__=='__main__':
    main()
