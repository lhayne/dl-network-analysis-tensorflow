from util import modeling
import tensorflow as tf
import networkx
import pandas as pd
import math
import numpy as np
import json
import gc

def main():
    BATCH_SIZE = 60

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape[0],-1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    stats = pd.DataFrame([],columns=['iteration','method','num_parameters','num_units','epochs','val_loss','val_accuracy'])

    for iteration in range(20):
        for keep_exponent in range(1,25):
            percent_to_keep = (1/(5/4)**keep_exponent)

            init_model = tf.keras.models.load_model('../models/initialized/lenet_iteration_'+str(iteration))

            # Katz unit pruning
            # calculate katz centrality
            digraph_model = modeling.DiGraphModel(inputs=init_model.inputs,outputs=init_model.outputs)
            unit_graph = digraph_model.get_networkx_graph()
            katz_centrality = networkx.katz_centrality(unit_graph)
            
            # generate masks
            dense_300_centrality = [katz_centrality['dense_300.'+str(i)] for i in range(300)]
            num_kept_mask_300 = math.ceil(len(dense_300_centrality) * percent_to_keep)
            dense_300_mask = modeling.percentHighestMask(dense_300_centrality,percent_to_keep)
            
            dense_100_centrality = [katz_centrality['dense_100.'+str(i)] for i in range(100)]
            num_kept_mask_100 = math.ceil(len(dense_100_centrality) * percent_to_keep)
            dense_100_mask = modeling.percentHighestMask(dense_100_centrality,percent_to_keep)

            # create model with applied unit mask
            model = tf.keras.Sequential(
                tf.keras.Input(shape=(784,)),
                tf.keras.layers.Dense(300,activation='relu'),
                modeling.UnitMaskLayer(name='mask_300'),
                tf.keras.layers.Dense(100,activation='relu'),
                modeling.UnitMaskLayer(name='mask_100'),
                tf.keras.layers.Dense(10,activation='softmax')
            )
            model.get_layer('mask_300').set_weight(dense_300_mask)
            model.get_layer('mask_100').set_weight(dense_100_mask)
            
            # load initial weights
            model.get_layer('dense_300').set_weights(init_model.get_layer('dense_300').get_weights())
            model.get_layer('dense_100').set_weights(init_model.get_layer('dense_100').get_weights())
            model.get_layer('dense_10').set_weights(init_model.get_layer('dense_10').get_weights())
            # train and save model
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012), metrics=['accuracy'])
            history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=5000,steps_per_epoch=10,shuffle=True,callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            mode='min',
                                            min_delta=0,
                                            patience=1000,
                                            restore_best_weights=True,
                    )])
            history = history.history
            model.save('../models/trained/katz_percent_'+str(keep_exponent)+'iteration_'+str(iteration))
            json.dump(history,open('../histories/katz_percent_'+str(keep_exponent)+'iteration_'+str(iteration)))

            best_epoch = np.argmin(history['val_loss'])
            stats.loc[len(stats)] = [iteration,'katz',None,(num_kept_mask_300,num_kept_mask_100),best_epoch,history['val_loss'][best_epoch],history['val_accuracy'][best_epoch]]

            del model
            gc.clear_session()


if __name__=='__main__':
    main()