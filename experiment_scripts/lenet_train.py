from util import modeling
import tensorflow as tf
import networkx
import pandas as pd
import numpy as np
import math
import json
import gc


def main():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    """
    for each iteration
        initialize model
        calculate katz centrality
        train a baseline model
        save model
        for each weight percentage
            lottery ticket pruning and train (save mask)
            katz pruning and train (save mask)
            random weight pruning and train (save mask)
            random unit pruning and train (save mask)
            write early stopping epoch and accuracy to csv
    """
    BUFFER_SIZE = 100000
    BATCH_SIZE = 60

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape [0],-1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    stats = pd.DataFrame([],columns=['iteration','method','num_parameters','num_units','epochs','val_loss','val_accuracy'])

    for iteration in range(10):
        tf.random.set_seed(iteration)
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(300,activation='relu',name='dense_300',kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(100,activation='relu',name='dense_100',kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(10,activation='softmax',name='dense_10',kernel_initializer='glorot_normal')
        ])
        model.build((784,))
        model.save('../models/initialized/lenet_iteration_'+str(iteration))

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
        model.save('../models/trained/lenet_iteration_'+str(iteration))
        json.dump(history,open('../histories/lenet_iteration_'+str(iteration)+'.json','w'))
	
        best_epoch = np.argmin(history['val_loss'])
        stats.loc[len(stats)] = [iteration,'intact',None,None,best_epoch,history['val_loss'][best_epoch],history['val_accuracy'][best_epoch]]
        stats.to_csv('../summary_stats/lenet_intact.csv')
        
        del model
        gc.collect()

if __name__=='__main__':
    main()
