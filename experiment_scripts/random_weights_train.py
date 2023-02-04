from util import modeling
import tensorflow as tf
import networkx
import pandas as pd
import math
import numpy as np
import json
import gc
import pickle

def main():
    BATCH_SIZE = 60

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape[0],-1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    stats = pd.DataFrame([],columns=['iteration','method','keep_percent','num_parameters','num_units','epochs','val_loss','val_accuracy'])

    for iteration in range(10):
        for keep_exponent in range(1,25):
            percent_to_keep = (1/(5/4)**keep_exponent)

            trained_model = tf.keras.models.load_model('../models/trained/lenet_iteration_'+str(iteration))

            dense_300_kernel = trained_model.get_layer('dense_300').get_weights()[0]
            num_kept_dense_300 = math.ceil(dense_300_kernel.size * percent_to_keep)

            dense_100_kernel = trained_model.get_layer('dense_100').get_weights()[0]
            num_kept_dense_100 = math.ceil(dense_100_kernel.size * percent_to_keep)

            # Random weight pruning
            # select random weights and generate mask
            dense_300_mask = np.zeros((dense_300_kernel.size,))
            dense_300_mask[np.random.choice(np.arange(dense_300_kernel.size),num_kept_dense_300)] = 1
            dense_300_mask = np.reshape(dense_300_mask,dense_300_kernel.shape)
            pickle.dump(dense_300_mask,open('../masks/random_weights_dense_300_percent_'+str(keep_exponent)+'_iteration_'+str(iteration)+'.pkl','wb'))

            dense_100_mask = np.zeros((dense_100_kernel.size,))
            dense_100_mask[np.random.choice(np.arange(dense_100_kernel.size),num_kept_dense_100)] = 1
            dense_100_mask = np.reshape(dense_100_mask,dense_100_kernel.shape)
            pickle.dump(dense_300_mask,open('../masks/random_weights_dense_100_percent_'+str(keep_exponent)+'_iteration_'+str(iteration)+'.pkl','wb'))

            # create model with applied kernel mask
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(784,)),
                tf.keras.layers.Dense(300,activation='relu',name='dense_300',kernel_constraint=modeling.KernelMaskConstraint(dense_300_mask)),
                tf.keras.layers.Dense(100,activation='relu',name='dense_100',kernel_constraint=modeling.KernelMaskConstraint(dense_100_mask)),
                tf.keras.layers.Dense(10,activation='softmax',name='dense_10')
            ])

            # load initial weights
            init_model = tf.keras.models.load_model('../models/initialized/lenet_iteration_'+str(iteration))
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
            model.save('../models/trained/random_weights_percent_'+str(keep_exponent)+'iteration_'+str(iteration))
            json.dump(history,open('../histories/random_weights_percent_'+str(keep_exponent)+'iteration_'+str(iteration),'w'))

            best_epoch = np.argmin(history['val_loss'])
            stats.loc[len(stats)] = [iteration,'random_weights',percent_to_keep,(num_kept_dense_300,num_kept_dense_100),None,best_epoch,history['val_loss'][best_epoch],history['val_accuracy'][best_epoch]]
            stats.to_csv('../summary_stats/lenet_random_weights.csv')
            
            del model
            del init_model
            gc.collect()

if __name__=='__main__':
    main()
