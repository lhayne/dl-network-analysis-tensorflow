from util import modeling
import tensorflow as tf
import networkx
import pandas as pd
import math
import numpy as np
import json

def percentHighestMask(array,percent_to_keep):
    """
    Generates a mask of same shape as 'array' with 'percent_to_keep' percent highest values in array set to one and rest set to zero.
    """
    num_kept = math.ceil(array.size() * percent_to_keep)
    kept_indices = np.argpartition(array,-num_kept,axis=None)[-num_kept:]
    mask = np.zeros((array.size()))
    mask[kept_indices] = 1
    mask = np.reshape(mask,array.shape)
    return mask

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

            num_kept_mask_300 = math.ceil(300 * percent_to_keep)
            num_kept_mask_100 = math.ceil(100 * percent_to_keep)

            # Random unit pruning
            # select random units and generate mask
            dense_300_mask = np.zeros((dense_300_mask.size(),))
            dense_300_mask[np.random.choice(np.arange(dense_300_mask.size()),num_kept_mask_300)] = 1
            dense_300_mask.reshape(dense_300_mask.shape)
            
            dense_100_mask = np.zeros((dense_100_mask.size(),))
            dense_100_mask[np.random.choice(np.arange(dense_100_mask.size()),num_kept_mask_100)] = 1
            dense_100_mask.reshape(dense_100_mask.shape)

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
            model.save('../models/trained/random_units_percent_'+str(keep_exponent)+'iteration_'+str(iteration))
            json.dump(history,open('../histories/random_units_percent_'+str(keep_exponent)+'iteration_'+str(iteration)))

            best_epoch = np.argmin(history['val_loss'])
            stats.loc[len(stats)] = [iteration,'random_units',None,(num_kept_mask_300,num_kept_mask_100),best_epoch,history['val_loss'][best_epoch],history['val_accuracy'][best_epoch]]



if __name__=='__main__':
    main()