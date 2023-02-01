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

    stats = pd.DataFrame([],columns=['iteration','method','num_parameters','num_units','epochs','val_loss','val_accuracy'])

    for iteration in range(20):
        for keep_exponent in range(1,20):
            percent_to_keep = (1/(5/4)**keep_exponent)

            # Lottery ticket
            # load trained model
            trained_model = tf.keras.models.load_model('../models/trained/lenet_iteration_'+str(iteration))
            
            # select lowest magnitude weights and generate mask
            dense_300_kernel = trained_model.get_layer('dense_300').get_weights()[0]
            num_kept_dense_300 = math.ceil(dense_300_kernel.size * percent_to_keep)
            dense_300_mask = modeling.percent_highest_mask(dense_300_kernel,percent_to_keep)
            pickle.dump(dense_300_mask,open('../masks/lottery_ticket_dense_300_percent_'+str(keep_exponent)+'_iteration_'+str(iteration)+'.pkl','w'))

            dense_100_kernel = trained_model.get_layer('dense_100').get_weights()[0]
            num_kept_dense_100 = math.ceil(dense_100_kernel.size * percent_to_keep)
            dense_100_mask = modeling.percent_highest_mask(dense_100_kernel,percent_to_keep)
            pickle.dump(dense_100_mask,open('../masks/lottery_ticket_dense_100_percent_'+str(keep_exponent)+'_iteration_'+str(iteration)+'.pkl','w'))

            # dense_10_kernel = trained_model.get_layer('dense_10').get_weights()[0]
            # percent_to_keep = (1/(10/9)**keep_exponent)
            # num_kept_10 = math.ceil(dense_10_kernel.size() * percent_to_keep)
            # dense_10_mask = percent_highest_mask(dense_10_kernel,percent_to_keep)

            # create model with applied kernel mask
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(784,)),
                tf.keras.layers.Dense(300,name='dense_300',activation='relu',kernel_constraint=modeling.KernelMaskConstraint(dense_300_mask)),
                tf.keras.layers.Dense(100,name='dense_100',activation='relu',kernel_constraint=modeling.KernelMaskConstraint(dense_100_mask)),
                tf.keras.layers.Dense(10,name='dense_10',activation='softmax')
            ])
            model.build((784,))

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
            model.save('../models/trained/lottery_ticket_percent_'+str(keep_exponent)+'_iteration_'+str(iteration))
            json.dump(history,open('../histories/lottery_ticket_percent_'+str(keep_exponent)+'_iteration_'+str(iteration)+'.json','w'))

            best_epoch = np.argmin(history['val_loss'])
            stats.loc[len(stats)] = [iteration,'lottery_ticket',(num_kept_dense_300,num_kept_dense_100),None,best_epoch,history['val_loss'][best_epoch],history['val_accuracy'][best_epoch]]
            stats.to_csv('../summary_stats/lenet_lottery_ticket.csv')

            del model
            del trained_model
            del init_model
            gc.collect()


if __name__=='__main__':
    main()
