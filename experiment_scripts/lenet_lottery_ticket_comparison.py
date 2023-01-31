from util import modeling
import tensorflow as tf
import networkx
import pandas as pd
import math

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
    x_train = x_train.reshape(x_train.shape[0] + (-1,))
    x_test = x_test.reshape(x_test.shape [0]+ (-1,))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    stats = pd.DataFrame([],columns=['iteration','method','num_parameters','num_units','epochs','val_loss','val_accuracy'])

    for iteration in range(10):
        model = tf.keras.Sequential(
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(300,activation='Relu',name='dense_300'),
            tf.keras.layers.Dense(100,activation='Relu',name='dense_100'),
            tf.keras.layers.Dense(10,name='dense_10')
        )

        model.save('../models/initialized/lenet_iteration_'+str(iteration))

        digraph_model = modeling.DiGraphModel(model.inputs,model.outputs)
        unit_graph = digraph_model.get_networkx_graph()
        katz_centrality = networkx.katz_centrality(unit_graph)

        digraph_model.compile(loss="categorical_crossentropy", 
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012), metrics=['accuracy'])

        history = digraph_model.fit(x_train,x_test,batch_size=BATCH_SIZE,epochs=5000,steps_per_epoch=10,shuffle=True,callbacks=[
                tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        mode='min',
                                        min_delta=0,
                                        patience=1000,
                                        restore_best_weights=True,
                )])
        
        model.save('../models/trained/lenet_iteration_'+str(iteration))

        for keep_exponent in range(1,25):
            # Lottery ticket
            # load trained model
            trained_model = tf.keras.models.load_model('../models/trained/lenet_iteration_'+str(iteration))
            # select lowest magnitude weights and generate mask
            dense_300_kernel = trained_model.get_layer('dense_300').get_weights()[0]
            percent_to_keep = (1/(5/4)**keep_exponent)
            dense_300_mask = percentHighestMask(dense_300_kernel,percent_to_keep)

            dense_100_kernel = trained_model.get_layer('dense_100').get_weights()[0]
            percent_to_keep = (1/(5/4)**keep_exponent)
            dense_100_mask = percentHighestMask(dense_100_kernel,percent_to_keep)

            dense_10_kernel = trained_model.get_layer('dense_10').get_weights()[0]
            percent_to_keep = (1/(10/9)**keep_exponent)
            dense_10_mask = percentHighestMask(dense_10_kernel,percent_to_keep)

            # create model with applied kernel mask
            model = tf.keras.Sequential(
                tf.keras.Input(shape=(784,)),
                tf.keras.layers.Dense(300,activation='Relu',kernel_constraint=modeling.KernelMask(dense_300_mask)),
                tf.keras.layers.Dense(100,activation='Relu',kernel_constraint=modeling.KernelMask(dense_100_mask)),
                tf.keras.layers.Dense(10,kernel_constraint=modeling.KernelMask(dense_10_mask))
            )

            # load initial weights
            init_model = tf.keras.models.load_model('../models/initialized/lenet_iteration_'+str(iteration))
            model.get_layer('dense_300').set_weights(init_model.get_layer('dense_300').get_weights())
            model.get_layer('dense_100').set_weights(init_model.get_layer('dense_100').get_weights())
            model.get_layer('dense_10').set_weights(init_model.get_layer('dense_10').get_weights())
            
            # train and save model
            model.compile(loss="categorical_crossentropy", 
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012), metrics=['accuracy'])
            history = model.fit(x_train,x_test,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=5000,steps_per_epoch=10,shuffle=True,callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            mode='min',
                                            min_delta=0,
                                            patience=1000,
                                            restore_best_weights=True,
                    )])

            # Katz unit pruning
            # select lowest magnitude katz nodes
            # generate mask
            # create model with applied unit mask
            # load initial weights
            # train and save model

            # Random weight pruning
            # select random weights
            # generate mask
            # create model with applied kernel mask
            # load initial weights
            # train and save model

            # Random unit pruning
            # select random units
            # generate mask
            # create model with applied unit mask
            # load initial weights
            # train and save model



if __name__=='__main__':
    main()