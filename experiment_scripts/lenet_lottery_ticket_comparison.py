from util import modeling
import tensorflow as tf

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0] + (-1,))
    x_test = x_test.reshape(x_test.shape [0]+ (-1,))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = tf.keras.Sequential(
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(300,activation='Relu'),
        tf.keras.layers.Dense(100,activation='Relu'),
        tf.keras.layers.Dense(10)
    )

    

if __name__=='__main__':
    main()