import tensorflow as tf
import numpy as np
from tensorflow import keras
from models.resnet import resnet_18

# In[1]:
tf.random.set_seed(22)
np.random.seed(22)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
# [b, 28, 28] => [b, 28, 28, 1]
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
y_train_ohe = tf.one_hot(y_train, depth=10).numpy()
y_test_ohe = tf.one_hot(y_test, depth=10).numpy()

# In[2]:
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[3]:
def main():
    num_classes = 10
    batch_size = 128
    epochs = 30

    # build model and optimizer
    model = resnet_18(num_classes=num_classes)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.build(input_shape=(None, 28, 28, 1))
    print("Number of variables in the model :", len(model.variables))
    model.summary()

    # train
    model.fit(x_train, y_train_ohe,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test_ohe),
              verbose=2)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=2)
    print("Final test loss and accuracy :", scores)


if __name__ == '__main__':
    main()
