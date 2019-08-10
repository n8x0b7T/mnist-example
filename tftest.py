import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import random

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

model_exists = os.path.isfile("./num.model")
if not model_exists:
    print("Create new model...")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=3)

    val_loss, val_acc = model.evaluate(x_train, y_train)
    print(val_loss, val_acc)

    model.save("num.model")
else:
    print("Load existing...")
    model = tf.keras.models.load_model("num.model")

# plt.imshow(x_train[7], cmap=plt.cm.binary)
# plt.show()

predictions = model.predict(x_test)

pick_sample = random.randint(0, len(predictions) - 1)

print("Prediction:")
print(np.argmax(predictions[pick_sample]))
plt.imshow(x_test[pick_sample])
plt.show()


# print(np.argmax(predictions[0]))
# plt.imshow(x_test[0)]
# plt.show()


print("Done")
