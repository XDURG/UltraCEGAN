from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import callbacks
import numpy as np
import os

im_height = 256
im_width = 256
batch_size = 128
epochs = 100

train_dir = './sythetic_ceus_train'
validation_dir = './sythetic_ceus_val'
model_dir = './model_dir'
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=5,
                                           width_shift_range=5,
                                           height_shift_range=5,
                                           shear_range=5,
                                           zoom_range=0,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')

total_train = train_data_gen.n
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')

total_val = val_data_gen.n
covn_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(im_height, im_width, 3))
covn_base.trainable = True
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D(name="glp_1"))
model.add(tf.keras.layers.Dense(2, activation='softmax', name="dense_1"))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
checkpointer = callbacks.ModelCheckpoint(os.path.join(model_dir, 'model_{epoch:03d}.hdf5'),
                                   verbose=1, save_weights_only=False, period=1)
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=[checkpointer])

accy = history.history['accuracy']
lossy = history.history['loss']
np_out = np.concatenate([np.array(accy).reshape((1, len(accy))), np.array(lossy).reshape((1, len(lossy)))], axis=0)
np.savetxt('./model_log.txt', np_out)
print("File saved successfully")