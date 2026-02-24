import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from Evaluation import net_evaluation


# U-Net Encoder Block
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x


# U-Net Model for Change Detection
def build_unet(input_shape, num_classes=2):
    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)

    def unet_encoder(inp):
        c1 = conv_block(inp, 32)
        p1 = layers.MaxPooling2D()(c1)
        c2 = conv_block(p1, 64)
        p2 = layers.MaxPooling2D()(c2)
        c3 = conv_block(p2, 128)
        p3 = layers.MaxPooling2D()(c3)
        c4 = conv_block(p3, 256)
        return c4

    f1 = unet_encoder(in1)
    f2 = unet_encoder(in2)

    diff = layers.Subtract()([f1, f2])
    x = layers.GlobalAveragePooling2D()(diff)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Training Function for U-Net Model
def Model_UNet(train_img1, train_img2, train_labels, epochs=10, batch_size=64):
    IMG_SIZE = 28
    print("Training U-Net on all images...")

    def reshape_images(data):
        reshaped = np.zeros((data.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype='float32')
        for i in range(data.shape[0]):
            temp = np.resize(data[i], (IMG_SIZE * IMG_SIZE, 1))
            reshaped[i] = temp.reshape((IMG_SIZE, IMG_SIZE, 1))
        return reshaped / 255.0

    X1 = reshape_images(train_img1)
    X2 = reshape_images(train_img2)
    Y = tf.keras.utils.to_categorical(train_labels, num_classes=2)

    model = build_unet((IMG_SIZE, IMG_SIZE, 1), num_classes=2)
    model.fit([X1, X2], Y, epochs=epochs, batch_size=batch_size, verbose=1)

    preds = model.predict([X1, X2])
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(Y, axis=1)

    Eval = net_evaluation(true_labels, pred_labels)
    return Eval
