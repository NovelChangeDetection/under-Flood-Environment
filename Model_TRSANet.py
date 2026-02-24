import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from Evaluation import net_evaluation


# Spatial Attention
def spatial_attention_module(inputs):
    avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return inputs * attention


# Transformer Encoder
def transformer_block(x, num_heads, ff_dim):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn = layers.Dropout(0.1)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ffn = layers.Dense(ff_dim, activation='relu')(out1)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(0.1)(ffn)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)


# TRSANet model builder
def build_trsanet(input_shape, num_classes=2):
    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)

    def extractor(inp):
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        return x

    f1 = extractor(in1)
    f2 = extractor(in2)

    diff = layers.Subtract()([f1, f2])
    res = layers.Conv2D(64, 3, padding='same', activation='relu')(diff)
    res = layers.Add()([res, diff])
    res = spatial_attention_module(res)

    x = layers.Reshape((-1, res.shape[-1]))(res)
    x = transformer_block(x, num_heads=4, ff_dim=64)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model([in1, in2], out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Reshaping helper to your format
def reshape_flat_images(flat_images, IMG_SIZE):
    reshaped = np.zeros((flat_images.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype='float32')
    for i in range(flat_images.shape[0]):
        temp = np.resize(flat_images[i], (IMG_SIZE * IMG_SIZE, 1))
        reshaped[i] = temp.reshape((IMG_SIZE, IMG_SIZE, 1))
    return reshaped / 255.0


# Full training on all data


def Model_TRSANet(train_img1, train_img2, train_labels, epochs=10, batch_size=64):
    IMG_SIZE = 28
    print("Training TRSANet on all images...")

    X1 = np.zeros((train_img1.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype='float32')
    for i in range(train_img1.shape[0]):
        temp = np.resize(train_img1[i], (IMG_SIZE * IMG_SIZE, 1))
        X1[i] = temp.reshape((IMG_SIZE, IMG_SIZE, 1))

    X2 = np.zeros((train_img2.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype='float32')
    for i in range(train_img2.shape[0]):
        temp = np.resize(train_img2[i], (IMG_SIZE * IMG_SIZE, 1))
        X2[i] = temp.reshape((IMG_SIZE, IMG_SIZE, 1))

    Y = tf.keras.utils.to_categorical(train_labels, num_classes=2)

    model = build_trsanet((IMG_SIZE, IMG_SIZE, 1), num_classes=2)
    model.fit([X1, X2], Y, epochs=epochs, batch_size=batch_size, verbose=1)

    preds = model.predict([X1, X2])
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(Y, axis=1)

    Eval = net_evaluation(true_labels, pred_labels)

    return Eval
