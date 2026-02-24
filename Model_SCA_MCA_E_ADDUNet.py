import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from Evaluation import net_evaluation
from tensorflow.keras.optimizers import Adam


# === Multi-Convolution Attention (MCA) Block ===
def mca_block(x):
    conv1 = layers.Conv2D(x.shape[-1], 1, padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(x.shape[-1], 3, padding='same', activation='relu')(x)
    conv5 = layers.Conv2D(x.shape[-1], 5, padding='same', activation='relu')(x)
    concat = layers.Concatenate()([conv1, conv3, conv5])
    attention = layers.Conv2D(x.shape[-1], 1, activation='sigmoid')(concat)
    return layers.Multiply()([x, attention])


# === Spatial Cross Attention (SCA) Block ===
def spatial_cross_attention(f1, f2):
    query = layers.Conv2D(f1.shape[-1], 1, padding='same')(f1)
    key = layers.Conv2D(f2.shape[-1], 1, padding='same')(f2)
    value = layers.Conv2D(f2.shape[-1], 1, padding='same')(f2)

    batch = tf.shape(query)[0]
    h, w, c = query.shape[1], query.shape[2], query.shape[3]
    query_flat = tf.reshape(query, (batch, -1, c))
    key_flat = tf.reshape(key, (batch, -1, c))
    value_flat = tf.reshape(value, (batch, -1, c))

    scores = tf.matmul(query_flat, key_flat, transpose_b=True)
    scores = tf.nn.softmax(scores, axis=-1)

    out = tf.matmul(scores, value_flat)
    out = tf.reshape(out, (batch, h, w, c))
    return layers.Add()([f1, out])


# === Dilated Dense Block ===
def dense_dilated_block(x, filters, dilation_rate):
    x1 = layers.Conv2D(filters, 3, dilation_rate=dilation_rate, padding='same', activation='relu')(x)
    x2 = layers.Concatenate()([x, x1])
    x3 = layers.Conv2D(filters, 3, dilation_rate=dilation_rate, padding='same', activation='relu')(x2)
    x4 = layers.Concatenate()([x2, x3])
    return x4


# === Encoder Block ===
def encoder_block(x, filters):
    x = dense_dilated_block(x, filters, dilation_rate=2)
    x = mca_block(x)
    p = layers.MaxPooling2D()(x)
    return x, p


# === Decoder Block ===
def decoder_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    skip = mca_block(skip)
    x = spatial_cross_attention(x, skip)
    x = layers.Concatenate()([x, skip])
    x = dense_dilated_block(x, filters, dilation_rate=2)
    return x


# === SCA-MCA-E-ADDUNet++ Final Model ===
def build_sca_mca_addunetpp(input_shape=(128, 128, 1), num_classes=1, hidden=5):
    inputs = Input(input_shape)

    e1, p1 = encoder_block(inputs, 32)
    e2, p2 = encoder_block(p1, 64)
    e3, p3 = encoder_block(p2, 128)
    e4, p4 = encoder_block(p3, 256)

    b = dense_dilated_block(p4, hidden, dilation_rate=4)

    d4 = decoder_block(b, e4, 256)
    d3 = decoder_block(d4, e3, hidden)
    d2 = decoder_block(d3, e2, 64)
    d1 = decoder_block(d2, e1, 32)

    out = layers.Conv2D(num_classes, 1, padding='same',
                        activation='sigmoid' if num_classes == 1 else 'softmax')(d1)

    model = models.Model(inputs, out)
    return model


# === Full Model Training & Evaluation ===
def Model_SCA_MCA_E_ADDUNet(Images1, Images2, train_labels, stepEpochs=10, sol=None):
    if sol is None:
        sol = [5, 0.01, 5]
    IMG_SIZE = 128  # Updated to match model input

    print("Training SCA-MCA-E-ADDUNet++ on all images...")

    def reshape_images(images):
        reshaped = np.zeros((images.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype='float32')
        for i in range(images.shape[0]):
            temp = np.resize(images[i], (IMG_SIZE * IMG_SIZE, 1))
            reshaped[i] = temp.reshape((IMG_SIZE, IMG_SIZE, 1))
        return reshaped / 255.0

    X1 = reshape_images(Images1)
    X2 = reshape_images(Images2)

    # Combine the two inputs (you may also subtract, concatenate, etc.)
    X = np.abs(X1 - X2)  # Change map-style input

    Y = train_labels.astype('float32')
    Y = Y.reshape((-1, 1)) if len(Y.shape) == 1 else Y

    # Build & train model
    model = build_sca_mca_addunetpp(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=1, hidden=sol[0])
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Fit
    model.fit(X1, X2, batch_size=2, epochs=25, validation_split=0.1, steps_per_epoch=stepEpochs)

    # Predict and evaluate
    preds = model.predict(X)
    pred_labels = (preds > 0.5).astype('int32').flatten()
    true_labels = Y.flatten().astype('int32')

    Eval = net_evaluation(true_labels, pred_labels)
    return Eval
