import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from Evaluation import net_evaluation


def create_cnn_model(input_shape, output_classes, params):
    model = Sequential()
    model.add(Conv2D(params['filters1'], (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(params['filters2'], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(output_classes, activation=params['final_activation']))
    model.compile(optimizer=Adam(learning_rate=params['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_data(image1, image2, label_mask, patch_size=5):
    h, w = image1.shape
    X, y = [], []
    pad = patch_size // 2
    image1 = np.pad(image1, ((pad, pad), (pad, pad)), mode='reflect')
    image2 = np.pad(image2, ((pad, pad), (pad, pad)), mode='reflect')
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            patch1 = image1[i - pad:i + pad + 1, j - pad:j + pad + 1]
            patch2 = image2[i - pad:i + pad + 1, j - pad:j + pad + 1]
            patch = np.stack([patch1, patch2], axis=-1)  # shape: (patch_size, patch_size, 2)
            X.append(patch)
            y.append(label_mask[i - pad, j - pad])
    return np.array(X), np.array(y)


def genetic_algorithm_search(param_space, X_data, y_data, input_shape, output_classes):
    best_score = 0
    best_params = None
    for i in range(3):  # You can increase this for deeper search
        params = {
            'filters1': np.random.choice(param_space['filters']),
            'filters2': np.random.choice(param_space['filters']),
            'dense_units': np.random.choice(param_space['dense_units']),
            'dropout': np.random.choice(param_space['dropout']),
            'lr': np.random.choice(param_space['lr']),
            'final_activation': 'softmax'
        }
        model = create_cnn_model(input_shape, output_classes, params)
        model.fit(X_data, y_data, epochs=3, batch_size=64, verbose=0)
        score = model.evaluate(X_data, y_data, verbose=0)[1]  # accuracy
        if score > best_score:
            best_score = score
            best_params = params
    return best_params


def Model_GACNN(image1, image2, label_mask, activation='softmax'):
    print('GA-CNN Change Detection - Full Dataset')

    image1 = image1.astype('float32') / 255.0
    image2 = image2.astype('float32') / 255.0
    X_data, y_data = prepare_data(image1, image2, label_mask, patch_size=5)
    y_data = np.eye(2)[y_data]  # One-hot encode labels (0/1 -> [1, 0] / [0, 1])

    input_shape = X_data.shape[1:]
    output_classes = 2

    # Hyperparameter search
    param_space = {
        'filters': [16, 32, 64],
        'dense_units': [64, 128],
        'dropout': [0.3, 0.5],
        'lr': [0.001, 0.0001]
    }

    best_params = genetic_algorithm_search(param_space, X_data, y_data, input_shape, output_classes)

    # Train final model on full dataset
    model = create_cnn_model(input_shape, output_classes, best_params)
    model.fit(X_data, y_data, epochs=10, batch_size=64, verbose=1)

    predictions = model.predict(X_data)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_data, axis=1)

    Eval = net_evaluation(true_labels, pred_labels)
    return Eval