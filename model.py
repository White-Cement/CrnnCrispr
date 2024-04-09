import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, Dense, Flatten, RepeatVector, Dropout, Concatenate, LSTM, GRU, Conv1D
from tensorflow.keras.models import Model

def CrnnCrispr(x1_train, x2_train, y_train, n, model_type=model_type, batch_size=batch_size, epochs=epochs):
    x1_input = Input(shape=(23, 4,))
    conv1 = Conv1D(256, 3, padding='same', activation='relu')(x1_input)
    conv2 = Conv1D(256, 3, padding='same', activation='relu')(conv1)
    conv3 = Conv1D(256, 3, padding='same', activation='relu')(conv2)
    conv4 = Conv1D(256, 3, padding='same', activation='relu')(conv3)
    conv5 = Conv1D(256, 3, padding='same', activation='relu')(conv4)
    flatv1 = Flatten()(conv5)
  
    x2_input = Input(shape=(24,))
    embedd = Embedding(7, 44, input_length=24)(x2_input)
    gruv1 = Bidirectional(GRU(64, activation='tanh', return_sequences=True))(embedd)
    concatv1 = Concatenate()([embedd, gruv1])
    gruv2 = Bidirectional(GRU(128, activation='tanh', return_sequences=True))(concatv1)
    flatv2 = Flatten()(gruv2)

    concatv2 = Concatenate()([flatv1, flatv2])
    rev = RepeatVector(1)(concatv2)
    lstmv1 = LSTM(1024, activation='tanh', return_sequences=True)(rev)
    lstmv2 = LSTM(512, activation='tanh', return_sequences=True)(lstmv1)
    flatv3 = Flatten()(lstmv2)
    denv1 = Dense(256, activation='relu')(flatv3)
    denv2 = Dense(128, activation='relu')(denv1)
    denv3 = Dense(64, activation='relu')(denv2)
    dropv1 = Dropout(0.3)(denv3)
    output = Dense(1, activation='linear', name="output")(dropv1)
    model = Model(inputs=[x1_input, x2_input], outputs=[output])
    model.summary()

    adamax = tf.keras.optimizers.Adamax(learning_rate=0.0001)
    model.compile(loss='mean_absolute_error', optimizer=adamax, metrics=['mae'])
    model.fit([x1_train, x2_train], y_train, batch_size=batch_size,
              epochs=epochs, verbose=2, validation_split=0.1,
              shuffle=False)
    return model
