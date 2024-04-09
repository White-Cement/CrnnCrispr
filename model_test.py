import tensorflow as tf
import scipy as sp
import numpy as np

model = tf.keras.models.load_model('model.h5') 

y_test_pred = model.predict([x1_test, x2_test])
y_test_pred = np.array(y_test_pred, 'float64')
y_test_pred = np.squeeze(y_test_pred)

spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]
personr = sp.stats.pearsonr(y_test, y_test_pred)[0]

print('spearmanrs:\n', spearmanr)
print('personrs:\n', personr)
