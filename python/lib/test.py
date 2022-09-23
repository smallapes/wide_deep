import tensorflow as tf
keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
10000)
columns = [keywords]
features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
'LSTM', 'Keras', 'RNN']])}
linear_prediction, _, _ = tf.compat.v1.feature_column.linear_model(features,
columns)

# or
import tensorflow as tf
keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
10000)
print("keywords", keywords)

keywords_embedded = tf.feature_column.embedding_column(keywords, 16)

columns = [keywords_embedded]
features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
'LSTM', 'Keras', 'RNN']])}
input_layer = tf.keras.layers.DenseFeatures(columns)
dense_tensor = input_layer(features)

