MESSAGE_QUEUE =  'video_duration_completeness'
CHECKPOINT_DIR = '/Users/yuxuebo/workspace/PythonHome/buzzbreak/data/serving_models/video_duration_completeness/online/'
TRAIN_DATA_DIR = '/Users/yuxuebo/workspace/PythonHome/buzzbreak/data/train_data/video_duration/'
TRAIN_DATA_COLUMNS = ['user', 'item', 'duration', 'length', 'gender', 'categories']

import tensorflow as tf
import numpy as np

n_user = int(5e7)
n_item = int(2e6)
n_category = 21
k_category = 4
k_gender = 2
k_user = 16

def get_labels_weights(features, params):
    completeness_unit = 0.5
    duration_unit = 10.0
    duration_coef = 0.5
    
    completeness = tf.cast(features['duration'], tf.float32)/(tf.cast(features['length'], tf.float32)+0.001)
    completeness = tf.clip_by_value(completeness, 0, 2)
    
    duration = tf.clip_by_value(features['duration'], 0, 100)
    duration = tf.cast(duration, tf.float32)
    
    completeness_weights = (completeness-completeness_unit)/completeness_unit
    duration_weights = (duration-duration_unit)/duration_unit
    weights = (1-duration_coef)*completeness_weights + duration_coef*duration_weights
    
    labels = (weights>0)
    weights = tf.abs(weights)
    return labels, weights

def model_fn(features, labels, mode, params):
    user = tf.clip_by_value(features['user'], 0, n_user-1)
    user_to_weights = tf.Variable(tf.zeros([n_user,]), name='user_to_weights')
    user_weights = tf.gather(user_to_weights, user)
    
    item = tf.clip_by_value(features['item'], 0, n_item-1)
    item_to_weights = tf.Variable(tf.zeros([n_item,]), name='item_to_weights')
    item_weights = tf.gather(item_to_weights, item)
    
    # category_x_item
    categories = features['categories']
    category_probes = tf.constant([2**i for i in range(n_category)], dtype=tf.int64)
    category_onehot = tf.bitwise.bitwise_and(tf.expand_dims(categories, axis=1), category_probes)
    category_onehot = tf.cast(category_onehot>0, tf.float32)
    item_category_vec = tf.Variable(tf.zeros([n_item, k_category]), name='item_category_vec')
    category_item_vec = tf.Variable(tf.random.truncated_normal([n_category, k_category], stddev=np.sqrt(1/k_category)),
                                    name='category_item_vec')
    item_category_vec_batch = tf.gather(item_category_vec, item)
    category_item_vec_batch = tf.matmul(category_onehot, category_item_vec)
    category_x_item = tf.reduce_sum(item_category_vec_batch*category_item_vec_batch, axis=1)

    # gender_x_item
    # gender 0 for null, 1 for male, 2 for female
    gender_mask = tf.cast(features['gender']>0, tf.float32)
    gender = tf.clip_by_value(features['gender'], 1, 2) - 1
    item_gender_vec = tf.Variable(tf.zeros([n_item, k_gender]), name='item_gender_vec')
    gender_item_vec = tf.Variable(tf.random.truncated_normal([2, k_gender], stddev=np.sqrt(1/k_gender)),
                                  name='gender_item_vec')
    
    item_gender_vec_batch = tf.gather(item_gender_vec, item)
    gender_item_vec_batch = tf.gather(gender_item_vec, gender)
    gender_x_item = tf.reduce_sum(item_gender_vec_batch*gender_item_vec_batch, axis=1)
    gender_x_item *= gender_mask
    
    # user_x_item
    item_user_vec = tf.Variable(
        tf.random.truncated_normal([n_item, k_user], stddev=np.sqrt(1/k_user)),
        name='item_user_vec')
    user_item_vec = tf.Variable(tf.zeros([n_user, k_user]), name='user_item_vec')
    user_item_vec_batch = tf.gather(user_item_vec, user)
    item_user_vec_batch = tf.gather(item_user_vec, item)
    user_x_item = tf.reduce_sum(user_item_vec_batch*item_user_vec_batch, axis=1)
    
    bias = tf.Variable(0.0, name='bias')
    logits = bias + user_weights + item_weights + category_x_item + gender_x_item + user_x_item
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.sigmoid(logits))
    
    labels, weights = get_labels_weights(features, params)
    
    pred = tf.sigmoid(logits)
    auc = tf.metrics.auc(labels=labels, predictions=pred, name='auc')
    weighted_auc = tf.metrics.auc(labels=labels, predictions=pred, name='weighted_auc', weights=weights)
    
    labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_sum(weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'auc': auc, 'weighted_auc': weighted_auc})
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope('train_metrics'):
        tf.summary.scalar('auc', auc[1])
        tf.summary.scalar('weighted_auc', weighted_auc[1])
        
    with tf.name_scope('train_stats'):
        tf.summary.scalar('bias', bias)
        tf.summary.histogram('user_weights', user_weights)
        tf.summary.histogram('item_weights', item_weights)
        tf.summary.histogram('category_x_item', category_x_item)
        tf.summary.histogram('gender_x_item', gender_x_item)
        tf.summary.histogram('user_x_item', user_x_item)
        
    train_op = tf.train.AdagradOptimizer(
        learning_rate=0.5, 
        initial_accumulator_value=1, 
    ).minimize(
        loss, 
        global_step=tf.train.get_global_step()
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
