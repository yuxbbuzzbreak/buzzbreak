import time
import tensorflow as tf
import numpy as np
import pyarrow as pa
import pika
import model_config
import logging
LOGGER = logging.getLogger('train_forever')

from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
import os

should_save_checkpoint = False

class MyCheckpointSaverHook(session_run_hook.SessionRunHook):
    def __init__(self, 
                 checkpoint_dir,
                 checkpoint_basename="model.ckpt"):
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        
    def _save(self, session):
        last_step = session.run(training_util._get_or_create_global_step_read())
        saver = ops.get_collection(ops.GraphKeys.SAVERS)[0]
        LOGGER.info('saving checkpoint to {} with step {}'.format(self._save_path, last_step))
        saver.save(session, self._save_path, global_step=last_step)
        
    def end(self, session):
        self._save(session)
        
    def after_run(self, run_context, run_values):
        global should_save_checkpoint
        if should_save_checkpoint:
            self._save(run_context.session)
            should_save_checkpoint = False

def make_generator(queue):
    global should_save_checkpoint
    LOGGER.info('consuming messages from queue: {}'.format(queue))
    parameters = pika.ConnectionParameters(
        host='localhost', heartbeat=0, socket_timeout=3600*1000, stack_timeout=3600*1000)
    
    prev_batch = None
    while True:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue)
        try:
            for method_frame, properties, body in channel.consume(queue, auto_ack=True, exclusive=True, inactivity_timeout=20):
                if body == b'end':
                    return

                if body is None:
                    LOGGER.info('consumer timeout')
                    if prev_batch:
                        should_save_checkpoint = True
                        tmp, prev_batch = prev_batch, None
                        yield tmp
                else:
                    tmp, prev_batch = prev_batch, pa.deserialize(body)
                    if tmp:
                        yield tmp
        except Exception as e:
            LOGGER.info(str(e))
            time.sleep(5)
            continue
            
columns = model_config.TRAIN_DATA_COLUMNS

def make_dataset(queue):
    ds = tf.data.Dataset.from_generator(
        generator=lambda: make_generator(queue),
        output_types={col: tf.int64 for col in columns}
    )
    ds = ds.prefetch(10)
    return ds


config = tf.estimator.RunConfig(save_summary_steps=100, 
                                keep_checkpoint_max=3,
                                log_step_count_steps=10000,
                                save_checkpoints_secs=None)

my_model = tf.estimator.Estimator(
    model_fn=model_config.model_fn,
    model_dir=model_config.CHECKPOINT_DIR,
    config=config,
    params=None
)

my_hook = MyCheckpointSaverHook(checkpoint_dir=my_model.model_dir)

queue = model_config.MESSAGE_QUEUE
my_model.train(input_fn=lambda: make_dataset(queue), hooks=[my_hook])
