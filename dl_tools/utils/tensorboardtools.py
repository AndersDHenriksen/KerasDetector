import tensorflow as tf
from tensorboard import default
from tensorboard import program
import logging
import sys


def tensorboard_launch(experiments_folder):
    # Remove http messages and tensorboard warnings
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    # Start tensorboard server
    if tf.VERSION == '1.10.0':
        tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
        tb.configure(argv=['--logdir', experiments_folder])
    else:  # 1.12
        tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', experiments_folder])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)
    # From: https://stackoverflow.com/questions/42158694/how-to-run-tensorboard-from-python-scipt-in-virtualenv/