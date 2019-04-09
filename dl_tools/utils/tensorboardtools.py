import tensorflow as tf
from tensorboard import default
from tensorboard import program
import logging
import sys
import numpy as np
from keras.callbacks import Callback


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


def make_image(tensor, uv_annotation=None, uv_prediction=None, u_annotation=None, u_prediction=None):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image, ImageDraw
    import io
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)

    for u, c in zip([u_annotation, u_prediction], [(255, 0, 0), (0, 255, 0)]):
        if u is not None:
            draw = ImageDraw.Draw(image)
            draw.line([(u, 0), (u, height)], fill=c)
    for uv, c in zip([uv_annotation, uv_prediction], [(255, 0, 0), (0, 255, 0)]):
        if uv is not None:
            u, v = uv
            draw = ImageDraw.Draw(image)
            draw.line([(u, max(0, v - 3)), (u, v + 3)], fill=c)
            draw.line([(max(0, u - 3), v), (u + 3, v)], fill=c)

    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)


class TensorBoardImage(Callback):  #TODO Merge with tensorboard callback
    def __init__(self, model, log_dir, validation_data, tag='Prediction', n=10):
        super().__init__()
        self.model = model
        self.log_dir = log_dir + 'img/'
        self.tag = tag
        self._random_index = sorted(np.random.choice(validation_data[0].shape[0], n, replace=False))
        self._input = validation_data[0][self._random_index]
        self._images = np.uint8(self._input[:, :, :, :3] * 255)
        self._annotations = validation_data[1][self._random_index]

    def on_epoch_end(self, epoch, logs={}):
        if 1 and epoch % 100 != 0:  # TODO convert to wallclock frequency?
            return
        predictions = self.model.predict(self._input)

        summaries = []
        for img, annotation, prediction, index in zip(self._images, self._annotations, predictions, self._random_index):
            if self._annotations.ndim == 1:
                image = make_image(img, u_annotation=annotation, u_prediction=prediction)
            else:
                image = make_image(img, uv_annotation=annotation, uv_prediction=prediction)
            summaries.append(tf.Summary.Value(tag='Predictions/{}'.format(index), image=image))
        with tf.summary.FileWriter(self.log_dir) as writer:
            writer.add_summary(tf.Summary(value=summaries), epoch)

# The code below almost worked, but in the end I could stop the images from beings in multiple tabs name predictions_X
#
# from keras.callbacks import TensorBoard
#
# class TensorBoardWithImages(TensorBoard):
#     def __init__(self, validation_data, n=10, **kwargs):
#         super(TensorBoardWithImages, self).__init__(**kwargs)
#         self._random_index = sorted(np.random.choice(validation_data[0].shape[0], n, replace=False))
#         self._input = validation_data[0][self._random_index]
#         self._images = np.uint8(self._input[:, :, :, :3] * 255)
#         # self._u_annotations = validation_data[1][self._random_index]
#
#     def on_epoch_end(self, epoch, logs=None):
#         super(TensorBoardWithImages, self).on_epoch_end(epoch, logs)
#
#         # with tf.Graph().as_default():
#         summary_op = tf.summary.image(name="predictions", tensor=self._images, max_outputs=3, family='test')
#         # summary_op = tf.summary.merge_all()  #it this needed?
#         self.writer.add_summary(summary_op.eval(session=self.sess), epoch)
#
#     def on_batch_end(self, batch, logs=None):
#         super(TensorBoardWithImages, self).on_batch_end(batch, logs)