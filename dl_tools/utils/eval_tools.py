import tensorflow as tf


def confusion_matrix(config, model, validation_gen, do_print=True):
    confusion_gen = validation_gen.image_data_generator.flow_from_directory(config.data_folder_test, shuffle=False,
        batch_size=config.batch_size, class_mode=validation_gen.class_mode, target_size=config.input_shape[:2])
    val_pred = model.predict_generator(confusion_gen, confusion_gen.samples // config.batch_size + 1)
    if validation_gen.class_mode == 'binary':
        val_pred = val_pred > .5
    else:
        val_pred = tf.math.argmax(val_pred, axis=1)
    confusion_matrix = tf.math.confusion_matrix(confusion_gen.classes, val_pred)
    if do_print:
        print(f'Confusion matrix:')
        for label, cm_row in zip(confusion_gen.class_indices.keys(), confusion_matrix):
            print(f'{label}: {cm_row}')
    return confusion_matrix
