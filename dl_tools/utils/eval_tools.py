import numpy as np


def confusion_matrix(config, model, validation_gen, do_print=True):
    confusion_gen = validation_gen.image_data_generator.flow_from_directory(config.data_folder_test, shuffle=False,
        batch_size=config.batch_size, class_mode=validation_gen.class_mode, target_size=config.input_shape[:2])
    val_pred = model.predict_generator(confusion_gen)
    if validation_gen.class_mode == 'binary':
        val_pred = val_pred > .5
    else:
        val_pred = np.argmax(val_pred, axis=1)
    confusion_matrix = compute_confusion_matrix(confusion_gen.classes, val_pred)
    if do_print:
        print(f'Confusion matrix:')
        for label, cm_row in zip(confusion_gen.class_indices.keys(), confusion_matrix):
            print(f'{label}: {cm_row}')
    return confusion_matrix


def compute_confusion_matrix(true, pred):
    cm = np.zeros((true.max() + 1, true.max() + 1), np.int)
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm


def show_errors(model, validation_gen):
    import matplotlib.pyplot as plt
    old_batch_size, validation_gen.batch_size = validation_gen.batch_size, 1
    key_name_dict = {v: k for k, v in validation_gen.class_indices.items()}
    for idx in range(validation_gen.n):
        x, y_label = validation_gen[idx]
        y_label = y_label[0].argmax()
        y_pred = model.predict(x)[0].argmax()
        if y_pred == y_label:
            continue
        image = ((x[0] + 1) * 127.5).astype(np.uint8)
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.imshow(image)
        plt.title(f"Prediction: {key_name_dict[y_pred]}. Correct {key_name_dict[y_label]}.")
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close(fig)
    validation_gen.batch_size = old_batch_size
