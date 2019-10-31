from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dropout, Flatten, Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model


class MobileNet2:
    @staticmethod
    def build(config, classes):

        # Build transfer learning network
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=config.input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        # Build new top
        X = base_model.output
        X = MaxPooling2D((7, 7))(X)  # Alternative X = DepthwiseConv2D((7, 7), activation='relu')(X)
        X = Flatten()(X)
        X = Dropout(0.5)(X)
        X = Dense(64, activation='relu')(X)
        X = Dropout(0.5)(X)
        X = Dense(classes, activation='softmax' if classes > 1 else 'sigmoid')(X)
        model = Model(inputs=base_model.input, outputs=X)
        model.is_in_warmup = True

        return model