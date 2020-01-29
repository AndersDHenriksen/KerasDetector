from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten, Input


class StandardNet:
    @staticmethod
    def build(config, classes):

        # 1.  Conv => Relu = > Pool
        input = Input(shape=config.input_shape)
        X = Conv2D(48, (7, 7), activation='relu')(input)
        X = MaxPooling2D(pool_size=(2, 2))(X)

        # 2. Conv => Relu = > Pool
        X = Conv2D(96, (5, 5), activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)

        # 3. Conv => Relu = > Pool
        X = Conv2D(128, (3, 3), activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)

        # 4. Conv => Relu = > Pool
        X = Conv2D(256, (3, 3), activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)

        # 5. Conv => Relu = > Pool
        X = Conv2D(256, (3, 3), activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)

        # 6. Conv => Relu = > Pool
        X = Conv2D(256, (3, 3), activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)

        # 7. Flat => Dropout
        X = Flatten()(X)
        X = Dropout(0.3)(X)

        # 8. FC => Relu => Dropout
        X = Dense(128, activation='relu')(X)
        X = Dropout(0.5)(X)

        # 9. FC => Relu
        X = Dense(16, activation='relu')(X)

        # 10. FC => Softmax
        output = Dense(classes, activation='softmax')(X)

        Model(input, output)

        return Model