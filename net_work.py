import keras.layers as ky
import keras.optimizers as ko
import keras.losses as ks
from keras.backend import concatenate
from keras.models import Model
import arg_params


class net():
    def __init__(self,):
        self.net_param = arg_params.net.parse_args()
        #第零层下降
        self.inputs = ky.Input(shape=(self.net_param.HEIGTH, self.net_param.WIDTH)+(3,))
        self.x = ky.Conv2D(32, 3, padding='same')(self.inputs)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con0 = ky.Activation("relu")(self.x)
        self.x = ky.MaxPooling2D(2, padding="same")(self.con0)

        #第一层下降
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con1 = ky.Activation("relu")(self.x)
        self.x = ky.MaxPooling2D(2, padding="same")(self.con1)

        #第二层下降
        self.x = ky.Conv2D(128, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(128, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con2 = ky.Activation("relu")(self.x)
        self.x = ky.MaxPooling2D(2, padding="same")(self.con2)

        #第三层下降
        self.x = ky.Conv2D(256, 3, activation='relu', padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(256, 3, activation='relu', padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con3 = ky.Activation("relu")(self.x)
        self.x = ky.MaxPooling2D(2, padding="same")(self.con3)

        # 第四层下降
        self.x = ky.Conv2D(512, 3, activation='relu', padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(512, 3, activation='relu', padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con4 = ky.Activation("relu")(self.x)

        # 第零层上升
        self.conT0 = concatenate([ky.Conv2DTranspose(256, (2, 2), strides=(
            2, 2), padding='same')(self.con4), self.con3], axis=-1)
        self.x = ky.Conv2D(256, 3, padding='same')(self.conT0)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(256, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT0 = ky.Activation("relu")(self.x)

        #第一层上升
        self.conT1 = concatenate([ky.Conv2DTranspose(128, (2, 2), strides=(
            2, 2), padding='same')(self.conT0), self.con2], axis=-1)
        self.x = ky.Conv2D(128, 3, padding='same')(self.conT1)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(128, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT1 = ky.Activation("relu")(self.x)

        #第二层上升
        self.conT2 = concatenate([ky.Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(self.conT1), self.con1], axis=-1)
        self.x = ky.Conv2D(64, 3, padding='same')(self.conT2)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT2 = ky.Activation("relu")(self.x)

        # 第三层上升
        self.conT3 = concatenate([ky.Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(self.conT2), self.con0], axis=-1)
        self.x = ky.Conv2D(32, 3, padding='same')(self.conT3)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT3 = ky.Activation("relu")(self.x)

        self.outputs = ky.Conv2D(1, 1, activation='sigmoid')(self.conT3)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

        self.Adam = ko.Adam(learning_rate=self.net_param.Lr, decay=self.net_param.Lr_decay, clipnorm=1)

        self.model.compile(self.Adam,
                           loss='binary_crossentropy',
                           #decay=self.net_param.Lr_decay,
                           metrics=['accuracy'])

class net_add_add():
    def __init__(self,):
        self.net_param = arg_params.net.parse_args()
        #第零层下降
        self.inputs = ky.Input(shape=(self.net_param.HEIGTH, self.net_param.WIDTH)+(3,))
        self.x = ky.Conv2D(32, 3, padding='same')(self.inputs)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con0 = ky.Activation("relu")(self.x)
        self.x = ky.MaxPooling2D(2, padding="same")(self.con0)

        #第一层下降
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con1 = ky.Activation("relu")(self.x)
        self.x1 = ky.MaxPooling2D(2, padding="same")(self.con1)

        # 第一个输出
        self.conT10 = concatenate([ky.Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(self.con1), self.con0], axis=-1)
        self.x = ky.Conv2D(32, 3, padding='same')(self.conT10)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT10 = ky.Activation("relu")(self.x)

        #第二层下降
        self.x = ky.Conv2D(128, 3, padding='same')(self.x1)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(128, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con2 = ky.Activation("relu")(self.x)
        self.x1 = ky.MaxPooling2D(2, padding="same")(self.con2)

        # 第二个输出
        self.conT21 = concatenate([ky.Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(self.con2), self.con1], axis=-1)
        self.x = ky.Conv2D(64, 3, padding='same')(self.conT21)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT21 = ky.Activation("relu")(self.x)

        self.conT20 = concatenate([ky.Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(self.conT21), self.conT10], axis=-1)
        self.x = ky.Conv2D(32, 3, padding='same')(self.conT20)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT20 = ky.Activation("relu")(self.x)

        #第三层下降
        self.x = ky.Conv2D(256, 3, activation='relu', padding='same')(self.x1)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(256, 3, activation='relu', padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con3 = ky.Activation("relu")(self.x)
        self.x1 = ky.MaxPooling2D(2, padding="same")(self.con3)

        # 第三个输出
        self.conT32 = concatenate([ky.Conv2DTranspose(128, (2, 2), strides=(
            2, 2), padding='same')(self.con3), self.con2], axis=-1)
        self.x = ky.Conv2D(128, 3, padding='same')(self.conT32)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(128, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT32 = ky.Activation("relu")(self.x)

        self.conT31 = concatenate([ky.Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(self.conT32), self.conT21], axis=-1)
        self.x = ky.Conv2D(64, 3, padding='same')(self.conT31)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT31 = ky.Activation("relu")(self.x)

        self.conT30 = concatenate([ky.Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(self.conT31), self.conT20], axis=-1)
        self.x = ky.Conv2D(32, 3, padding='same')(self.conT30)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT30 = ky.Activation("relu")(self.x)

        # 第四层下降
        self.x = ky.Conv2D(512, 3, activation='relu', padding='same')(self.x1)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(512, 3, activation='relu', padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.con4 = ky.Activation("relu")(self.x)

        # 第零层上升
        self.conT0 = concatenate([ky.Conv2DTranspose(256, (2, 2), strides=(
            2, 2), padding='same')(self.con4), self.con3], axis=-1)
        self.x = ky.Conv2D(256, 3, padding='same')(self.conT0)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(256, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT0 = ky.Activation("relu")(self.x)

        #第一层上升
        self.conT1 = concatenate([ky.Conv2DTranspose(128, (2, 2), strides=(
            2, 2), padding='same')(self.conT0), self.conT32], axis=-1)
        self.x = ky.Conv2D(128, 3, padding='same')(self.conT1)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(128, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT1 = ky.Activation("relu")(self.x)

        #第二层上升
        self.conT2 = concatenate([ky.Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(self.conT1), self.conT31], axis=-1)
        self.x = ky.Conv2D(64, 3, padding='same')(self.conT2)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(64, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT2 = ky.Activation("relu")(self.x)

        # 第三层上升
        self.conT3 = concatenate([ky.Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(self.conT2), self.conT30], axis=-1)
        self.x = ky.Conv2D(32, 3, padding='same')(self.conT3)
        self.x = ky.BatchNormalization()(self.x)
        self.x = ky.Activation("relu")(self.x)
        self.x = ky.Conv2D(32, 3, padding='same')(self.x)
        self.x = ky.BatchNormalization()(self.x)
        self.conT3 = ky.Activation("relu")(self.x)

        self.outputs = ky.Conv2D(1, 1, activation='sigmoid')(self.conT3)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

        self.Adam = ko.Adam(learning_rate=self.net_param.Lr, decay=self.net_param.Lr_decay, clipnorm=1)

        self.model.compile(self.Adam,
                           loss='binary_crossentropy',
                           #decay=self.net_param.Lr_decay,
                           metrics=['accuracy'])