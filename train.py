import numpy as np
from keras.utils import load_img
import keras.utils as ku
import keras.callbacks as kc
import os, net_work, arg_params, time
from PIL import Image
import matplotlib.pyplot as plt


class OxfordPets(ku.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            # plt.imshow(x[j])
            # plt.show()
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = np.array(img)
            img[img > 0] = 1
            img[img == 0] = 0
            img = Image.fromarray(img)
            y[j] = np.expand_dims(img, 2)
            # plt.imshow(y[j])
            # plt.show()
            # # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y

t_start = time.perf_counter()
print("started in :", t_start, 's')

net_param = arg_params.net.parse_args()
train_params = arg_params.train.parse_args()

train_input_pth = [os.path.join(train_params.Trainx_pth, i) for i in os.listdir(train_params.Trainx_pth)]
train_target_pth = [os.path.join(train_params.Trainy_pth, i) for i in os.listdir(train_params.Trainy_pth)]

np.random.seed(2022)
index = np.random.permutation(len(train_input_pth))

train_input_pth = list(np.array(train_input_pth)[index])
train_target_pth = list(np.array(train_target_pth)[index])
val_num = train_params.Val_rate*len(train_input_pth)

train_input_img_paths = train_input_pth[:-int(val_num)]
train_target_img_paths = train_target_pth[:-int(val_num)]
val_input_img_paths = train_input_pth[-int(val_num):]
val_target_img_paths = train_target_pth[-int(val_num):]

#print(len(train_input_img_paths), len(train_target_img_paths), len(val_input_img_paths), len(val_target_img_paths))
train_gen = OxfordPets(train_params.Batch_size, (net_param.HEIGTH, net_param.WIDTH), train_input_img_paths, train_target_img_paths)
val_gen = OxfordPets(train_params.Batch_size, (net_param.HEIGTH, net_param.WIDTH), val_input_img_paths, val_target_img_paths)
if net_param.If_add == 1:
    mymodel = net_work.net_add_add()
elif net_param.If_add == 0:
    mymodel = net_work.net()
model = mymodel.model

callbacks = [
    kc.ModelCheckpoint(train_params.Best_model_pth, save_best_only=True)
]

history = model.fit(train_gen, epochs=train_params.Epochs, validation_data=val_gen, callbacks=callbacks)
model.save(train_params.Model_pth)

print("used time: ", time.perf_counter()-t_start, 's')

plt.plot([i for i in range(len(history.history['loss']))], history.history['loss'], linewidth=1, color="orange", label="loss")
plt.plot([i for i in range(len(history.history['accuracy']))], history.history['accuracy'], linewidth=1, color="green", label="acc")
plt.plot([i for i in range(len(history.history['val_loss']))], history.history['val_loss'], linewidth=1, color="black", label="val_loss")
plt.plot([i for i in range(len(history.history['val_accuracy']))], history.history['val_accuracy'], linewidth=1, color="blue", label="val_acc")
plt.legend()
plt.show()


