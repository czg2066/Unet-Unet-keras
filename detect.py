import tensorflow as tf
import net_work, arg_params
import numpy as np
import cv2

net_param = arg_params.net.parse_args()
train_params = arg_params.train.parse_args()
test_params = arg_params.test.parse_args()


if net_param.If_add == 1:
    mymodel = net_work.net_add_add()
elif net_param.If_add == 0:
    mymodel = net_work.net()
model = mymodel.model
model.load_weights(test_params.Model_pth)
Is_video = False
Is_image = False
try:
    img = cv2.imread(test_params.Device)
    cv2.resize(img, img.shape[:2])
    print("It is Image!")
    Is_image = True
except:
    camera = cv2.VideoCapture(test_params.Device)
    print("It is Video!")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if test_params.Save_size == -1:
        print("Origal size!")
        ret, img = camera.read()
        new_video_size = (img.shape[1], img.shape[0])
    else:
        new_video_size = test_params.Save_size
    if test_params.Save_pth_name == -1:
        new_video_pth = test_params.Device
    else:
        new_video_pth = test_params.Save_pth_name
    new_video = cv2.VideoWriter(new_video_pth, fourcc, test_params.Save_video_fps, new_video_size)
    Is_video = True

res_img = None
if test_params.If_save:
    print("begin to write video!")
while True:
    if Is_video:
        ret, img = camera.read()
        if test_params.If_save:
            new_video.write(res_img)
        if img is None:
            new_video.release()
            print("Video END!")
            break
    img2 = cv2.resize(img, (net_param.HEIGTH, net_param.WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img2)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    val_preds = model.predict(img_array)
    predict_img = val_preds[0]
    predict_img[predict_img > test_params.Sperete_rate] = 255
    predict_img[predict_img <= test_params.Sperete_rate] = 0
    predict_img = np.array(predict_img, dtype='uint8')
    mid_img = np.zeros(img2.shape, dtype='uint8')
    mid_img[:, :, test_params.Color] = predict_img[:, :, 0]
    res_img = cv2.addWeighted(mid_img, test_params.Layers_rate, img2, 1-test_params.Layers_rate, 0)
    res_img = cv2.resize(res_img, (img.shape[1], img.shape[0]))
    cv2.imshow("res_img", res_img)
    key = cv2.waitKey(100)
    if key == ord('q'):
        new_video.release()
        print("user cancel!")
        break
    if Is_image:
        print("Image END!")
        if test_params.If_save == 0:
            cv2.waitKey(0)
            break
        if test_params.Save_pth == -1:
            new_image_pth = test_params.Device
        else:
            new_image_pth = test_params.Save_pth
        cv2.imwrite(new_image_pth, res_img)
        cv2.waitKey(0)
        break

