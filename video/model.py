from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import pandas as pd

from lib.wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import json


def predict(file_name):
    classifier = load_model("model_files/emotion_little_vgg_2.h5")
    pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"

    modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
    emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.8, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Define our model parameters
    depth = 16
    k = 8
    weight_file = None
    margin = 0.4
    image_dir = None

    # Get our weight file
    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="model_files",
                               file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)
    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    detector = dlib.get_frontal_face_detector()

    # Initialize Webcam
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    k = 0
    frames_dict = {}
    toJSON = {}
    prev_time = 0

    # dataframe to store data frame, start_time, end_time, age, gen, emotion
    df = pd.DataFrame(columns=["frame", "start_time", "end_time", "age", "gen", "emotion"])

    # # video length in seconds
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("Video length in seconds: {0}".format(video_length))

    # video length in minutes
    video_length_min = video_length / 60
    print("Video length in minutes: {0}".format(video_length_min))

    # # number of frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # number of frames per minute
    frames_per_min = fps * 60
    print("Frames per minute: {0}".format(frames_per_min))

    while True:
        ret, frame = cap.read()
        # ret, buffer = cv2.imencode('.jpg', frame)
        # get frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print("Current frame is: {0}".format(currentFrame))

        # # current time in milliseconds
        current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        # print("Current time in milliseconds: {0}".format(current_time_msec))
        # # current time in seconds
        current_time_sec = current_time_msec / 1000
        # print("Current time in seconds: {0}".format(current_time_sec))
        # current time in minutes
        # current_time_min = current_time_sec / 60
        current_time_min = current_time_sec
        print("*"*50)
        print("Current time in minutes: {0}".format(current_time_min))
        print("Previous time in minutes: {0}".format(prev_time))
        print("Current frame is: {0}".format(currentFrame))
        print("*" * 50)

        # concat start_time, end_time and frame
        # df = pd.concat([df, pd.DataFrame([[currentFrame, prev_time, current_time_min, 0, 0, 0]],
        #                                  columns=["frame", "start_time", "end_time", "age", "gen", "emotion"])])

        preprocessed_faces_emo = []

        if np.shape(frame) == ():
            print("End Of Array")
            break
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        detected = detector(frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        preprocessed_faces_emo = []
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
                face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA)
                face_gray_emo = face_gray_emo.astype("float") / 255.0
                face_gray_emo = img_to_array(face_gray_emo)
                face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                preprocessed_faces_emo.append(face_gray_emo)

            # make a prediction for Age and Gender
            results = model.predict(np.array(faces))
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            values = []
            # make a prediction for Emotion
            emo_labels = []
            for i, d in enumerate(detected):
                preds = classifier.predict(preprocessed_faces_emo[i])[0]
                emo_labels.append(emotion_classes[preds.argmax()])

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}, {}".format(int(predicted_ages[i]),
                                            "F" if predicted_genders[i][0] > 0.4 else "M", emo_labels[i])
                draw_label(frame, (d.left(), d.top()), label)
                values.append(label.split(', '))

            print(k)
            print(values)
            print("-"*150)
            # for each list in values(list) append to df
            for i in range(len(values)):
                df = pd.concat([df, pd.DataFrame([[k, prev_time, current_time_min, values[i][0], values[i][1], values[i][2]]], columns=["frame", "start_time", "end_time", "age", "gen", "emotion"])], ignore_index=True)

            # # concat to df
            # df = pd.concat([df, pd.DataFrame([[k, prev_time, current_time_min, values[0][0], values[0][1], values[0][2]]], columns=["frame", "start_time", "end_time", "age", "gen", "emotion"])], ignore_index=True)



            frames_dict[k] = values
            key = f"Frame:{k}"
            toJSON[key] = values
            out.write(frame)

        k += 1
        prev_time = current_time_min
        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    with open('filtered_frames.json', 'w') as fp:
        json_str = json.dumps(toJSON, indent=4) + '\n'
        fp.write(json_str)

    # with open('data.json', 'w', encoding='utf-8') as f:
    #     json.dump(toJSON, f, ensure_ascii=False, indent=4)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("End Of Program")

    print(df.head(10))

    # save to csv
    df.to_csv("filtered_frames.csv", index=False)

    return frames_dict
