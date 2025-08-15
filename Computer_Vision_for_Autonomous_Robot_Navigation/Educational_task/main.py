# -*- coding: utf-8 -*-
import eval
import cv2
import pandas as pd

"""Файл служит для определения точности вашего алгоритма.
   Не редактируёте его!!!
   Для получения оценки точности, запустите файл на исполнение.
"""


def extract_obect_list(row):
    x = int(row[3])
    y = int(row[4])
    x2 = int(row[5])
    y2 = int(row[6])
    true_obj = (x, y, x2, y2)
    return true_obj


def inspect(user_obj_list, true_obj_list):
    miss = 0
    hit = 0
    if IoU(user_obj_list, true_obj_list) > 0.6:
        hit += 1
    else:
        miss += 1

    return miss, hit


def IoU(user_box,true_box):
    """IoU = Area of overlap / Area of union
       Output: 0.0 .. 1.0
       Не важно в каком порядке передаются рамки, IoU не изменится.
    """

    xA = max(user_box[0], true_box[0])
    yA = max(user_box[1], true_box[1])
    xB = min(user_box[2], true_box[2])
    yB = min(user_box[3], true_box[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (user_box[2] - user_box[0] + 1) * (user_box[3] - user_box[1] + 1)
    boxBArea = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def main():
    csv_file = "/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/annotations.csv"
    
    data = pd.read_csv(csv_file, sep=';')
    data = data.sample(frac=1)

    all_miss_detection = 0
    all_good_detection = 0


    for row in data.itertuples():
        image = cv2.imread("/home/arrma/PROGRAMMS/Arhipelag_2025/Computer_Vision_for_Autonomous_Robot_Navigation/Educational_task/" + row[1])
        user_obj = eval.predict_box(image)

        if isinstance(user_obj, tuple) and len(user_obj) == 4 and all(isinstance(x, int) for x in user_obj):
            if user_obj == ():
                all_miss_detection += 1
                
            else:
                true_obj = extract_obect_list(row)
                miss_detection, good_detection = inspect(user_obj, true_obj)
                all_miss_detection += miss_detection
                all_good_detection += good_detection

        else:
            print("Недопустимый формат выходных данных.\n" +
                + "Ваша функция должна возвращать кортеж (tuple) с четырьмя целыми числами (int).\n")
            all_miss_detection += 1


        cv2.imshow("frame", image)
        cv2.waitKey(0)

    total_object = len(data.index)
    print("Из " + str(total_object) + " объектов, верно детектированы и распознаны " + str(all_good_detection))
    print("При этом, совершено " + str(all_miss_detection) + " ошибочных детектирований")
    if (all_good_detection - all_miss_detection) <= 0:
        score = 0
        print("Алгоритм чаще ошибался, чем выдавал верные ответы")
    else:
        score = (all_good_detection - all_miss_detection) / total_object
    print("Точность: " + str(score))


if __name__ == '__main__':
    main()
