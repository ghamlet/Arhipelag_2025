# -*- coding: utf-8 -*-
import eval
import cv2
import pandas as pd

"""Файл служит для определения точности вашего алгоритма.
   Не редактируёте его!!!
   Для получения оценки точности, запустите файл на исполнение.
"""

def extract_object_list(row):
    object_list = []
    for i in range(4):
        color = row[2+i*2]  # 2 4 6 8
        # print(color)
        angle = row[3+i*2]  # 3 5 7 9
        # print(angle)
        if color != "empty":
            angle = int(angle)
            object_list.append([color, angle])
        # print()

    return object_list


def inspect(user_obj_list, true_obj_list):
    if len(user_obj_list) ==  len(true_obj_list):
        for i,point in enumerate(true_obj_list):
            if point[0] == user_obj_list[i][0] and abs(point[1]-user_obj_list[i][1])<=10:
                # Цвет верный, угол +-10 градусов верный
                pass
            else:
                return False
        return True  # Если все проверки пройдены и цикл завершился

    else:
        return False


def main():
    MAIN_DIR = "Computer_Vision_for_Autonomous_Robot_Navigation/user_task/"

    user_data_list = eval.preliminary_operations()
    csv_file = MAIN_DIR + "annotations.csv"
    data = pd.read_csv(csv_file, sep=';')
    data = data.sample(frac=1)

    all_good_detection = 0

    for row in data.itertuples():
        image = cv2.imread(MAIN_DIR + row[1])
        
        user_obj_list = eval.predict_color_and_angle(image, user_data_list)
        true_obj_list = extract_object_list(row)
        print("True : ", true_obj_list)
        print("User : ", user_obj_list)
        print()
        if inspect(user_obj_list, true_obj_list):
            all_good_detection += 1

    total_object = len(data.index)
    print("Для " + str(all_good_detection) + " изображения(ий) из " + str(total_object) + " верно определены цвета и угловые смещения объектов.")
    score = all_good_detection  / total_object
    print("Точность алгоритма: " + str(score))


if __name__ == '__main__':
    main()
