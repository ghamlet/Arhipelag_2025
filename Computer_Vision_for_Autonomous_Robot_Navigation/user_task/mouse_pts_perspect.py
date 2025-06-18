"""Программа для выделения перспективы с изображения по точкам отмеченным с помощью мышки точкам.
При расставлении точек важно соблюдать порядок их выделения: первая точка ставится в любом месте,
вторая на смежной стороне, очень важно чтобы третья точка была по диагонали от второй,
четвертая в оставшееся место"""


import cv2  
import numpy as np
import pandas as pd
import math


path_to_images = "Computer_Vision_for_Autonomous_Robot_Navigation/user_task/images/0iHW4V-mK9-ijO-aRbz5g.jpg"
img = cv2.imread(path_to_images) 
     

NEW_SIZE_X = 800
NEW_SIZE_Y = 600
pts2 = np.float32([[0,0],
                   [NEW_SIZE_X,0],
                   [0,NEW_SIZE_Y],
                   [NEW_SIZE_X,NEW_SIZE_Y]
                   ])     # координаты прошлых точек на новом изображении после трансформации





pointsList = []
def mousePoints(event, x, y, flags, params):
    # функция для получения координат курсора мыши
    if event == cv2.EVENT_LBUTTONDOWN: 
        if len(pointsList) < 4: 
            cv2.circle(img, (x,y), 1,(0,0,255), cv2.FILLED)
            pointsList.append([x,y])

    

def trackbar(minblue=0, mingreen=0, minred=0, maxblue=255, maxgreen=255, maxred=255):
    # создание трекеров для каждого цвета

    cv2.namedWindow( "Trackbar")
    cv2.createTrackbar('minb', 'Trackbar', minblue, 255, lambda x: None)
    cv2.createTrackbar('ming', 'Trackbar', mingreen, 255, lambda x: None)
    cv2.createTrackbar('minr', 'Trackbar', minred, 255, lambda x: None)
    cv2.createTrackbar('maxb', 'Trackbar', maxblue, 255, lambda x: None)
    cv2.createTrackbar('maxg', 'Trackbar', maxgreen, 255, lambda x: None)
    cv2.createTrackbar('maxr', 'Trackbar', maxred, 255, lambda x: None)



points_of_quadr = False
main_exit = False
press_y = True



while True:

    #программа будет висеть во вложенном цикле пока на фото не будет отмечено четыре точки
    while not points_of_quadr:
        cv2.imshow("input", img)   #важно чтобы эти два окна имели одинаковое название
        cv2.setMouseCallback("input", mousePoints)
        
        k = cv2.waitKey(1)
         # при нажатии на клавишу q программа завершится
        if k == ord('q'):
            exit()

        elif k == ord('x'):
            cv2.destroyAllWindows()
            pointsList = pointsList[:-1]
            img = cv2.imread(path_to_images) 

            for pt in pointsList:
                cv2.circle(img, pt, 5,(0,0,255), cv2.FILLED)

        
        if len(pointsList) % 4 == 0 and len(pointsList) != 0:
            points_of_quadr = True
            press_y = False



    while not press_y:
        cv2.imshow("input", img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            exit()
       
        elif k == ord('y'):
            press_y = True

        elif k == ord('x'):
            cv2.destroyAllWindows()
            pointsList = pointsList[:-1]
            img = cv2.imread(path_to_images) 

            for pt in pointsList:
                cv2.circle(img, pt, 1,(0,0,255), cv2.FILLED)



            # если в массиве появилось 4 точки то можно находить перспективу
    print(pointsList)

    pts1 = np.float32(pointsList)

    M = cv2.getPerspectiveTransform(pts1,pts2) #получение перспективы
    dst = cv2.warpPerspective(img,M,(NEW_SIZE_X, NEW_SIZE_Y))
    
    cv2.imshow("transform", dst)

    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    trackbar() # создаем окно в трекбарами

    

    #в бесконечном цикле считываем положения ползунков для получения значений цветов
    while True:
        
        minb = cv2.getTrackbarPos('minb', 'Trackbar')
        ming = cv2.getTrackbarPos('ming', 'Trackbar')
        minr = cv2.getTrackbarPos('minr', 'Trackbar')
        maxb = cv2.getTrackbarPos('maxb', 'Trackbar')
        maxg = cv2.getTrackbarPos('maxg', 'Trackbar')
        maxr = cv2.getTrackbarPos('maxr', 'Trackbar')

        # применяем пороги цветов
        mask = cv2.inRange(hsv,(minb,ming,minr),(maxb,maxg,maxr))

        #пиксели подходящего цвета останутся своего цвета а все остальные станут черными
        result = cv2.bitwise_and(dst, dst, mask=mask)


        cv2.imshow('result', result)

        k = cv2.waitKey(10)
        # при нажатии на клавишу q программа завершится
        if k == ord('q'):
            exit()

        elif k == ord('s'):  # при нажатии клавиши s вам предложат ввести описание для значения порогового цвета и само значение сохранится в текстовый файл
            with open("trackbars_save.txt", "a") as f:
                title = input("\nEnter the description \nTo cancel, write no: ")
                
                if title not in ("n", "N", "no", "No", "NO"):
                    f.write(f"{title}:  {minb, ming, minr}, {maxb, maxg, maxr}" +"\n")
                    print("save\n")
        

        elif k == ord('x'):
            cv2.destroyAllWindows()
            pointsList = pointsList[:-1]
            img = cv2.imread(path_to_images) 

            for pt in pointsList:
                cv2.circle(img, pt, 5,(0,0,255), cv2.FILLED)

            points_of_quadr = False
            press_y = False
            break
           
