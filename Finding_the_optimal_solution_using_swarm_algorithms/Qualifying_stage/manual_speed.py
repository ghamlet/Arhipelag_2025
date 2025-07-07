from pioneer_sdk import Pioneer
import time


if __name__ == "__main__":
    print("start")
    pioneer_mini = Pioneer(name="pioneer", ip="127.0.0.1", mavlink_port=8000, connection_method="udpout", 
                           device="dev/serial0", baud=115200, logger=True, log_connection=True)
    

    time.sleep(3)


    pioneer_mini.arm()
    pioneer_mini.takeoff()
    time.sleep(3)




    pioneer_mini.go_to_local_point(x=0, y=0, z=2, yaw=0)
    while not pioneer_mini.point_reached():
        pass

    target = (4, -4)
    t = time.time()



    while True:
        # по диагонали
        pioneer_mini.set_manual_speed_body_fixed(vx= 1, vy=-1, vz=0, yaw_rate=0)  # значения не играют роли. важен знак чисел
        time.sleep(0.05)

        cur_pose = pioneer_mini.get_local_position_lps(get_last_received=True)
        drone_x, drone_y = cur_pose[:2]
        if target[0] - 0.1 <= drone_x <= target[0] + 0.1:
            print("point-----------")
            for i in range(10):
                pioneer_mini.set_manual_speed_body_fixed(vx= 0, vy=0, vz=0, yaw_rate=0)
            break

        
        # через 10 секунд остановится
        if time.time() - t > 10:
            break

   

    pioneer_mini.land()

    pioneer_mini.close_connection()
    del pioneer_mini
