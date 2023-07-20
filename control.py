import serial
import sys
import termios
import tty
import threading

from camera.simoutaneous_sender import send_to_server

# 打开串口
ser = serial.Serial('/dev/ttyS0', 9600,timeout=1)  # 根据你的连接方式和串口号进行调整

# 获取键盘输入
def send_to_arduino(command, num):
    # 打开串口连接
    # 将字符和数字组合成一个字符串
    data_to_send = f"{command}{num}\n"

    try:
        # 将数据发送给Arduino
        ser.write(data_to_send.encode())

        # 关闭串口连接
        ser.close()

        print(f"已发送命令：{command}，数字：{num} 到Arduino。")
    except serial.SerialException as e:
        print(f"发送数据时出现错误：{e}")

# 读取键盘指令并发送给 Arduino
def test():
    send_to_arduino('w','20')

def main():
    arduino_thread = threading.Thread(target=send_to_arduino, args=('w', 5))
    server_thread = threading.Thread(target=send_to_server)

    # Start the threads
    arduino_thread.start()
    server_thread.start()

    # Wait for both threads to finish (you can add a timeout if needed)
    arduino_thread.join()
    server_thread.join()

if __name__ == "__main__":
    # pass
   send_to_arduino('d','20')
   



