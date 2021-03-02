from __future__ import print_function
import time
import threading
from tkinter import ttk
import serial
import tkinter as tk
from tkinter import Tk, Menu, Label, Frame, Canvas, RIGHT, messagebox, Text, Entry, Button, ttk
import controller
from configurations import *
import numpy as np


from PIL import Image
from PIL import ImageTk
# import Image

import threading
import datetime
import imutils
import cv2
import os

import pyaudio
import speech_recognition as sr
from gtts import gTTS
import gtts
import statistics

class View():

    serial_data = ''
    filter_data = ''
    angle0=0.0
    start_time = time.time()
    robotcommand=0
    keeper_operation_mode=0

    def __init__(self, parent, controller):
        self.controller = controller
        self.parent = parent
        self.create_platform()


    def create_platform(self):
        self.create_top_menu()
        self.create_io()

    def create_top_menu(self):
        self.menu_bar=Menu(self.parent)
        self.create_file_menu()
        self.create_about_menu()

    def load_fab_code_menu_clicked(self):
        pass

    def convert_gcode_menu_clicked(self):
        pass

    def on_about_menu_clicked(self):
        messagebox.showinfo("Robot arm interface:",
                            "This is an robot arm\n interface with inverse kinematics")

    def create_file_menu(self):
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Load Fab-Code", command=self.load_fab_code_menu_clicked)
        self.file_menu.add_command(
            label="Convert G-Code to Fab-Code", command=self.convert_gcode_menu_clicked)
        self.file_menu.add_command(
            label="Exit", command=self.parent.destroy)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.parent.config(menu=self.menu_bar)

    def create_about_menu(self):
        self.about_menu = Menu(self.menu_bar, tearoff=0)
        self.about_menu.add_command(
            label="About", command=self.on_about_menu_clicked)
        self.menu_bar.add_cascade(label="About", menu=self.about_menu)
        self.parent.config(menu=self.menu_bar)


    def cam0_control(self):
        if (self.cam0_on_off["text"]=="On"):
            self.cam0_on_off["text"]="Off"
        else:
            self.cam0_on_off["text"] = "On"

        if (self.cam0_on_off["text"] == "Off"):
            # (self.combobox["value"])
            # cam_No=self.cam0_combobox.index()
            try:
                self.cap0 = cv2.VideoCapture(self.cam0_combobox.current())
                # self.cap0 = cv2.VideoCapture(self.cam0_combobox.current())
                self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.t1 = threading.Thread(target=self.cam0_show())
                self.t1.daemon = True
                self.t1.start()

                Brightness = self.cap0.get(cv2.CAP_PROP_BRIGHTNESS)
                Contrast = self.cap0.get(cv2.CAP_PROP_CONTRAST)
                Saturation = self.cap0.get(cv2.CAP_PROP_SATURATION)
                Exposure = self.cap0.get(cv2.CAP_PROP_EXPOSURE)
                Gain = self.cap0.get(cv2.CAP_PROP_GAIN)
                print(Brightness, Contrast, Saturation, Exposure, Gain)

            except:
                self.cam0_on_off["text"] = "On"
                messagebox.showerror("Error","Camera # is not available")
        else:
            self.cap0.release()

    def cam0_show(self):
        # dst0=self.controller.ball_detection(self.cap0, 'B2.npz')
        _, self.frame = self.cap0.read()
        self.center_coordinates0 = (120,50)
        self.radius0 = 20
        self.color0 = (0,0,255)
        self.thickness0 = 1
        self.frame_circle = cv2.circle(self.frame,self.center_coordinates0,self.radius0,self.color0,self.thickness0)
        self.cv2image = cv2.cvtColor(self.frame_circle, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(self.cv2image)
        # img2 = cv2.cvtColor(self.frame_circle,cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img2)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lcam0.imgtk = imgtk
        self.lcam0.configure(image=imgtk)
        self.lcam0.after(10, self.cam0_show)

    def cam0_luminance_calculation(self):
        if (self.cam0_on_off["text"] == "Off"):
            try:
                img2 = cv2.cvtColor(self.frame_circle, cv2.COLOR_BGR2RGB)
                self.luminance_sum = 0.0
                for x in range((self.center_coordinates0[1] - self.radius0 - 2),
                               (self.center_coordinates0[1] + self.radius0 + 2)):
                    first_red = 0
                    in_circle = 0
                    for y in range((self.center_coordinates0[0] - self.radius0 - 2),
                                   (self.center_coordinates0[0] + self.radius0 + 2)):
                        if (first_red == 0 and img2[x, y, 0] == 255):
                            first_red = 1
                            continue
                        if (first_red == 1 and img2[x, y, 0] < 255):
                            in_circle = 1
                            self.luminance_sum += (0.2126 * img2[x, y, 0] + 0.7152 * img2[x, y, 1] + 0.00722 * img2[
                                x, y, 2])/1000
                            continue
                        if (first_red == 1 and in_circle == 1 and img2[x, y, 0] == 255):
                            break

                self.ecam0_7.delete(0, 'end')
                self.ecam0_7.insert(0, self.luminance_sum)
            except:
                messagebox.showerror("Error", "can't calculate iluminance")
        else:
            messagebox.showerror("Error", "Camera is off")

    def cam0_saveImg(self):
        if (self.cam0_on_off["text"] == "Off"):
            try:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                filename0 = os.path.join("/home/howard/Desktop/DIAGNOSER/OPTICS/IMG_" + timestr + ".jpg")
                cv2.imwrite(filename0,self.frame)
            except:
                messagebox.showerror("Error", "can't save the photo")
        else:
            messagebox.showerror("Error", "Camera is off")

    def apply_camera0_setting(self):
        brightnessSetting = self.ecam0_3.get()
        brightnessSetting = float(brightnessSetting)
        self.cap0.set(cv2.CAP_PROP_BRIGHTNESS,brightnessSetting)
        contrastSetting = self.ecam0_4.get()
        contrastSetting = float(contrastSetting)
        self.cap0.set(cv2.CAP_PROP_CONTRAST, contrastSetting)
        saturationSetting = self.ecam0_5.get()
        saturationSetting = float(saturationSetting)
        self.cap0.set(cv2.CAP_PROP_SATURATION, saturationSetting)
        gainSetting = self.ecam0_6.get()
        gainSetting = float(gainSetting)
        self.cap0.set(cv2.CAP_PROP_GAIN, gainSetting)

    def camera0_reset(self):
        self.cap0.set(cv2.CAP_PROP_BRIGHTNESS,0.5)
        self.cap0.set(cv2.CAP_PROP_CONTRAST, 0.337)
        self.cap0.set(cv2.CAP_PROP_SATURATION, 0.43)
        self.cap0.set(cv2.CAP_PROP_GAIN, 0.0)
        self.ecam0_3.delete(0,'end')
        self.ecam0_3.insert(0, "0.5")
        self.ecam0_4.delete(0,'end')
        self.ecam0_4.insert(0, "0.337")
        self.ecam0_5.delete(0,'end')
        self.ecam0_5.insert(0, "0.43")
        self.ecam0_6.delete(0,'end')
        self.ecam0_6.insert(0, "0.0")



    # def connect(self):
    #     self.t3 = threading.Thread(target=self.get_data)
    #     self.t3.daemon = True
    #     self.t3.start()
    #
    # def get_data(self):
    #     # global filter_data
    #
    #     while (1):
    #         try:
    #             self.serial_data = self.ArduinoSerial.readline().decode('ascii').strip('\n').strip('\r')
    #
    #             # self.filter_data = self.serial_data.split(',')
    #             # print(self.filter_data)
    #             self.content_text.insert(tk.END, "--- receive from arduino ---" + self.serial_data + '\n')
    #             self.content_text.insert(tk.END, "--- seconds ---" + str(time.time() - self.start_time) + '\n')
    #             self.content_text.see("end")
    #
    #         except TypeError:
    #             pass

    def LED_send_data(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        if (self.button_send_LED["text"]=="LEDs on"):
            self.button_send_LED["text"]="LEDs off"
        else:
            self.button_send_LED["text"] = "LEDs on"
        if (self.button_send_LED["text"] == "LEDs off"):
            try:
                self.ArduinoSerial.write(("l,1").encode("utf-8"))
                print("l,1")
            except:
                messagebox.showerror("Error", "Can't access LED")
        else:
            try:
                self.ArduinoSerial.write(("l,0").encode("utf-8"))
                print("l,0")
            except:
                messagebox.showerror("Error", "Can't access LED")


    # def send_arduino(self):
    #     self.connect()
    #     # self.robotcommand=1
    #     # self.send_data()
    #     # time.sleep(2)
    #     self.robotcommand=0
    #     self.send_data()


    def create_io(self):
        ## camera0 components
        self.lcam0 = tk.Label()
        self.lcam0.place(x=10, y=105)
        # self.lcam0_1 = tk.Label(text="Optics Camera")
        # self.lcam0_1.place(x=100, y=8)
        self.lcam0_2 = tk.Label(text="Camera #")
        self.lcam0_2.place(x=10, y=28)
        self.cam0_on_off = tk.Button(text="On", command=self.cam0_control, width=6, height=1)
        self.cam0_on_off.place(x=220, y=24)
        self.cam0_combobox = ttk.Combobox(values=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                                          width=8)
        self.cam0_combobox.place(x=100,y=28)
        self.cam0_takePhoto = tk.Button(text="Take a Photo", command=self.cam0_saveImg, width=15, height=1)
        self.cam0_takePhoto.place(x=300, y=24)
        self.lcam0_3 = tk.Label(text="Brightness")
        self.lcam0_3.place(x=10, y=58)
        self.lcam0_4 = tk.Label(text="Contrast")
        self.lcam0_4.place(x=110, y=58)
        self.lcam0_5 = tk.Label(text="Saturation")
        self.lcam0_5.place(x=210, y=58)
        self.lcam0_6 = tk.Label(text="Gain")
        self.lcam0_6.place(x=310, y=58)
        self.ecam0_3 = tk.Entry(width=5)
        self.ecam0_3.place(x=10, y=78)
        self.ecam0_3.insert(0,"0.5")
        self.ecam0_4 = tk.Entry(width=5)
        self.ecam0_4.place(x=110, y=78)
        self.ecam0_4.insert(0, "0.337")
        self.ecam0_5 = tk.Entry(width=5)
        self.ecam0_5.place(x=210, y=78)
        self.ecam0_5.insert(0, "0.43")
        self.ecam0_6 = tk.Entry(width=5)
        self.ecam0_6.place(x=310, y=78)
        self.ecam0_6.insert(0, "0.0")
        self.cam0_set = tk.Button(text="Apply Setting", command=self.apply_camera0_setting, width=10, height=1)
        self.cam0_set.place(x=380, y=73)
        self.cam0_reset = tk.Button(text="Reset", command=self.camera0_reset, width=6, height=1)
        self.cam0_reset.place(x=510, y=73)
        self.cam0_luminance = tk.Button(text="Luminance", command=self.cam0_luminance_calculation, width=10, height=1)
        self.cam0_luminance.place(x=600, y=73)
        self.ecam0_7 = tk.Entry(width=10)
        self.ecam0_7.place(x=710, y=78)
        self.ecam0_7.insert(0, "0.0")

        ## arduino components
        self.ArduinoSerial = serial.Serial('/dev/ttyACM0', 115200)  # open serial port    sudo chmod a+rw /dev/ttyUSB0
        # # self.ArduinoSerial = serial.Serial('/dev/ttyUSB0', 9600)  # open serial port    sudo chmod a+rw /dev/ttyUSB0
        self.button_send_LED = Button(text="LEDs on", command=self.LED_send_data, width=12)
        self.button_send_LED.place(x=510, y=24)

def main(controller):
    root = Tk()
    root.title("Optical detection system controller")
    View(root, controller)
    root.geometry('1200x900')
    root.mainloop()


def init_program():
    robot_controller = controller.Controller()
    main(robot_controller)

if __name__ == "__main__":
    init_program()
