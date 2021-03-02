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
import statistics

import matplotlib.pyplot as plt
import math
import logging
from time import gmtime, strftime, localtime
from pySerialTransfer import pySerialTransfer as txfer

class View():

    link_ardu0 = 0  # arduino 0: led
    link_ardu1 = 0  # arduino 1: stage
    link_temp = 0  # temp arduino
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


    # def send_arduino(self):
    #     self.connect()
    #     # self.robotcommand=1
    #     # self.send_data()
    #     # time.sleep(2)
    #     self.robotcommand=0
    #     self.send_data()

    ## arduino init
    def recognize_board(self):
        try:
            send_size = 0
            str_ = "no"
            str_size = self.link_temp.tx_obj(str_, send_size) - send_size
            send_size += str_size
            self.link_temp.send(send_size)
            while not self.link_temp.available():
                if self.link_temp.status < 0:
                    if self.link_temp.status == txfer.CRC_ERROR:
                        print('ERROR: CRC_ERROR')
                    elif self.link_temp.status == txfer.PAYLOAD_ERROR:
                        print('ERROR: PAYLOAD_ERROR')
                    elif self.link_temp.status == txfer.STOP_BYTE_ERROR:
                        print('ERROR: STOP_BYTE_ERROR')
                    else:
                        print('ERROR: {}'.format(self.link_temp.status))
            rec_str_ = self.link_temp.rx_obj(obj_type=str, obj_byte_size=2, start_pos=0)
            if rec_str_ == "00":
                self.link_ardu0 = self.link_temp
                print("coonected successfully to ardu0")
                # Press ctrl-c or ctrl-d on the keyboard to exit
            elif rec_str_ == "01":
                self.link_ardu1 = self.link_temp
                print("coonected successfully to ardu1")
        except (KeyboardInterrupt, EOFError, SystemExit):
            print("sending command exits.")



    ## connect ACM0
    def create_io_ACM0_1(self):

        try:
            self.link_temp = txfer.SerialTransfer('/dev/ttyACM0')
            self.link_temp.open()
            time.sleep(2)  # allow some time for the Arduino to completely reset
            print("ACM0 arduino serial transfer link is connected successfully!")
            self.recognize_board()
        except KeyboardInterrupt:
            try:
                self.link_temp.close()
                print("ACM0 arduino serial transfer connection fails due to interruption!")
            except:
                print("ACM0 arduino serial transfer connection fails due to unknown reason!")

    ## connect ACM1
    def create_io_ACM1_1(self):
        try:
            self.link_temp = txfer.SerialTransfer('/dev/ttyACM1')
            self.link_temp.open()
            time.sleep(2)  # allow some time for the Arduino to completely reset
            print("ACM1 arduino serial transfer link is connected successfully!")
            self.recognize_board()
        except KeyboardInterrupt:
            try:
                self.link_temp.close()
                print("ACM1 arduino serial transfer connection fails due to interruption!")
            except:
                print("ACM1 arduino serial transfer connection fails due to unknown reason!")


    def LED_send_data(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        if (self.button_send_LED["text"]=="LEDs on"):
            self.button_send_LED["text"]="LEDs off"
        else:
            self.button_send_LED["text"] = "LEDs on"
        if (self.button_send_LED["text"] == "LEDs off"):
            try:
                # self.ArduinoSerial.write(("l,1").encode("utf-8"))
                # print("l,1")
                self.str_ = "1"
                self.led_board_send_command()
            except:
                messagebox.showerror("Error", "Can't access LED")
        else:
            try:
                # self.ArduinoSerial.write(("l,0").encode("utf-8"))
                # print("l,0")
                self.str_ = "0"
                self.led_board_send_command()
            except:
                messagebox.showerror("Error", "Can't access LED")

    def led_board_send_command(self):
        try:
            # user_input = input()
            send_size = 0
            str_size = self.link_ardu0.tx_obj(self.str_, send_size) - send_size
            send_size += str_size
            self.link_ardu0.send(send_size)
            while not self.link_ardu0.available():
                if self.link_ardu0.status < 0:
                    if self.link_ardu0.status == txfer.CRC_ERROR:
                        print('ERROR: CRC_ERROR')
                    elif self.link_ardu0.status == txfer.PAYLOAD_ERROR:
                        print('ERROR: PAYLOAD_ERROR')
                    elif self.link_ardu0.status == txfer.STOP_BYTE_ERROR:
                        print('ERROR: STOP_BYTE_ERROR')
                    else:
                        print('ERROR: {}'.format(self.link_ardu0.status))
            rec_str_ = self.link_ardu0.rx_obj(obj_type=str, obj_byte_size=2, start_pos=0)
            print("send: ", self.str_)
            print("recv: ", rec_str_)

            # Press ctrl-c or ctrl-d on the keyboard to exit
        except (KeyboardInterrupt, EOFError, SystemExit):
            print("sending command exits.")
            # break

    def home_stage(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        try:
            # self.ArduinoSerial.write(("l,1").encode("utf-8"))
            # print("l,1")

            self.str_ = "homeleft"
            self.stage_board_send_command()
            time.sleep(2)
            self.str_ = "homeright"
            self.stage_board_send_command()
            time.sleep(2)
            self.str_ = "hometurnleft"
            self.stage_board_send_command()
        except:
            messagebox.showerror("Error", "Can't home stage")

    def stage_turn_right(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        try:
            # self.ArduinoSerial.write(("l,1").encode("utf-8"))
            # print("l,1")

            # step_text = self.entry_stage_turn_right.get()
            self.str_ = "turnright," + self.entry_stage_turn_right.get()
            self.stage_board_send_command()
        except:
            messagebox.showerror("Error", "Can't turn right")

    def stage_turn_left(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        try:
            # self.ArduinoSerial.write(("l,1").encode("utf-8"))
            # print("l,1")
            self.str_ = "turnleft," + self.entry_stage_turn_left.get()
            self.stage_board_send_command()
        except:
            messagebox.showerror("Error", "Can't turn left")

    def stage_move_right(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        try:
            # self.ArduinoSerial.write(("l,1").encode("utf-8"))
            # print("l,1")
            self.str_ = "moveright," + self.entry_stage_move_right.get()
            self.stage_board_send_command()
        except:
            messagebox.showerror("Error", "Can't move right")

    def stage_move_left(self):
        # self.ArduinoSerial.write((str(self.robotcommand)).encode("utf-8"))
        try:
            # self.ArduinoSerial.write(("l,1").encode("utf-8"))
            # print("l,1")
            self.str_ = "moveleft," + self.entry_stage_move_left.get()
            self.stage_board_send_command()
        except:
            messagebox.showerror("Error", "Can't move left")


    def stage_board_send_command(self):
        try:
            # user_input = input()
            send_size = 0
            str_size = self.link_ardu1.tx_obj(self.str_, send_size) - send_size
            send_size += str_size
            self.link_ardu1.send(send_size)
            while not self.link_ardu1.available():
                if self.link_ardu1.status < 0:
                    if self.link_ardu1.status == txfer.CRC_ERROR:
                        print('ERROR: CRC_ERROR')
                    elif self.link_ardu1.status == txfer.PAYLOAD_ERROR:
                        print('ERROR: PAYLOAD_ERROR')
                    elif self.link_ardu1.status == txfer.STOP_BYTE_ERROR:
                        print('ERROR: STOP_BYTE_ERROR')
                    else:
                        print('ERROR: {}'.format(self.link_ardu1.status))
            rec_str_ = self.link_ardu1.rx_obj(obj_type=str, obj_byte_size=2, start_pos=0)
            print("send: ", self.str_)
            print("recv: ", rec_str_)

            # Press ctrl-c or ctrl-d on the keyboard to exit
        except (KeyboardInterrupt, EOFError, SystemExit):
            print("sending command exits.")
            # break



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

        # # self.ArduinoSerial = serial.Serial('/dev/ttyUSB0', 9600)  # open serial port    sudo chmod a+rw /dev/ttyUSB0
        self.button_send_LED = Button(text="LEDs on", command=self.LED_send_data, width=12)
        self.button_send_LED.place(x=460, y=24)

        self.button_stage_home = Button(text="HOME STAGE", command=self.home_stage, width=12)
        self.button_stage_home.place(x=610, y=24)

        self.button_stage_turn_right = Button(text="STAGE TURN->", command=self.stage_turn_right, width=12)
        self.button_stage_turn_right.place(x=810, y=24)
        self.entry_stage_turn_right = tk.Entry(width=6)
        self.entry_stage_turn_right.place(x=940, y=24)
        self.entry_stage_turn_right.insert(0, 0)

        self.button_stage_turn_left = Button(text="STAGE TURN<-", command=self.stage_turn_left, width=12)
        self.button_stage_turn_left.place(x=810, y=54)
        self.entry_stage_turn_left = tk.Entry(width=6)
        self.entry_stage_turn_left.place(x=940, y=54)
        self.entry_stage_turn_left.insert(0, 0)

        self.button_stage_move_right = Button(text="STAGE MOVE->", command=self.stage_move_right, width=12)
        self.button_stage_move_right .place(x=810, y=84)
        self.entry_stage_move_right  = tk.Entry(width=6)
        self.entry_stage_move_right .place(x=940, y=84)
        self.entry_stage_move_right .insert(0, 0)

        self.button_stage_move_left = Button(text="STAGE MOVE<-", command=self.stage_move_left, width=12)
        self.button_stage_move_left.place(x=810, y=114)
        self.entry_stage_move_left = tk.Entry(width=6)
        self.entry_stage_move_left.place(x=940, y=114)
        self.entry_stage_move_left.insert(0, 0)

        self.create_io_ACM0_1()
        self.create_io_ACM1_1()

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
