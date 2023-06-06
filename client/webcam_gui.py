import pickle
import socket
import struct
import snap7

import numpy as np
import tkinter as tk
import pyrealsense2 as rs
import os
import cv2

from PIL import Image, ImageTk


class WebcamApp:
    def __init__(self, window_title="Object detection inference client"):
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_app)
        self.window.geometry("800x640")
        self.framerate = 100
        self.window.config(bg="#3B4252")
        self.x = 655
        self.y = 535
        self.frame_counter = 0

        # Properties
        self.server_ip = "127.0.0.1"
        self.server_port = 6666
        self.socket = None
        self.sock_connected = False
        self.detect = False
        self.snapshot_img = None

        # Realsense
        self.realsense = False
        self.pipeline = None
        self.config = None
        self.pipeline_wrapper = None
        self.pipeline_profile = None
        self.device = None
        self.device_product_line = None

        # PLC
        self.plc = snap7.client.Client()
        self.plc_ip = "10.7.14.118"
        self.plc_connected = False
        self.plc_data_byte = None

        # Video
        self.canvas = tk.Label(
            width=640,
            height=640,
            bd=0,
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.place(x=0, y=0)
        self.video_capture = cv2.VideoCapture(0)

        # Ip addr entry
        self.ip_entry = tk.Entry(
            width=15,
            bg="#4C566A",
            bd=0,
            highlightthickness=0,
            relief="flat",
            fg="white",
        )
        self.ip_entry.pack()
        self.ip_entry.place(x=self.x, y=self.y)
        self.ip_entry.insert(0, "localhost")
        self.ip_entry.bind("<FocusOut>", self.get_ip)
        self.ip_entry.bind("<Button-1>", self.ip_onclick)

        # Port entry
        self.port_entry = tk.Entry(
            width=15,
            bg="#4C566A",
            bd=0,
            highlightthickness=0,
            relief="flat",
            fg="white",
        )
        self.port_entry.pack()
        self.port_entry.place(x=self.x, y=self.y + 30)
        self.port_entry.insert(0, "6666")
        self.port_entry.bind("<FocusOut>", self.get_port)
        self.port_entry.bind("<Button-1>", self.port_onclick)

        # Connect button
        self.connect_btn = tk.Button(
            text="Connect",
            command=self.connect_to_socket,
            width=12,
            bd=0,
            fg="white",
            highlightthickness=0,
            relief="flat",
            anchor="center",
            justify="center",
        )
        self.connect_btn.place(x=self.x, y=self.y + 60)
        self.connect_btn.config(bg="#5E81AC")

        # Detect button
        self.detect_btn = tk.Button(
            text="Detect",
            command=self.start_detection,
            width=12,
            bd=0,
            fg="white",
            highlightthickness=0,
            relief="flat",
            anchor="center",
            justify="center",
        )
        self.detect_btn.place(x=self.x, y=250)
        self.detect_btn.config(bg="#A3BE8C")

        # Stop button
        self.stop_btn = tk.Button(
            text="Stop",
            command=self.stop_detection,
            width=12,
            bd=0,
            fg="white",
            highlightthickness=0,
            relief="flat",
            anchor="center",
            justify="center",
        )
        self.stop_btn.place(x=self.x, y=285)
        self.stop_btn.config(bg="#BF616A")

        # Realsense Buttnon
        self.realsense_btn = tk.Button(
            text="Use RealSense",
            command=self.use_realsense,
            width=12,
            bd=0,
            fg="white",
            highlightthickness=0,
            relief="flat",
            anchor="center",
            justify="center",
        )
        self.realsense_btn.place(x=self.x, y=320)
        self.realsense_btn.config(bg="#88c0d0")

        # Realsense Buttnon
        self.realsense_btn = tk.Button(
            text="Snapshot",
            command=self.snapshot,
            width=12,
            bd=0,
            fg="white",
            highlightthickness=0,
            relief="flat",
            anchor="center",
            justify="center",
        )
        self.realsense_btn.place(x=self.x, y=355)
        self.realsense_btn.config(bg="#d08770")

        # PLC Buttnon
        self.plc_button = tk.Button(
            text="PLC Connect",
            command=self.connect_to_plc,
            width=12,
            bd=0,
            fg="white",
            highlightthickness=0,
            relief="flat",
            anchor="center",
            justify="center",
        )
        self.plc_button.place(x=self.x, y=385)
        self.plc_button.config(bg="#d08770")

        self.update_frame()

        self.window.mainloop()

    def close_app(self):
        if self.socket is not None:
            self.socket.close()
        if self.realsense:
            self.pipeline.stop()
        self.video_capture.release()
        self.window.destroy()

    def update_frame(self):
        if self.realsense:
            frame = self.realsense_capture()
        else:
            frame = self.generic_capture()

        if frame is not None:
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.snapshot_img = frame
            self.canvas.photo_image = photo
            self.canvas.configure(image=photo)

        self.window.after(self.framerate, self.update_frame)

        if (self.frame_counter % 5 == 0) and (self.sock_connected) and self.detect:
            self.send_frame()
            pred_class = int.from_bytes(self.socket.recv(1024), byteorder="little")
            print(pred_class)
            if self.plc_connected:
                self.null_db()
                snap7.util.set_byte(self.plc_data_byte, pred_class, 128)
                self.plc.db_write(100, 0, self.plc_data_byte)
            # print(int.from_bytes(msg, byteorder="big"))

        self.frame_counter += 1

    def ip_onclick(self, event):
        self.ip_entry.delete(0, tk.END)

    def port_onclick(self, event):
        self.port_entry.delete(0, tk.END)

    def get_ip(self, event):
        self.server_ip = self.ip_entry.get()

    def get_port(self, event):
        try:
            self.server_port = int(self.port_entry.get())
        except:
            print("Invalid port entry")

    def start_detection(self, *args):
        self.detect = True

    def stop_detection(self, *args):
        self.detect = False

    def use_realsense(self, *args):
        self.video_capture.release()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(
            self.device.get_info(rs.camera_info.product_line)
        )

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                print("Found RealSense")
                found_rgb = True
        if not found_rgb:
            print("RGB Camera required")
            exit(0)

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.realsense = True

    def realsense_capture(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        color_image = cv2.resize(color_image, (640, 640))

        return color_image

    def generic_capture(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 640))
            return frame
        else:
            return None

    def connect_to_socket(self, *args):
        if self.server_port is not None or self.server_ip is not None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.sock_connected = True
            print(f"[*] Connected to {self.server_ip}:{self.server_port}")
        else:
            print("Enter full server information")

    def connect_to_plc(self, *args):
        try:
            self.plc.connect(self.plc_ip, 0, 1)
        except:
            print("Could not connect to PLC!")
        else:
            print("Succesfully connected to PLC!")
            self.plc_connected = True
            self.plc_data_byte = self.plc.db_read(100, 0, 9)

    def send_frame(self, *args):
        if self.sock_connected:
            # img = cv2.cvtColor(self.snapshot_img, cv2.COLOR_BGR2GRAY)
            data = pickle.dumps(self.snapshot_img)
            # print(f"Sending {len(data)} bytes")
            data_size_bytes = struct.pack("!I", len(data))
            self.socket.sendall(data_size_bytes)
            self.socket.sendall(data)
        else:
            print("Cannot send image because socket not connected")

    def snapshot(self, *args):
        image_dir = "D:\\FEI-STU\\TP\\network\\client\\dataset"
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

        highest_num = 0
        for image_file in image_files:
            num = int(image_file[7:-4])
            if num > highest_num:
                highest_num = num
        next_num = highest_num + 1
        next_filename = f"screws_{next_num:03}.png"
        next_filepath = os.path.join(image_dir, next_filename)
        frame = cv2.cvtColor(self.snapshot_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(next_filepath, frame)

    def null_db(self, *args):
        for i in range(0, 9):
            snap7.util.set_byte(self.plc_data_byte, i, 0)

        # self.plc.db_write(100, 0, self.plc_data_byte)


if __name__ == "__main__":
    app = WebcamApp()
