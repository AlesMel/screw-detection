import sys
import cv2
import socket
import pickle
import struct
import argparse
import numpy as np
import tensorflow as tf

from ultralytics import YOLO

BUF_SIZE = 4096

# Create the parser
parser = argparse.ArgumentParser(description="Program description")

# Create a mutually exclusive group
group = parser.add_mutually_exclusive_group()

# Add the options to the group
group.add_argument("--detect", action="store_true", help="Option A description")
group.add_argument("--classify", action="store_true", help="Option B description")

# Parse the command-line arguments
args = parser.parse_args()

# Check which option was chosen
if args.detect:
    print("Using YOLO detection")
    model = YOLO("yolov8n.pt")

elif args.classify:
    print("Using CNN classification")
    model = tf.keras.models.load_model("../ml/classifier/mdl/classifier_mdl.h5")

else:
    print("No mode chosen ['--detect', '--classify']")
    sys.exit()


def main():
    # Configuration
    host = "127.0.0.1"
    port = 6666

    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"[*] Listening on {host}:{port}")

    client_socket, client_address = server_socket.accept()
    while True:
        data_size_bytes = client_socket.recv(4)
        data_size = struct.unpack("!I", data_size_bytes)[0]

        data = b""
        while len(data) < data_size:
            remaining_data = data_size - len(data)
            chunk = client_socket.recv(min(remaining_data, BUF_SIZE))
            if not chunk:
                break
            data += chunk

        if len(data) != data_size:
            print("Error: Incomplete data received")
        else:
            received = pickle.loads(data)
            if args.detect:
                results = model.predict(received, imgsz=640, conf=0.5)
                res_plotted = results[0].plot()
                cv2.imshow("result", res_plotted)
                # client_socket.sendall(b"niceru\n")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    server_socket.close()
                    break
            elif args.classify:
                result = model.predict(np.expand_dims(received, axis=0), verbose=0)
                predicted_class = np.argmax(result, axis=1)
                print(predicted_class)
                client_socket.sendall(predicted_class[0])
                # confidence = np.max(predictions)


if __name__ == "__main__":
    main()

