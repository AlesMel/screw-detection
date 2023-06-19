import socket
import time
import numpy as np

# Constants
HOST = "127.0.0.1"
PORT = 6666
DATA = np.array([1, 2, 3, 4, 5])

def main():
    socket_config()

def socket_config():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _socket:
        _socket.bind((HOST, PORT))
        print("[OK] Binding")
        _socket.listen()
        print("[OK] Listening")
        conn, addr = _socket.accept()
        print("[OK] Accepted")
        with conn:
            print(f"Connected from {addr}")
            while True:
                print(f"[RAW]: {DATA}")
                print(f"[UI8]: {DATA.astype(np.uint8)}")
                conn.sendall(DATA.astype(np.uint8))
                time.sleep(1)

if __name__ == "__main__":
    main()