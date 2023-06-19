import socket

# Constants
HOST = "127.0.0.1"
PORT = 6666

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _socket:
        _socket.connect((HOST, PORT))
        while True:
            data = _socket.recv(256)
            print(f"[RAW]: {data}")
            print(f"[CON]: {[d for d in data]}")

if __name__ == "__main__":
    main()