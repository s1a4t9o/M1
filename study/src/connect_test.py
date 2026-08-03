#HIDAS制御アプリとの接続テスト

import socket
import time

HOST = "192.168.11.4"   #HIDAS Dock
PORT = 60001

def main():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)

            print(f"{HOST}:{PORT} に接続します")
            sock.connect((HOST, PORT))
            print("接続成功")

            # 状態取得コマンド
            message = "CMB 000 100 002 000 200 100 000 000 000 000 000 000 000 000 001 000\n"
            sock.sendall(message.encode("utf-8"))
            print(f"送信: {message.strip()}")

            response = sock.recv(8192).decode("utf-8")
            print(f"受信: {response.strip()}")

            time.sleep(10)

            # 状態取得コマンド
            message = "CMB 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000\n"
            sock.sendall(message.encode("utf-8"))
            print(f"送信: {message.strip()}")

            response = sock.recv(8192).decode("utf-8")
            print(f"受信: {response.strip()}")

    except ConnectionRefusedError:
        print("接続拒否：Visual Studio側のプログラムが起動しているか確認してください。")
    except socket.timeout:
        print("タイムアウト：接続または応答を確認してください。")
    except OSError as error:
        print(f"通信エラー: {error}")


if __name__ == "__main__":
    main()