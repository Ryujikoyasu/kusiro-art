# src/serial_handler.py
import threading
import queue
import serial
import time
import numpy as np

class SerialWriterThread(threading.Thread):
    """
    Arduinoへのシリアル通信をバックグラウンドで処理するスレッド。
    メインループのパフォーマンスに影響を与えないように設計されている。
    """
    def __init__(self, port, baudrate, magic_byte, pixel_count):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.magic_byte = magic_byte
        self.pixel_count = pixel_count
        self.queue = queue.Queue(maxsize=2)
        self.running = False
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1, write_timeout=1)
            print(f"Successfully connected to Arduino on {self.port}")
            time.sleep(2) # Arduinoのリセット待機
            return True
        except serial.SerialException as e:
            print(f"FATAL: Could not connect to Arduino: {e}")
            return False

    def run(self):
        self.running = True
        if not self.connect():
            self.running = False
            return
            
        while self.running:
            try:
                # キューから色データを取得（タイムアウト付き）
                colors = self.queue.get(timeout=1)
                
                # パケットを構築: [マジックバイト] + [R,G,B, R,G,B, ...]
                packet = bytearray([self.magic_byte]) + colors.astype(np.uint8).tobytes()

                if self.ser and self.ser.is_open:
                    self.ser.write(packet)
            except queue.Empty:
                continue # データがなければループを続ける
            except Exception as e:
                print(f"Serial thread error: {e}")
                self.running = False
        
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("Serial thread stopped.")

    def send(self, data):
        """メインスレッドから描画データをこのスレッドに渡す"""
        if not self.running: return
        
        # キューが満杯なら、古いデータを捨てて新しいデータを入れる（最新の描画を優先）
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(data)

    def close(self):
        """スレッドを安全に停止させる"""
        print("Stopping serial thread...")
        self.running = False