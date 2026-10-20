"""Camera frames from the Crazyflie AI-deck, read over its CPX-over-TCP protocol.

In simulation the stream comes from CrazySim's crazysim_cpx.py bridge on tcp://127.0.0.1:5050; on
hardware it comes from the AI-deck's WiFi streamer (port 5000 on the deck's IP). Everything that
consumes vision -- the viewer now, the looming detector and connectome policy later -- should get its
frames through FrameSource so it works unchanged on both.
"""
import socket
import struct
import threading
import time

import cv2
import numpy as np

CPX_FUNCTION_APP = 5
CPX_HEADER = struct.Struct('<HBB')     # length (counts route + function bytes), route, function
IMG_HEADER = struct.Struct('<BHHBBI')  # magic, width, height, depth, format, size
IMG_MAGIC = 0xBC
FMT_RAW, FMT_JPEG = 0, 1


class FrameSource:
    """Background reader that always holds the newest AI-deck frame.

    get_frame() blocks until a frame newer than the last one it returned arrives, so a consumer never
    processes the same frame twice and never works through a stale backlog.

    The CrazySim bridge serves one client at a time: a second FrameSource (say, camera_view.py while a
    flight script also reads frames) disconnects the first, so keep one per stream.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 5050, reconnect_delay: float = 2.0):
        self.host = host
        self.port = port
        self.connected = False
        self.frame_time = 0.0  # time.monotonic() when the frame last returned by get_frame() arrived
        self._reconnect_delay = reconnect_delay
        self._cond = threading.Condition()
        self._frame = None
        self._stamp = 0.0
        self._seq = 0
        self._returned_seq = 0
        self._running = False
        self._thread = None

    def __enter__(self) -> 'FrameSource':
        return self.start()

    def __exit__(self, *exc):
        self.close()

    @property
    def frames_received(self) -> int:
        return self._seq

    def start(self) -> 'FrameSource':
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, name='FrameSource', daemon=True)
            self._thread.start()
        return self

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_frame(self, timeout: float = 1.0) -> np.ndarray:
        """Newest frame not yet returned, as an H x W uint8 grayscale array.

        Raises TimeoutError if no new frame arrives within `timeout` seconds.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._returned_seq, timeout):
                raise TimeoutError(f'no camera frame from tcp://{self.host}:{self.port} within {timeout} s')
            self._returned_seq = self._seq
            self.frame_time = self._stamp
            return self._frame

    def latest(self) -> np.ndarray | None:
        """Newest frame without waiting (possibly the same one as last call); None before the first.

        For control loops that run at their own rate and just want the freshest view.
        """
        with self._cond:
            return self._frame

    def _run(self):
        while self._running:
            try:
                with socket.create_connection((self.host, self.port), timeout=3.0) as sock:
                    sock.settimeout(1.0)  # lets close() be noticed while the stream is idle
                    self.connected = True
                    self._read_frames(sock)
            except OSError:
                pass  # bridge not up yet, restarted, or another client took the stream
            finally:
                self.connected = False
            if self._running:
                time.sleep(self._reconnect_delay)

    def _read_frames(self, sock: socket.socket):
        header, pixels = None, bytearray()
        while self._running:
            length, _route, function = CPX_HEADER.unpack(self._recv_exact(sock, CPX_HEADER.size))
            data = self._recv_exact(sock, length - 2) if length > 2 else b''
            if function & 0x3F != CPX_FUNCTION_APP:
                continue  # CRTP traffic the bridge also forwards on this socket
            if len(data) == IMG_HEADER.size and data[0] == IMG_MAGIC:
                header, pixels = IMG_HEADER.unpack(data), bytearray()
                continue
            if header is None:
                continue  # joined mid-frame; wait for the next image header
            pixels += data
            size = header[5]
            if len(pixels) >= size:
                frame = self._decode(header, bytes(pixels[:size]))
                header = None
                if frame is not None:
                    self._publish(frame)

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                if not self._running:
                    raise ConnectionAbortedError('FrameSource closed')
                continue
            if not chunk:
                raise ConnectionError('camera stream closed by peer')
            buf += chunk
        return bytes(buf)

    @staticmethod
    def _decode(header: tuple, payload: bytes) -> np.ndarray | None:
        _magic, width, height, _depth, fmt, _size = header
        if fmt == FMT_JPEG:
            return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if len(payload) != width * height:
            return None
        # Mono8 from the sim's HM01B0 model. A color AI-deck sends Bayer here; demosaic it before use.
        return np.frombuffer(payload, np.uint8).reshape(height, width)

    def _publish(self, frame: np.ndarray):
        with self._cond:
            self._frame = frame
            self._stamp = time.monotonic()
            self._seq += 1
            self._cond.notify_all()
