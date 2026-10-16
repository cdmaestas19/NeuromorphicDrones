"""Live view of the drone's AI-deck camera (CrazySim's FPV render, or a real AI-deck).

Run from Windows with camera.cmd (inside WSL: bash py.sh camera_view.py). Start the sim first.
Keys: q or Esc quits, s saves the current frame to flypilot/snapshots/.
"""
import argparse
import pathlib
import time

import cv2
import numpy as np

from frames import FrameSource

WINDOW = 'AI-deck camera'


def _placeholder(text: str, scale: float) -> np.ndarray:
    img = np.zeros((int(244 * scale), int(324 * scale), 3), np.uint8)
    cv2.putText(img, text, (10, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5050,
                    help='CPX TCP port: 5050 for the CrazySim bridge, 5000 for a real AI-deck')
    ap.add_argument('--scale', type=float, default=2.0, help='window magnification (default 2x)')
    args = ap.parse_args()

    snap_dir = pathlib.Path(__file__).resolve().parent / 'snapshots'
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW, _placeholder(f'waiting for tcp://{args.host}:{args.port} ...', args.scale))
    print(f'Reading tcp://{args.host}:{args.port}. q/Esc quits, s saves a snapshot.')

    frame, fps, t_prev = None, 0.0, None
    with FrameSource(args.host, args.port) as cam:
        try:
            while True:
                try:
                    frame = cam.get_frame(timeout=0.05)
                    if t_prev is not None:
                        fps = 0.9 * fps + 0.1 / max(cam.frame_time - t_prev, 1e-3)
                    t_prev = cam.frame_time
                    view = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
                    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                    cv2.putText(view, f'{frame.shape[1]}x{frame.shape[0]}  {fps:4.1f} fps', (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(WINDOW, view)
                except TimeoutError:
                    pass

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('s') and frame is not None:
                    snap_dir.mkdir(exist_ok=True)
                    path = snap_dir / time.strftime('frame_%Y%m%d_%H%M%S.png')
                    cv2.imwrite(str(path), frame)
                    print(f'saved {path}')
        except KeyboardInterrupt:
            pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
