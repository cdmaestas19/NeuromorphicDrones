"""One command for the whole stack: start the sim, open the camera view, and optionally fly.

    launch.cmd               sim + camera window
    launch.cmd waypoints     sim + camera window, then fly the baseline waypoint route
    launch.cmd --no-camera waypoints

Ctrl-C during a flight lands the drone and then shuts everything down. Otherwise, closing the MuJoCo
window or pressing Ctrl-C shuts everything down. Sim and camera output go to flypilot/logs/.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from flight import SIM_URI, sim_is_up

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / 'logs'
FLIGHTS = {'waypoints': 'fly_waypoints.py'}


def start(name: str, cmd: list[str]) -> subprocess.Popen:
    """Start a background process in its own session, so terminal Ctrl-C never reaches it."""
    LOG_DIR.mkdir(exist_ok=True)
    out = open(LOG_DIR / f'{name}.log', 'w')
    return subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def stop(proc: subprocess.Popen, timeout: float = 5.0):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def log_tail(name: str, lines: int = 15) -> str:
    text = (LOG_DIR / f'{name}.log').read_text(errors='replace').splitlines()
    return '\n'.join(text[-lines:])


def check_gpu():
    """Print the OpenGL renderer and warn on software rendering, where flights fail (see run.txt)."""
    try:
        out = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return
    renderer = next((line.split(':', 1)[1].strip() for line in out.splitlines()
                     if 'OpenGL renderer string' in line), None)
    if renderer is None:
        return
    print(f'Graphics: {renderer}', flush=True)
    if any(sw in renderer for sw in ('llvmpipe', 'softpipe', 'Basic Render')):
        print('WARNING: no GPU acceleration. The sim will fall behind real time and flights can drift '
              'or tumble. See "No GPU" in flypilot/run.txt.', flush=True)


def wait_for_sim(sim: subprocess.Popen, timeout: float = 90.0):
    print('Starting CrazySim (MuJoCo window will open)...', flush=True)
    deadline = time.monotonic() + timeout
    while not sim_is_up(SIM_URI):
        if sim.poll() is not None:
            sys.exit(f'The sim exited during startup. Last lines of logs/sim.log:\n{log_tail("sim")}')
        if time.monotonic() > deadline:
            sys.exit(f'The sim did not come up within {timeout:.0f} s. See logs/sim.log.')
        time.sleep(0.5)
    print('Sim is up.', flush=True)


def wait_for_camera(camera: subprocess.Popen, timeout: float = 20.0):
    """Wait until the camera view is connected to the CPX bridge before any flight connects.

    When a camera client connects, the bridge registers itself on the sim's cflib port, which briefly
    steals replies from whoever else is talking to the drone. A flight connecting at that moment can
    lose part of its handshake, so let the camera settle first.
    """
    deadline = time.monotonic() + timeout
    while '[cpx] Client connected' not in (LOG_DIR / 'sim.log').read_text(errors='replace'):
        if camera.poll() is not None:
            print(f'The camera view exited. Last lines of logs/camera.log:\n{log_tail("camera")}', flush=True)
            return
        if time.monotonic() > deadline:
            print('Camera view not connected yet; continuing without waiting.', flush=True)
            return
        time.sleep(0.2)


def run_flight(script: str, sim: subprocess.Popen) -> bool:
    """Run a flight script in the foreground; returns True if the user pressed Ctrl-C during it."""
    interrupted = False

    def on_sigint(_sig, _frame):
        nonlocal interrupted
        interrupted = True  # the flight script got the same Ctrl-C and is landing; wait for it

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        flight = subprocess.Popen([sys.executable, script], cwd=HERE)
        while flight.poll() is None:
            if sim.poll() is not None:
                print('The sim closed during the flight.', flush=True)
            time.sleep(0.2)
    finally:
        signal.signal(signal.SIGINT, previous)
    print(f'{script} finished with exit code {flight.returncode}.', flush=True)
    return interrupted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('flight', nargs='?', choices=sorted(FLIGHTS), help='flight to run once the sim is up')
    ap.add_argument('--no-camera', action='store_true', help="don't open the camera window")
    args = ap.parse_args()

    check_gpu()
    procs = []
    try:
        sim = start('sim', ['bash', 'run_sim.sh'])
        procs.append(sim)
        wait_for_sim(sim)
        if not args.no_camera:
            camera = start('camera', [sys.executable, 'camera_view.py'])
            procs.append(camera)
            if args.flight:
                wait_for_camera(camera)
        if args.flight and run_flight(FLIGHTS[args.flight], sim):
            return
        print('Running. Close the MuJoCo window or press Ctrl-C to shut everything down.', flush=True)
        while sim.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print('Shutting down...', flush=True)
        for proc in reversed(procs):
            stop(proc)
        subprocess.run(['bash', 'stop_sim.sh'], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('Done.', flush=True)


if __name__ == '__main__':
    main()
