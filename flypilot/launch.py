"""One command for the whole stack: start the sim, optionally fly, all shown in the MuJoCo window.

    launch.cmd                          sim only (drone camera inset top-left of the 3D view)
    launch.cmd waypoints                ... then fly the baseline waypoint route
    launch.cmd avoid                    ... with the avoidance dashboard on the right of the 3D view and
                                        the Stops window, then the avoidance demo
    launch.cmd avoid --route square     (options after the flight name go to the flight script)
    launch.cmd deliver                  the same with no stops yet: add them in the Stops window (or
                                        double-click the floor) (avoid_demo.py --route live)
    launch.cmd avoid --route crossing   a route in the street scene (people, a cart): the sim starts on
                                        the scene the route asks for; deliver takes --scene street
Every flight follows paths planned on a map (planner.py).
    launch.cmd --camera-window [waypoints]   also open the separate camera_view window (checks the
                                        AI-deck stream's fps; not with avoid/deliver, which read the stream)
    launch.cmd avoid --screen main      where the windows open (screens.py): auto (default: a screen
                                        other than the main one, if there is one), main, 1, 2, ..., off
    launch.cmd avoid --headless         no windows at all (test runs while the PC is used for a game:
                                        nothing pops up or takes the focus)

Ctrl-C during a flight lands the drone and then shuts everything down. Otherwise, closing the MuJoCo
window or pressing Ctrl-C shuts everything down. Sim, camera and dashboard output go to flypilot/logs/.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import screens
from flight import SIM_URI, sim_is_up
from planner import SCENES
from routes import ROUTES

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / 'logs'
# flight name -> (script, dashboard.py args (None: no dashboard), default script args)
FLIGHTS = {
    'waypoints': ('fly_waypoints.py', None, []),
    'avoid': ('avoid_demo.py', ['--stops'], []),
    'deliver': ('avoid_demo.py', ['--stops'], ['--route', 'live']),
}


def start(name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
    """Start a background process in its own session, so terminal Ctrl-C never reaches it."""
    LOG_DIR.mkdir(exist_ok=True)
    out = open(LOG_DIR / f'{name}.log', 'w')
    return subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                            env=None if env is None else {**os.environ, **env})


def uses_brain(script_args: list[str]) -> bool:
    """Whether the flight's policy is the fly brain (it needs flybrain/brain_server.py running)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--policy', default='geometric')
    return p.parse_known_args(script_args)[0].policy == 'flybrain'


def start_brain(timeout: float = 180.0) -> subprocess.Popen:
    """Start the fly-brain server (its own Python environment) and wait until it listens."""
    print('Starting the fly brain (escape circuit of the FlyWire connectome)...', flush=True)
    brain = start('brain', ['bash', str(HERE.parent / 'flybrain' / 'py.sh'), '-u', 'brain_server.py'])
    deadline = time.monotonic() + timeout
    while 'ready' not in (LOG_DIR / 'brain.log').read_text(errors='replace'):
        if brain.poll() is not None:
            sys.exit(f'The fly brain exited during startup. Last lines of logs/brain.log:\n{log_tail("brain")}')
        if time.monotonic() > deadline:
            sys.exit(f'The fly brain did not start within {timeout:.0f} s; see logs/brain.log')
        time.sleep(0.5)
    print(log_tail('brain', 1), flush=True)
    return brain


def scene_for(script: str | None, script_args: list[str]) -> str:
    """The scene the flight runs in: its route's (routes.py), or --scene for live stops."""
    if script != 'avoid_demo.py':
        return 'obstacles'
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--route', default='blocked')  # avoid_demo.py's default
    p.add_argument('--scene', default='obstacles')
    ns, _ = p.parse_known_args(script_args)
    route = ROUTES.get(ns.route)
    name = route.scene if route else ns.scene
    return name if name in SCENES else 'obstacles'  # avoid_demo.py reports a bad name itself


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
    print('Starting CrazySim...', flush=True)
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


def run_flight(script: str, script_args: list[str], sim: subprocess.Popen) -> bool:
    """Run a flight script in the foreground; returns True if the user pressed Ctrl-C during it."""
    interrupted = False

    def on_sigint(_sig, _frame):
        nonlocal interrupted
        interrupted = True  # the flight script got the same Ctrl-C and is landing; wait for it

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        flight = subprocess.Popen([sys.executable, script, *script_args], cwd=HERE)
        sim_closed = False
        while flight.poll() is None:
            if not sim_closed and sim.poll() is not None:
                print('The sim closed during the flight.', flush=True)
                sim_closed = True
            time.sleep(0.2)
    finally:
        signal.signal(signal.SIGINT, previous)
    print(f'{script} finished with exit code {flight.returncode}.', flush=True)
    return interrupted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('flight', nargs='?', choices=sorted(FLIGHTS), help='flight to run once the sim is up')
    ap.add_argument('--camera-window', action='store_true',
                    help='also open the separate camera view window (its fps readout checks the AI-deck stream)')
    ap.add_argument('--headless', action='store_true',
                    help='no windows at all: no 3D view, no dashboard (test runs while the PC is used for '
                         'something else; the flight and its log are the same)')
    ap.add_argument('--screen', default='auto',
                    help='where the windows open: auto (a screen other than the main one, if there is one), '
                         'main, a number (1 = main), or off (wherever they open)')
    args, script_args = ap.parse_known_args()
    if script_args and not args.flight:
        ap.error(f'unrecognized arguments: {" ".join(script_args)}')
    script, dashboard, default_args = FLIGHTS[args.flight] if args.flight else (None, None, [])
    script_args = default_args + script_args
    if args.headless and args.camera_window:
        ap.error('--headless opens no windows: drop --camera-window')
    if args.camera_window and dashboard is not None:
        ap.error(f'--camera-window cannot be combined with {args.flight}: the camera bridge serves one client, '
                 'and the avoidance demo needs it')

    check_gpu()
    # Closing the terminal (SIGHUP) or being killed (SIGTERM) must clean up like Ctrl-C does: the sim
    # and viewers run in their own sessions, so nothing else would stop them.
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)
    stop_leftovers()
    procs = []
    placer = None
    if not args.headless:
        monitor = screens.pick(args.screen)
        if monitor is not None and screens.WindowPlacer.available():
            placer = screens.WindowPlacer(monitor)
            placer.start()
            print(f'Windows open on the {monitor.width} x {monitor.height} screen (--screen {args.screen}).', flush=True)
    try:
        scene = scene_for(script, script_args)
        if scene != 'obstacles':
            print(f'Scene: {scene} ({SCENES[scene].name})', flush=True)
        sim_env = {'SCENE': str(SCENES[scene])}
        if args.headless:
            sim_env['CRAZYSIM_HEADLESS'] = '1'  # CrazySim's launch script then leaves out the viewer (our patch)
        sim = start('sim', ['bash', 'run_sim.sh'], env=sim_env)
        procs.append(sim)
        wait_for_sim(sim)
        if dashboard is not None and not args.headless:
            procs.append(start('dashboard', [sys.executable, 'dashboard.py', *dashboard, '--scene', str(SCENES[scene])]))
        if uses_brain(script_args):
            procs.append(start_brain())
        if args.camera_window:
            camera = start('camera', [sys.executable, 'camera_view.py'])
            procs.append(camera)
            if args.flight:
                wait_for_camera(camera)
        if script and run_flight(script, script_args, sim):
            return
        print('Running. Press Ctrl-C to shut everything down.' if args.headless else
              'Running. Close the MuJoCo window or press Ctrl-C to shut everything down.', flush=True)
        while sim.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if placer is not None:
            placer.stop()
        print('Shutting down...', flush=True)
        for proc in reversed(procs):
            stop(proc)
        stop_leftovers()
        print('Done.', flush=True)


def _interrupt(_sig, _frame):
    raise KeyboardInterrupt


def stop_leftovers():
    """Kill sim processes and viewers from an earlier run that didn't shut down cleanly."""
    subprocess.run(['bash', 'stop_sim.sh'], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
