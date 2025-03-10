import subprocess
import time
from modules.ultrasonic import get_distance

# Configuration
APP_COMMAND = ['python3', 'app.py']  # Adjust as needed (e.g. python vs python3)
DISTANCE_THRESHOLD = 40.0  # in centimeters
CHECK_INTERVAL = 5         # seconds between sensor checks
AWAY_TIME = 20             # seconds to wait after user leaves before stopping the app

def main():
    app_process = None
    away_start = None

    try:
        while True:
            distance = get_distance()
            if distance == -1:
                print("Sensor error: Check wiring or sensor. Retrying...")
                time.sleep(CHECK_INTERVAL)
                continue

            print(f"Measured distance: {distance} cm")

            # If a user is detected (within threshold)
            if distance <= DISTANCE_THRESHOLD:
                # Reset any away timer
                away_start = None
                # If the Flask app isn’t already running, start it.
                if app_process is None:
                    print("User detected. Starting Flask app...")
                    app_process = subprocess.Popen(APP_COMMAND)
                else:
                    # If the process has died unexpectedly, restart it.
                    if app_process.poll() is not None:
                        print("Flask app process terminated unexpectedly. Restarting...")
                        app_process = subprocess.Popen(APP_COMMAND)
            else:
                # No user detected
                if app_process is not None:
                    # Start the away timer if not already started.
                    if away_start is None:
                        away_start = time.time()
                        print("User not detected. Starting away timer...")
                    # If user is away for the specified AWAY_TIME, terminate the Flask app.
                    elif time.time() - away_start >= AWAY_TIME:
                        print("User away. Terminating Flask app...")
                        app_process.terminate()
                        try:
                            app_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            print("Flask app did not terminate in time; killing process.")
                            app_process.kill()
                        app_process = None
                        away_start = None
                else:
                    away_start = None

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("Shutting down sensor monitor...")
        if app_process is not None:
            app_process.terminate()
            try:
                app_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app_process.kill()

if __name__ == '__main__':
    main()
