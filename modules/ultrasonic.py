import RPi.GPIO as GPIO
import time

# Define GPIO pin for SIG
SIG = 17  # Change if using another GPIO pin

# Setup GPIO Mode
GPIO.setmode(GPIO.BCM)

def get_distance():
    """Measures the distance using the ultrasonic sensor."""
    # Set SIG as OUTPUT to send the trigger pulse
    GPIO.setup(SIG, GPIO.OUT)
    GPIO.output(SIG, False)  # Ensure it's low before triggering
    time.sleep(0.002)  # Small delay to stabilize

    GPIO.output(SIG, True)  # Send 10µs trigger pulse
    time.sleep(0.00001)
    GPIO.output(SIG, False)

    # Set SIG as INPUT to receive the echo
    GPIO.setup(SIG, GPIO.IN)

    pulse_start = time.time()
    timeout = pulse_start + 0.06  # 20ms timeout

    # Wait for echo start (avoid infinite loop)
    while GPIO.input(SIG) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return -1  # Timeout error

    pulse_end = time.time()
    timeout = pulse_end + 0.02  # 20ms timeout

    # Wait for echo end (avoid infinite loop)
    while GPIO.input(SIG) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return -1  # Timeout error

    # Calculate distance (Speed of sound = 343m/s)
    duration = pulse_end - pulse_start
    distance = (duration * 34300) / 2  # Convert to cm
    return round(distance, 2)
