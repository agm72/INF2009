import adafruit_dht
import board
import time

# Define the sensor and the pin
DHT_SENSOR = adafruit_dht.DHT22(board.D4)

def get_temperature_humidity():
    """
    Reads temperature (C) and humidity (%) from the DHT sensor.
    Returns a tuple (temperature, humidity), or (None, None) if the reading failed.
    """
    try:
        temperature_c = DHT_SENSOR.temperature
        humidity = DHT_SENSOR.humidity

        if temperature_c is None or humidity is None:
            raise RuntimeError("Sensor read returned None")

        return round(temperature_c, 2), round(humidity, 2)
    
    except RuntimeError as err:
        print(f"[ERROR] Sensor read error: {err}")
        return None, None
    
    except Exception as err:
        print(f"[ERROR] Unexpected error: {err}")
        return None, None
