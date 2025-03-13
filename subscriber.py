import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, message):
    payload = message.payload.decode()
    try:
        data = json.loads(payload)
        username = data.get("username", "Unknown")
        date_of_birth = data.get("date_of_birth", "Unknown")
        print(f"Received user data - Username: {username}, Date of Birth: {date_of_birth}")
    except json.JSONDecodeError:
        print(f"Received non-JSON message: {payload}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Subscriber")
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("face/recognized")
print("Subscriber is listening on 'face/recognized' topic...")
client.loop_forever()
