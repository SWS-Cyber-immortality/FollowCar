import paho.mqtt.client as mqtt
import cv2
import io
import time
import zlib  # Import the zlib library

# MQTT broker configuration
broker_address = "172.25.101.155"  # Update with your broker address
broker_port = 1883  # Update with your broker port
topic = "Video"  # Update with your desired topic

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution (adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Connect to the MQTT broker
client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to MQTT broker")
    else:
        print("Connection failed")

client.on_connect = on_connect
client.connect(broker_address, broker_port, 60)
client.loop_start()

try:
    # Capture and publish video frames continuously
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if ret:
            # Convert the frame to JPEG format
            _, encoded_frame = cv2.imencode(".jpg", frame)

             # Compress the frame using zlib
            compressed_frame = zlib.compress(encoded_frame.tobytes())

            # Publish the compressed frame to the MQTT broker
            client.publish(topic, compressed_frame)

        # Wait for a while before capturing the next frame
        time.sleep(0.1)

finally:
    # Clean up resources
    client.loop_stop()
    client.disconnect()
    cap.release()
    cv2.destroyAllWindows()
