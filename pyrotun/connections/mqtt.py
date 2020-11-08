import os

import paho.mqtt.client as pahomqttclient


class MqttConnection:
    def __init__(self, host="", port=1883):

        if not host:
            self.host = os.getenv("MQTT_HOST")
        else:
            self.host = host

        if not self.host:
            raise ValueError("MQTT_HOST not provided")

        self.client = pahomqttclient.Client()
        self.client.connect(self.host, port)

    def publish(self, topic, payload):
        self.client.publish(topic=topic, payload=str(payload))
