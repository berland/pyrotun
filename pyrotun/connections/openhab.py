import os
import openhab


class OpenHABConnection:
    def __init__(self, openhab_url=""):

        if not openhab_url:
            self.openhab_url = os.getenv("OPENHAB_URL")
        else:
            self.openhab_url = openhab_url

        if not openhab_url:
            raise ValueError("openhab_url not set")

        self.client = openhab.openHAB(self.openhab_url)

    def get_item(self, item_name):
        return self.client.get_item(item_name).state

    def set_item(self, item_name, new_state):
        self.client.get_item(item_name).state = new_state
