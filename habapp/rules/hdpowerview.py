import base64
from typing import Dict

import HABApp
import requests
from HABApp.core.events import ValueUpdateEvent, ValueUpdateEventFilter
from HABApp.openhab.items import StringItem

HUB = "hdpowerview.r40"
HUB = "192.168.1.84"


def _get_scenes(old_scenes) -> Dict[str, dict]:
    response = requests.get(f"http://{HUB}/api/scenes")
    if response.ok:
        alldata = response.json()
        scenelist: list = alldata.get("sceneData", [])
        scenedict = {}
        for scene in scenelist:
            scenename = base64.b64decode(scene["name"]).decode("utf-8")
            scenedict[scenename] = scene
        return scenedict
    return old_scenes


class ActivatePowerviewScene(HABApp.Rule):
    def __init__(self):
        super().__init__()

        self.scene_item = StringItem.get_item("PowerviewScene")
        self.scene_item.listen_event(self.on_change, ValueUpdateEventFilter())

        self.scenes = []
        self.scenes = _get_scenes(self.scenes)
        print(self.scenes.keys())

    def on_change(self, event: ValueUpdateEvent) -> None:
        requested_scene: str = event.value
        if requested_scene in self.scenes:
            self.activate_scene(self.scenes[requested_scene])
        else:
            print(
                f"Unknown scene {requested_scene}, "
                f"must be among {self.scenes.keys()}"
            )

    def activate_scene(self, scene: dict) -> None:
        print(f"running scene {scene}")
        url = f"http://{HUB}/api/scenes?sceneId={scene['id']}"
        print(f"queryign url {url}")
        response = requests.get(url)
        print(f"ok from powerview? : {response.ok}")


ActivatePowerviewScene()
