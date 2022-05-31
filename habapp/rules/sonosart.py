import requests

import HABApp
from HABApp.openhab.items import StringItem
from HABApp.core.events import ValueChangeEvent

SONOSART_STATICFILE = "/etc/openhab/html/sonosart.png"


class UpdateSonosArt(HABApp.Rule):
    def __init__(self):
        super().__init__()

        self.sonosart_item = StringItem.get_item("SonosKjokkenAlbumArtUrl")

        self.listen_event(self.sonosart_item, self.on_change, ValueChangeEvent)

    def on_change(self, event: ValueChangeEvent) -> None:
        if event.value.startswith("http"):
            try:
                png_bytes = requests.get(event.value).content
                with open(SONOSART_STATICFILE, "wb") as filehandle:
                    filehandle.write(png_bytes)
                print(f"Wrote {len(png_bytes)} PNG bytes to {SONOSART_STATICFILE}")
            except Exception as ex:
                print(f"Could not get URL {event.value}, {ex}")
        else:
            print(f"Not an URL for SonosArt: {event.value}")


UpdateSonosArt()
