import HABApp
import requests
from HABApp.core.events import ValueUpdateEvent, ValueUpdateEventFilter
from HABApp.openhab.items import StringItem

SONOSART_STATICFILE = "/etc/openhab/html/sonosart.png"


class UpdateSonosArt(HABApp.Rule):
    def __init__(self):
        super().__init__()

        self.sonosart_item = StringItem.get_item("SonosKjokkenAlbumArtUrl")
        self.sonosart_item.listen_event(self.on_change, ValueUpdateEventFilter())

    def on_change(self, event: ValueUpdateEvent) -> None:
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
