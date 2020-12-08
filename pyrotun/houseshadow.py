#!/usr/bin/env python
"""
This code is from:
    https://community.openhab.org/t/show-current-sun-position-and-shadow-of-house-generate-svg/34764

slightly adapted for syntax, behaviour, blackening, flake8ing etc.

Depends on env variables::
    LONGITUTE
    LATITUTE
    LOCAL_CITY
    TIMEZONE

"""  # noqa

import os
import math
from datetime import datetime, date, time
import pytz
import argparse

import pylunar
import pandas
import astral
from astral.sun import sun  # noqa
from astral import moon

import pyrotun

logger = pyrotun.getLogger()

WIDTH = 100
HEIGHT = 100
PRIMARY_COLOR = "#1b3024"
LIGHT_COLOR = "#26bf75"
BG_COLOR = "#555555"
SUN_COLOR = "#ffff66"
SUN_DOWN_COLOR = "#888811"
SUN_RADIUS = 8

MOON_COLOR = "#999999"
MOON_RADIUS = 6
STROKE_WIDTH = "1"
FILENAME = "/etc/openhab2/html/husskygge.svg"
LATITUDE = None
LONGITUDE = None

# Shape of the house in a 100 by 100 units square

SHAPE = [
    {"x": 21, "y": 80},
    {"x": 56, "y": 80},
    {"x": 56, "y": 70},
    {"x": 64, "y": 70},
    {"x": 64, "y": 48},
    {"x": 55, "y": 48},
    {"x": 55, "y": 39},
    {"x": 33, "y": 39},  # nordnordvest
    {"x": 33, "y": 61},
    {"x": 21, "y": 61},
    {"x": 21, "y": 80},
]

df = pandas.DataFrame(SHAPE)
df["x"] = df["x"] * 1.5 - 10
df["y"] = df["y"] * 1.5 - 40
SHAPE = df.to_dict(orient="records")

HOURS = 1
DEGS = []


class shadow(object):
    def __init__(self):

        self.debug = False
        timezone = pytz.timezone(os.getenv("TIMEZONE"))

        LATITUDE = float(os.getenv("LOCAL_LATITUDE"))
        LONGITUDE = float(os.getenv("LOCAL_LONGITUDE"))

        assert LATITUDE is not None
        assert LONGITUDE is not None
        assert os.getenv("LOCAL_CITY")
        assert os.getenv("TIMEZONE")

        self.city = astral.LocationInfo(
            "HOME", os.getenv("LOCAL_CITY"), os.getenv("TIMEZOME"), LATITUDE, LONGITUDE
        )
        logger.debug(self.city)
        self.now = timezone.localize(datetime.now())
        self.sun = astral.sun.sun(self.city.observer, self.now)
        logger.debug(self.sun)
        self.sun_azimuth = astral.sun.azimuth(self.city.observer, self.now)
        logger.debug("Sun azimuth: " + str(self.sun_azimuth))
        self.sun_elevation = astral.sun.elevation(self.city.observer, self.now)
        logger.debug("Sun elevation: " + str(self.sun_elevation))

        self.sunrise_azimuth = astral.sun.azimuth(
            self.city.observer, self.sun["sunrise"]
        )
        self.sunset_azimuth = astral.sun.azimuth(self.city.observer, self.sun["sunset"])

        for i in range(0, 24, HOURS):
            a = astral.sun.azimuth(
                self.city.observer,
                timezone.localize(datetime.combine(date.today(), time(i))),
            )
            if a is None:
                a = 0
            DEGS.extend([float(a)])

        self.moon_info = pylunar.MoonInfo(
            self.decdeg2dms(LATITUDE), self.decdeg2dms(LONGITUDE)
        )
        self.moon_info.update(self.now)
        self.moon_azimuth = self.moon_info.azimuth()
        self.moon_elevation = self.moon_info.altitude()

        if self.sun_elevation > 0:
            self.elevation = self.sun_elevation
        else:
            self.elevation = self.moon_elevation

    def decdeg2dms(self, dd):
        negative = dd < 0
        dd = abs(dd)
        minutes, seconds = divmod(dd * 3600, 60)
        degrees, minutes = divmod(minutes, 60)
        if negative:
            if degrees > 0:
                degrees = -degrees
            elif minutes > 0:
                minutes = -minutes
            else:
                seconds = -seconds
        return (degrees, minutes, seconds)

    def generatePath(self, stroke, fill, points, attrs=None):
        p = ""
        p = (
            p
            + '<path stroke="'
            + stroke
            + '" stroke-width="'
            + STROKE_WIDTH
            + '" fill="'
            + fill
            + '" '
        )
        if attrs is not None:
            p = p + " " + attrs + " "
        p = p + ' d="'
        for point in points:
            if points.index(point) == 0:
                p = p + "M" + str(point["x"]) + " " + str(point["y"])
            else:
                p = p + " L" + str(point["x"]) + " " + str(point["y"])
        p = p + '" />'
        return p

    def generateArc(self, dist, stroke, fill, start, end, attrs=None):
        p = ""
        try:
            angle = end - start
            if angle < 0:
                angle = 360 + angle
            p = (
                p
                + '<path d="M'
                + str(self.degreesToPoint(start, dist)["x"])
                + " "
                + str(self.degreesToPoint(start, dist)["y"])
                + " "
            )
            p = p + "A" + str(dist) + " " + str(dist) + " 0 "
            if angle < 180:
                p = p + "0 1 "
            else:
                p = p + "1 1 "
            p = (
                p
                + str(self.degreesToPoint(end, dist)["x"])
                + " "
                + str(self.degreesToPoint(end, dist)["y"])
                + '"'
            )
            p = p + ' stroke="' + stroke + '"'
            if fill is not None:
                p = p + ' fill="' + fill + '" '
            else:
                p = p + ' fill="none" '
            if attrs is not None:
                p = p + " " + attrs + " "
            else:
                p = p + ' stroke-width="' + STROKE_WIDTH + '"'
            p = p + " />"
        except ValueError:
            # (other exceptions should maybe also be caught here?)
            p = ""

        return p

    def degreesToPoint(self, d, r):
        coordinates = {"x": 0, "y": 0}
        cx = WIDTH / 2
        cy = HEIGHT / 2
        d2 = 180 - d
        coordinates["x"] = cx + math.sin(math.radians(d2)) * r
        coordinates["y"] = cy + math.cos(math.radians(d2)) * r

        return coordinates

    def generateSVG(self):
        realSun_pos = self.degreesToPoint(self.sun_azimuth, 10000)

        sun_pos = self.degreesToPoint(self.sun_azimuth, WIDTH / 2)
        moon_pos = self.degreesToPoint(self.moon_azimuth, WIDTH / 2)

        minPoint = -1
        maxPoint = -1

        i = 0

        minAngle = 999
        maxAngle = -999
        for point in SHAPE:
            # Angle of close light source
            angle = -math.degrees(
                math.atan2(point["y"] - sun_pos["y"], point["x"] - sun_pos["x"])
            )
            # Angle of distant light source (e.g. sun_pos)
            angle = -math.degrees(
                math.atan2(point["y"] - realSun_pos["y"], point["x"] - realSun_pos["x"])
            )
            distance = math.sqrt(
                math.pow(sun_pos["y"] - point["y"], 2)
                + math.pow(sun_pos["x"] - point["x"], 2)
            )
            if angle < minAngle:
                minAngle = angle
                minPoint = i
            if angle > maxAngle:
                maxAngle = angle
                maxPoint = i
            point["angle"] = angle
            point["distance"] = distance
            if self.debug:
                print(
                    str(i).ljust(10),
                    ":",
                    str(point["x"]).ljust(10),
                    str(point["y"]).ljust(10),
                    str(round(angle, 7)).ljust(10),
                    str(round(distance)).ljust(10),
                )
            i = i + 1

        if self.debug:
            print("Min Point = ", minPoint)
            print("Max Point = ", maxPoint)
            print("")

        i = minPoint
        k = 0
        side1Distance = 0
        side2Distance = 0
        side1Done = False
        side2Done = False
        side1 = []
        side2 = []
        while True:
            if side1Done is False:
                side1Distance = side1Distance + SHAPE[i]["distance"]
                if i != minPoint and i != maxPoint:
                    SHAPE[i]["side"] = 1
                if i == maxPoint:
                    side1Done = True
                side1.append({"x": SHAPE[i]["x"], "y": SHAPE[i]["y"]})
            if side1Done is True:
                side2Distance = side2Distance + SHAPE[i]["distance"]
                if i != minPoint and i != maxPoint:
                    SHAPE[i]["side"] = 2
                if i == minPoint:
                    side2Done = True
                side2.append({"x": SHAPE[i]["x"], "y": SHAPE[i]["y"]})

            i = i + 1
            if i > len(SHAPE) - 1:
                i = 0

            if side1Done and side2Done:
                break

            k = k + 1
            if k == 20:
                break

        svg = '<?xml version="1.0" encoding="utf-8"?>'
        svg = (
            svg
            + '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
            + '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
        )
        svg = (
            svg
            + '<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" '
            + 'xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" '
            + ' viewBox="-10 -10 120 120" xml:space="preserve">'
        )

        # background
        svg = (
            svg
            + '<circle cx="'
            + str(int(WIDTH / 2))
            + '" cy="'
            + str(int(HEIGHT / 2))
            + '" r="'
            + str(int(WIDTH / 2) - 1)
            + '" fill="'
            + BG_COLOR
            + '"/>'
        )

        minPointShadowX = SHAPE[minPoint]["x"] + WIDTH * math.cos(
            math.radians(minAngle)
        )
        minPointShadowY = SHAPE[minPoint]["y"] - HEIGHT * math.sin(
            math.radians(minAngle)
        )
        maxPointShadowX = SHAPE[maxPoint]["x"] + WIDTH * math.cos(
            math.radians(maxAngle)
        )
        maxPointShadowY = SHAPE[maxPoint]["y"] - HEIGHT * math.sin(
            math.radians(maxAngle)
        )

        shadow = (
            [{"x": maxPointShadowX, "y": maxPointShadowY}]
            + side2
            + [{"x": minPointShadowX, "y": minPointShadowY}]
        )

        svg = svg + '<defs><mask id="shadowMask">'
        svg = svg + '     <rect width="100%" height="100%" fill="black"/>'
        svg = (
            svg
            + '     <circle cx="'
            + str(int(WIDTH / 2))
            + '" cy="'
            + str(int(HEIGHT / 2))
            + '" r="'
            + str(int(WIDTH / 2) - 1)
            + '" fill="white"/>'
        )
        svg = svg + "</mask></defs>"

        svg = svg + self.generatePath("none", PRIMARY_COLOR, SHAPE)

        shadow_svg = self.generatePath(
            "none", "black", shadow, 'mask="url(#shadowMask)" fill-opacity="0.5"'
        )

        if self.elevation > 0:
            svg = svg + self.generatePath(LIGHT_COLOR, "none", side2)
        else:
            svg = svg + self.generatePath(PRIMARY_COLOR, "none", side2)

        if self.elevation > 0:
            svg = svg + shadow_svg

        svg = svg + self.generateArc(
            WIDTH / 2, PRIMARY_COLOR, "none", self.sunset_azimuth, self.sunrise_azimuth
        )
        svg = svg + self.generateArc(
            WIDTH / 2, LIGHT_COLOR, "none", self.sunrise_azimuth, self.sunset_azimuth
        )

        svg = svg + self.generatePath(
            LIGHT_COLOR,
            "none",
            [
                self.degreesToPoint(self.sunrise_azimuth, int(WIDTH / 2) - 2),
                self.degreesToPoint(self.sunrise_azimuth, int(WIDTH / 2) + 2),
            ],
        )
        svg = svg + self.generatePath(
            LIGHT_COLOR,
            "none",
            [
                self.degreesToPoint(self.sunset_azimuth, int(WIDTH / 2) - 2),
                self.degreesToPoint(self.sunset_azimuth, int(WIDTH / 2) + 2),
            ],
        )

        for i in range(0, len(DEGS)):
            if i == len(DEGS) - 1:
                j = 0
            else:
                j = i + 1
            if i % 2 == 0:
                svg = svg + self.generateArc(
                    int(WIDTH / 2) + 8,
                    PRIMARY_COLOR,
                    "none",
                    DEGS[i],
                    DEGS[j],
                    'stroke-width="3" stroke-opacity="0.2"',
                )
            else:
                svg = svg + self.generateArc(
                    int(WIDTH / 2) + 8,
                    PRIMARY_COLOR,
                    "none",
                    DEGS[i],
                    DEGS[j],
                    'stroke-width="3" ',
                )

        svg = svg + self.generatePath(
            LIGHT_COLOR,
            "none",
            [
                self.degreesToPoint(DEGS[0], int(WIDTH / 2) + 5),
                self.degreesToPoint(DEGS[0], int(WIDTH / 2) + 11),
            ],
        )
        svg = svg + self.generatePath(
            LIGHT_COLOR,
            "none",
            [
                self.degreesToPoint(DEGS[int((len(DEGS)) / 2)], int(WIDTH / 2) + 5),
                self.degreesToPoint(DEGS[int((len(DEGS)) / 2)], int(WIDTH / 2) + 11),
            ],
        )

        # moon drawing: compute left and right arcs
        phase = moon.phase(self.now)
        if self.debug:
            print("phase: " + str(phase))
        left_radius = MOON_RADIUS
        left_sweep = 0
        right_radius = MOON_RADIUS
        right_sweep = 0
        if phase > 14:
            right_radius = MOON_RADIUS - (
                2.0 * MOON_RADIUS * (1.0 - ((phase % 14) * 0.99 / 14.0))
            )
            if right_radius < 0:
                right_radius = right_radius * -1.0
                right_sweep = 0
            else:
                right_sweep = 1

        if phase < 14:
            left_radius = MOON_RADIUS - (
                2.0 * MOON_RADIUS * (1.0 - ((phase % 14) * 0.99 / 14.0))
            )
            if left_radius < 0:
                left_radius = left_radius * -1.0
                left_sweep = 1

        if self.moon_elevation > 0:
            svg = (
                svg
                + '<path stroke="none" stroke-width="0" fill="'
                + MOON_COLOR
                + '" d="M '
                + str(moon_pos["x"])
                + " "
                + str(moon_pos["y"] - MOON_RADIUS)
                + " A "
                + str(left_radius)
                + " "
                + str(MOON_RADIUS)
                + " 0 0 "
                + str(left_sweep)
                + " "
                + str(moon_pos["x"])
                + " "
                + str(moon_pos["y"] + MOON_RADIUS)
                + "   "
                + str(right_radius)
                + " "
                + str(MOON_RADIUS)
                + " 0 0 "
                + str(right_sweep)
                + " "
                + str(moon_pos["x"])
                + " "
                + str(moon_pos["y"] - MOON_RADIUS)
                + ' z" />'
            )

        # sun drawing
        if self.sun_elevation > 0:
            svg = (
                svg
                + '<circle cx="'
                + str(sun_pos["x"])
                + '" cy="'
                + str(sun_pos["y"])
                + '" r="'
                + str(SUN_RADIUS)
                + '" stroke="none" stroke-width="0" fill="'
                + SUN_COLOR
                + '55" />'
            )
            svg = (
                svg
                + '<circle cx="'
                + str(sun_pos["x"])
                + '" cy="'
                + str(sun_pos["y"])
                + '" r="'
                + str(SUN_RADIUS - 1)
                + '" stroke="none" stroke-width="0" fill="'
                + SUN_COLOR
                + '99" />'
            )
            svg = (
                svg
                + '<circle cx="'
                + str(sun_pos["x"])
                + '" cy="'
                + str(sun_pos["y"])
                + '" r="'
                + str(SUN_RADIUS - 2)
                + '" stroke="'
                + SUN_COLOR
                + '" stroke-width="0" fill="'
                + SUN_COLOR
                + '" />'
            )
        else:
            svg = (
                svg
                + '<circle cx="'
                + str(sun_pos["x"])
                + '" cy="'
                + str(sun_pos["y"])
                + '" r="'
                + str(SUN_RADIUS / 1.5)
                + '" stroke="none" stroke-width="0" fill="'
                + SUN_DOWN_COLOR
                + '55" />'
            )
            svg = (
                svg
                + '<circle cx="'
                + str(sun_pos["x"])
                + '" cy="'
                + str(sun_pos["y"])
                + '" r="'
                + str(SUN_RADIUS / 1.5 - 1)
                + '" stroke="none" stroke-width="0" fill="'
                + SUN_DOWN_COLOR
                + '99" />'
            )
            svg = (
                svg
                + '<circle cx="'
                + str(sun_pos["x"])
                + '" cy="'
                + str(sun_pos["y"])
                + '" r="'
                + str(SUN_RADIUS / 1.5 - 2)
                + '" stroke="'
                + SUN_DOWN_COLOR
                + '" stroke-width="0" fill="'
                + SUN_DOWN_COLOR
                + '" />'
            )

        svg = svg + "</svg>"

        return svg

async def amain(filename=None):
    svg = shadow().generateSVG()
    with open(filename, "w") as f_handle:
        f_handle.write(svg)
    logger.info("Written SVG to %s", filename)

def main(filename=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--svgfile", help="SVG file to write to")
    args = parser.parse_args()

    s = shadow()
    svg = s.generateSVG()

    if filename is None:
        filename = args.svgfile

    with open(filename, "w") as f_handle:
        f_handle.write(svg)
    logger.info("Written SVG to %s", filename)


if __name__ == "__main__":
    main()
