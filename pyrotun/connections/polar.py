import os
import pathlib

from lxml import etree
import json
import dotenv

from .accesslink import AccessLink

import pyrotun

logger = pyrotun.getLogger(__name__)

dotenv.load_dotenv()


def pretty_print_json(data):
    return json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)


def pretty_print_xml(xmltext):
    tree = etree.fromstring(bytes(bytearray(xmltext, encoding="utf-8")))
    return etree.tostring(tree, pretty_print=True).decode()


class Polar(object):
    """Personal application for getting
    last exercise from  Polar API (v3)"""

    def __init__(self):
        self.config = {}
        self.config["client_id"] = os.getenv("POLAR_CLIENT_ID")
        self.config["client_secret"] = os.getenv("POLAR_CLIENT_SECRET")

        # These are obtained through "autorization.py"
        self.config["user_id"] = os.getenv("POLAR_USER_ID")
        self.config["access_token"] = os.getenv("POLAR_ACCESS_TOKEN")

        self.accesslink = AccessLink(
            client_id=self.config["client_id"],
            client_secret=self.config["client_secret"],
        )

    def persist_new_training(self):
        transaction = self.accesslink.training_data.create_transaction(
            user_id=self.config["user_id"], access_token=self.config["access_token"]
        )

        if not transaction:
            logger.info("No new exercises available. Go running.")
            return

        resource_urls = transaction.list_exercises()["exercises"]

        for url in resource_urls:
            exercise_summary = transaction.get_exercise_summary(url)
            start_time = exercise_summary["start-time"]

            exportdir = pathlib.Path.home() / "polar_dump" / str(start_time)
            exportdir.mkdir(parents=True, exist_ok=True)

            heart_rate_zones = transaction.get_heart_rate_zones(url)
            gpx = transaction.get_gpx(url)
            tcx = transaction.get_tcx(url)

            (exportdir / "exercise_summary").write_text(
                pretty_print_json(exercise_summary)
            )
            logger.info("Someone has been exercising! Persisting to disk")
            logger.info(pretty_print_json(exercise_summary))
            (exportdir / "heart_rate_zones").write_text(
                pretty_print_json(heart_rate_zones)
            )

            if gpx:
                (exportdir / "gpx").write_text(pretty_print_xml(gpx))

            if tcx:
                (exportdir / "tcx").write_text(pretty_print_xml(tcx))

            logger.info("Dumped data in " + str(exportdir))
        transaction.commit()

    def get_physical_info(self):
        transaction = self.accesslink.physical_info.create_transaction(
            user_id=self.config["user_id"], access_token=self.config["access_token"]
        )
        if not transaction:
            logger.info("No new physical information available.")
            return

        resource_urls = transaction.list_physical_infos()["physical-informations"]

        for url in resource_urls:
            physical_info = transaction.get_physical_info(url)

            print("Physical info:")
            print(physical_info)

            # pretty_print_json(physical_info)

        transaction.commit()

    def get_user_information(self):
        user_info = self.accesslink.users.get_information(
            user_id=self.config["user_id"], access_token=self.config["access_token"]
        )
        pretty_print_json(user_info)
