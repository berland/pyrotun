import pyrotun

from pyrotun.connections.polar import Polar

logger = pyrotun.getLogger(__name__)


def main():

    polar = Polar()

    polar.get_user_information()
    polar.get_physical_info()
    polar.persist_new_training()


if __name__ == "__main__":
    main()
