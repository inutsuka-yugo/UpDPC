from pycromanager import (
    Acquisition,
    Core,
    multi_d_acquisition_events,
    Studio,
    XYTiledAcquisition,
)
from time import sleep
from decimal import Decimal, ROUND_HALF_UP


def round_to_2dp(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def round_to_1dp(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


mmc = Core()
studio = Studio()
pm = studio.positions()

z_stage = mmc.get_focus_device()


def getXY():
    """Return: xnow, ynow"""
    xnow = mmc.get_x_position()
    ynow = mmc.get_y_position()
    return round_to_1dp(xnow), round_to_1dp(ynow)


def getZ():
    return round_to_2dp(mmc.get_position(z_stage))


def moveXY(x, y):
    x = round_to_1dp(x)
    y = round_to_1dp(y)
    mmc.set_xy_position(x, y)
    while getXY() != (x, y):
        sleep(0.1)
        mmc.set_xy_position(x, y)
    print("(X, Y) moved to", (x, y))


def moveZ_small(z):
    z = round_to_2dp(z)
    mmc.set_position(z_stage, z)
    for _ in range(10):
        if -0.01 <= z - getZ() <= 0.01:
            # return print("[Small step] Z moved to", z)
            return 1
        sleep(0.1)
        # mmc.set_position(z_stage, z)
    return print("[Small step] Timeout, Z moved to", z)


def moveZ(z, step=0.25):
    sleep(0.1)
    znow = getZ()
    if -0.01 <= z - znow <= 0.01:
        return print("Z moved to", z)
    elif z > znow:
        moveZ_small(znow + min(step, z - znow))
        moveZ(z, step)
    else:
        moveZ_small(znow - min(step, znow - z))
        moveZ(z, step)


def set_channel(channel="BF"):
    mmc.set_config("Pycro", channel)
    sleep(2)
