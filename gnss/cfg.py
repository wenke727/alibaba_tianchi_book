from pathlib import Path

DATA_FOLDER = "../data/trajectory_sensors"

GNSS_TYPE_DICT = {
    0: "UNKNOWN",
    1: "GPS",
    2: "SBAS",
    3: "GLONASS",
    4: "QZSS",
    5: "BEIDOU",
    6: "GALILEO",
    7: "IRNSS",
}

KEYWORD_2_COLUMN = {
    'light': ["values"], 
    'mobile': ['cid'],
}

LABEL = "label"