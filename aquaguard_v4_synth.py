"""
AquaGuard v4 - synthetic Jordan water-network data generator (train + test)
===========================================================================
Reproduces the AquaGuard-Model repo files EXACTLY (byte-identical CSVs):
    jordan_v4_train.csv  150,005 rows x 44 cols   sha256 7eebeef1...6307
    jordan_v4_test.csv    30,020 rows x 44 cols   sha256 550322a1...ff500

PROVENANCE: this is the generator recovered from the original AquaGuard-AI
conversation (generate_v4_dataset.py + its two recorded sed patches).
Sections 1-5 below (network, physics, demand, anomalies, generate_branch) are
unchanged; only the validation suite and progress printing were removed and the
run/verify wrapper added.  It is NOT a new model fitted to the numbers.

PHYSICS (all in darcy_dp_psi):
    Q  = flow[L/min] / 60000                      [m^3/s]
    v  = Q / (pi D^2 / 4),  D = 0.05 m (fixed)    [m/s]
    Re = v D / nu,  nu = 1.004e-6 m^2/s
    eps/D = 0.26e-3 / D                           (cast-iron roughness)
    f  = Churchill (1977) Darcy friction factor
    dh = f (L/D) v^2 / (2g),  2g = 19.62          [m of water]
    dP = 1.422 * dh                               [PSI], clipped to [0.01, 60]
The Hazen-Williams coefficient is stored as a column but does NOT enter dP.

STATISTICS: every random draw comes from numpy Generator(PCG64):
    train -> default_rng(2024), test -> default_rng(2025); the draw ORDER
    (eid, supply, inlet flow, inlet pressure, then per-segment noises) matters.
    Needs numpy>=1.17.  Last verified: numpy 2.5.3, pandas 3.0.5, Python 3.13.

USAGE:
    python3 aquaguard_v4_synth.py --out ./data            # generate
    python3 aquaguard_v4_synth.py --out ./data --verify   # + check sha256
"""

import argparse, hashlib, math, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

SEED = 2024
EXPECTED_SHA256 = {
    "jordan_v4_train.csv": "7eebeef1c1aab45953a771812a29642d054b9c89630d9e6b0f8c99615cce6307",
    "jordan_v4_test.csv":  "550322a1011670f3b56bb88ca94b2488b8c4df495ea60a3e38e60d765e2ff500",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  JORDAN WATER NETWORK DEFINITION
#     12 governorates — each with an independent water source.
#     31 branches — each a linear chain of 3–4 consecutive checkpoint segments.
#     95 total segments.
# ══════════════════════════════════════════════════════════════════════════════

NETWORK = {

    "Amman": {
        "source":         "King-Abdullah-Canal / Zai-WTP",
        "main_node":      "AMM-Main",
        "nrw_rate":        0.43,
        "flow_range":     (80, 1400),    # L/min at main meter
        "pressure_range": (60, 125),     # PSI at main meter
        "env_code":        0,
        "pop_density":     5540,
        "elevation_m":     780,
        "branches": {
            "Amman-East": {
                "split": 0.26,
                "segs": [
                    {"from":"AMM-Main",   "to":"Marka",        "len":820, "age":22, "hw":125, "bg":0.004},
                    {"from":"Marka",      "to":"Abu-Nsair",    "len":650, "age":28, "hw":118, "bg":0.005},
                    {"from":"Abu-Nsair",  "to":"Hashmi",       "len":480, "age":35, "hw":108, "bg":0.007},
                    {"from":"Hashmi",     "to":"Basman",       "len":390, "age":40, "hw":100, "bg":0.008},
                ],
            },
            "Amman-West": {
                "split": 0.28,
                "segs": [
                    {"from":"AMM-Main",   "to":"Abdoun",       "len":750, "age":15, "hw":140, "bg":0.003},
                    {"from":"Abdoun",     "to":"Sweifieh",     "len":620, "age":18, "hw":136, "bg":0.003},
                    {"from":"Sweifieh",   "to":"Jubaiha",      "len":530, "age":22, "hw":128, "bg":0.004},
                    {"from":"Jubaiha",    "to":"Khilda",       "len":460, "age":20, "hw":132, "bg":0.004},
                ],
            },
            "Amman-North": {
                "split": 0.22,
                "segs": [
                    {"from":"AMM-Main",   "to":"Shmeisani",    "len":680, "age":20, "hw":132, "bg":0.003},
                    {"from":"Shmeisani",  "to":"Rabieh",       "len":710, "age":25, "hw":126, "bg":0.005},
                    {"from":"Rabieh",     "to":"Tla-Al-Ali",   "len":590, "age":18, "hw":138, "bg":0.003},
                ],
            },
            "Amman-South": {
                "split": 0.15,
                "segs": [
                    {"from":"AMM-Main",   "to":"Sahab",        "len":900, "age":30, "hw":115, "bg":0.006},
                    {"from":"Sahab",      "to":"Yadoudeh",     "len":800, "age":35, "hw":108, "bg":0.007},
                    {"from":"Yadoudeh",   "to":"Muwaqqar",     "len":720, "age":40, "hw":102, "bg":0.008},
                ],
            },
            "Amman-Central": {
                "split": 0.09,
                "segs": [
                    {"from":"AMM-Main",   "to":"Gardens",      "len":600, "age":12, "hw":145, "bg":0.002},
                    {"from":"Gardens",    "to":"Um-Uthaina",   "len":550, "age":15, "hw":141, "bg":0.002},
                    {"from":"Um-Uthaina", "to":"Khalda",       "len":490, "age":18, "hw":138, "bg":0.003},
                ],
            },
        },
    },

    "Irbid": {
        "source":         "Yarmouk-River-WTP",
        "main_node":      "IRB-Main",
        "nrw_rate":        0.45,
        "flow_range":     (60, 800),
        "pressure_range": (45, 100),
        "env_code":        0,
        "pop_density":     3280,
        "elevation_m":     620,
        "branches": {
            "Irbid-Central": {
                "split": 0.35,
                "segs": [
                    {"from":"IRB-Main",   "to":"Husn",         "len":700, "age":20, "hw":130, "bg":0.004},
                    {"from":"Husn",       "to":"Manara",       "len":620, "age":25, "hw":122, "bg":0.005},
                    {"from":"Manara",     "to":"Ramtha",       "len":550, "age":30, "hw":115, "bg":0.006},
                ],
            },
            "Irbid-North": {
                "split": 0.28,
                "segs": [
                    {"from":"IRB-Main",   "to":"Al-Huson",     "len":680, "age":18, "hw":135, "bg":0.003},
                    {"from":"Al-Huson",   "to":"Bait-Ras",     "len":590, "age":22, "hw":128, "bg":0.004},
                    {"from":"Bait-Ras",   "to":"Kufr-Asad",   "len":510, "age":28, "hw":120, "bg":0.005},
                ],
            },
            "Irbid-West": {
                "split": 0.22,
                "segs": [
                    {"from":"IRB-Main",   "to":"Koura",        "len":850, "age":35, "hw":108, "bg":0.007},
                    {"from":"Koura",      "to":"Kufrinja",     "len":780, "age":40, "hw":100, "bg":0.008},
                    {"from":"Kufrinja",   "to":"Deir-Abi-Said","len":650, "age":45, "hw":95,  "bg":0.010},
                ],
            },
            "Irbid-South": {
                "split": 0.15,
                "segs": [
                    {"from":"IRB-Main",   "to":"Aydoun",       "len":760, "age":28, "hw":118, "bg":0.006},
                    {"from":"Aydoun",     "to":"Bani-Kinana",  "len":680, "age":32, "hw":112, "bg":0.006},
                    {"from":"Bani-Kinana","to":"Tibne",        "len":600, "age":38, "hw":106, "bg":0.007},
                ],
            },
        },
    },

    "Zarqa": {
        "source":         "Zarqa-Municipal-Reservoirs",
        "main_node":      "ZRQ-Main",
        "nrw_rate":        0.52,
        "flow_range":     (50, 620),
        "pressure_range": (35, 90),
        "env_code":        1,
        "pop_density":     4120,
        "elevation_m":     580,
        "branches": {
            "Zarqa-Old-City": {
                "split": 0.42,
                "segs": [
                    {"from":"ZRQ-Main",    "to":"Zarqa-Center", "len":920, "age":42, "hw":96,  "bg":0.010},
                    {"from":"Zarqa-Center","to":"Rusaifa",       "len":850, "age":48, "hw":90,  "bg":0.012},
                    {"from":"Rusaifa",     "to":"New-Zarqa",     "len":780, "age":52, "hw":86,  "bg":0.013},
                ],
            },
            "Zarqa-Industrial": {
                "split": 0.35,
                "segs": [
                    {"from":"ZRQ-Main",    "to":"Ind-Zone-ZRQ", "len":800, "age":30, "hw":112, "bg":0.007},
                    {"from":"Ind-Zone-ZRQ","to":"Hashimiyya",    "len":720, "age":35, "hw":106, "bg":0.008},
                    {"from":"Hashimiyya",  "to":"Dhiban-Ind",    "len":650, "age":38, "hw":101, "bg":0.008},
                ],
            },
            "Zarqa-East": {
                "split": 0.23,
                "segs": [
                    {"from":"ZRQ-Main",    "to":"Azraq",         "len":1200,"age":20, "hw":138, "bg":0.005},
                    {"from":"Azraq",       "to":"Safawi",         "len":1100,"age":15, "hw":142, "bg":0.004},
                    {"from":"Safawi",      "to":"Ruwaished",      "len":900, "age":12, "hw":146, "bg":0.003},
                ],
            },
        },
    },

    "Balqa": {
        "source":         "Balqa-WAJ-Supply",
        "main_node":      "BLQ-Main",
        "nrw_rate":        0.47,
        "flow_range":     (30, 480),
        "pressure_range": (38, 95),
        "env_code":        2,
        "pop_density":     720,
        "elevation_m":     850,
        "branches": {
            "Salt-District": {
                "split": 0.42,
                "segs": [
                    {"from":"BLQ-Main",    "to":"Salt-City",     "len":720, "age":25, "hw":125, "bg":0.005},
                    {"from":"Salt-City",   "to":"Shuna-North",   "len":650, "age":30, "hw":118, "bg":0.006},
                    {"from":"Shuna-North", "to":"Kafrein",       "len":580, "age":35, "hw":111, "bg":0.007},
                ],
            },
            "Fuheis-Area": {
                "split": 0.35,
                "segs": [
                    {"from":"BLQ-Main",    "to":"Fuheis",        "len":620, "age":18, "hw":138, "bg":0.003},
                    {"from":"Fuheis",      "to":"Mahis",         "len":560, "age":22, "hw":132, "bg":0.004},
                    {"from":"Mahis",       "to":"Zay",           "len":500, "age":28, "hw":124, "bg":0.005},
                ],
            },
            "Jordan-Valley": {
                "split": 0.23,
                "segs": [
                    {"from":"BLQ-Main",    "to":"Shunah-South",  "len":900, "age":20, "hw":130, "bg":0.005},
                    {"from":"Shunah-South","to":"Rama",          "len":820, "age":25, "hw":124, "bg":0.006},
                    {"from":"Rama",        "to":"Karameh",       "len":740, "age":30, "hw":117, "bg":0.007},
                ],
            },
        },
    },

    "Karak": {
        "source":         "Karak-WAJ-Supply",
        "main_node":      "KRK-Main",
        "nrw_rate":        0.56,
        "flow_range":     (15, 220),
        "pressure_range": (25, 72),
        "env_code":        3,
        "pop_density":     85,
        "elevation_m":     930,
        "branches": {
            "Karak-Central": {
                "split": 0.60,
                "segs": [
                    {"from":"KRK-Main",    "to":"Karak-City",   "len":950, "age":45, "hw":90,  "bg":0.011},
                    {"from":"Karak-City",  "to":"Mazar",        "len":880, "age":50, "hw":85,  "bg":0.013},
                    {"from":"Mazar",       "to":"Mutah",        "len":800, "age":55, "hw":80,  "bg":0.014},
                ],
            },
            "Karak-North": {
                "split": 0.40,
                "segs": [
                    {"from":"KRK-Main",    "to":"Al-Qasr",      "len":820, "age":38, "hw":98,  "bg":0.009},
                    {"from":"Al-Qasr",     "to":"Safi",         "len":760, "age":42, "hw":93,  "bg":0.010},
                    {"from":"Safi",        "to":"Al-Lajjun",    "len":680, "age":48, "hw":87,  "bg":0.012},
                ],
            },
        },
    },

    "Mafraq": {
        "source":         "Mafraq-Groundwater-Wells",
        "main_node":      "MFQ-Main",
        "nrw_rate":        0.50,
        "flow_range":     (20, 320),
        "pressure_range": (28, 78),
        "env_code":        4,
        "pop_density":     13,
        "elevation_m":     690,
        "branches": {
            "Mafraq-Central": {
                "split": 0.55,
                "segs": [
                    {"from":"MFQ-Main",    "to":"Mafraq-City",  "len":1100,"age":15, "hw":140, "bg":0.004},
                    {"from":"Mafraq-City", "to":"Rhab",         "len":980, "age":18, "hw":136, "bg":0.004},
                    {"from":"Rhab",        "to":"Umm-Al-Jimal", "len":850, "age":22, "hw":130, "bg":0.005},
                ],
            },
            "Mafraq-North": {
                "split": 0.45,
                "segs": [
                    {"from":"MFQ-Main",    "to":"Safawi-Town",  "len":1300,"age":12, "hw":145, "bg":0.003},
                    {"from":"Safawi-Town", "to":"Ruwaished-N",  "len":1200,"age":15, "hw":142, "bg":0.003},
                    {"from":"Ruwaished-N", "to":"Azraq-North",  "len":1100,"age":18, "hw":138, "bg":0.004},
                ],
            },
        },
    },

    "Aqaba": {
        "source":         "Aqaba-Desalination-Plant",
        "main_node":      "AQB-Main",
        "nrw_rate":        0.41,
        "flow_range":     (25, 380),
        "pressure_range": (35, 92),
        "env_code":        5,
        "pop_density":     310,
        "elevation_m":     10,
        "branches": {
            "Aqaba-Port": {
                "split": 0.55,
                "segs": [
                    {"from":"AQB-Main",    "to":"Aqaba-Port",   "len":700, "age":20, "hw":132, "bg":0.004},
                    {"from":"Aqaba-Port",  "to":"Ind-Zone-AQB", "len":650, "age":25, "hw":126, "bg":0.005},
                    {"from":"Ind-Zone-AQB","to":"South-Aqaba",  "len":580, "age":18, "hw":138, "bg":0.004},
                ],
            },
            "Aqaba-Residential": {
                "split": 0.45,
                "segs": [
                    {"from":"AQB-Main",    "to":"Aqaba-Center", "len":620, "age":15, "hw":140, "bg":0.003},
                    {"from":"Aqaba-Center","to":"Quweira",      "len":580, "age":20, "hw":134, "bg":0.004},
                    {"from":"Quweira",     "to":"Wadi-Rum-Twn", "len":520, "age":25, "hw":128, "bg":0.005},
                ],
            },
        },
    },

    "Madaba": {
        "source":         "Madaba-WAJ-Supply",
        "main_node":      "MDB-Main",
        "nrw_rate":        0.51,
        "flow_range":     (15, 280),
        "pressure_range": (30, 85),
        "env_code":        3,
        "pop_density":     200,
        "elevation_m":     760,
        "branches": {
            "Madaba-Center": {
                "split": 0.60,
                "segs": [
                    {"from":"MDB-Main",    "to":"Madaba-City",  "len":680, "age":22, "hw":128, "bg":0.005},
                    {"from":"Madaba-City", "to":"Madaba-East",  "len":620, "age":28, "hw":120, "bg":0.006},
                    {"from":"Madaba-East", "to":"Dhiban-North", "len":550, "age":35, "hw":112, "bg":0.007},
                ],
            },
            "Mount-Nebo": {
                "split": 0.40,
                "segs": [
                    {"from":"MDB-Main",    "to":"Mount-Nebo",   "len":750, "age":18, "hw":136, "bg":0.004},
                    {"from":"Mount-Nebo",  "to":"Libb",         "len":680, "age":22, "hw":130, "bg":0.005},
                    {"from":"Libb",        "to":"Nitl",         "len":610, "age":28, "hw":122, "bg":0.006},
                ],
            },
        },
    },

    "Jerash": {
        "source":         "Jerash-WAJ-Supply",
        "main_node":      "JRS-Main",
        "nrw_rate":        0.48,
        "flow_range":     (12, 250),
        "pressure_range": (30, 82),
        "env_code":        2,
        "pop_density":     585,
        "elevation_m":     510,
        "branches": {
            "Jerash-City": {
                "split": 0.55,
                "segs": [
                    {"from":"JRS-Main",    "to":"Jerash-North", "len":650, "age":20, "hw":130, "bg":0.005},
                    {"from":"Jerash-North","to":"Sakib",        "len":580, "age":25, "hw":124, "bg":0.006},
                    {"from":"Sakib",       "to":"Kufr-Khal",   "len":520, "age":30, "hw":118, "bg":0.006},
                ],
            },
            "Jerash-South": {
                "split": 0.45,
                "segs": [
                    {"from":"JRS-Main",    "to":"Burma",        "len":720, "age":28, "hw":118, "bg":0.006},
                    {"from":"Burma",       "to":"Rihaba",       "len":660, "age":32, "hw":113, "bg":0.007},
                    {"from":"Rihaba",      "to":"Mujib-Jerash", "len":580, "age":38, "hw":106, "bg":0.008},
                ],
            },
        },
    },

    "Ajloun": {
        "source":         "Ajloun-WAJ-Supply",
        "main_node":      "AJL-Main",
        "nrw_rate":        0.54,
        "flow_range":     (10, 200),
        "pressure_range": (25, 75),
        "env_code":        3,
        "pop_density":     392,
        "elevation_m":     1050,
        "branches": {
            "Ajloun-Central": {
                "split": 0.55,
                "segs": [
                    {"from":"AJL-Main",    "to":"Anjara",       "len":680, "age":38, "hw":100, "bg":0.009},
                    {"from":"Anjara",      "to":"Ibbin",        "len":620, "age":42, "hw":95,  "bg":0.010},
                    {"from":"Ibbin",       "to":"Orjan",        "len":550, "age":48, "hw":90,  "bg":0.012},
                ],
            },
            "Ajloun-North": {
                "split": 0.45,
                "segs": [
                    {"from":"AJL-Main",    "to":"Kufranja",     "len":720, "age":42, "hw":96,  "bg":0.010},
                    {"from":"Kufranja",    "to":"Rajib",        "len":660, "age":45, "hw":92,  "bg":0.011},
                    {"from":"Rajib",       "to":"Al-Husun-AJL", "len":590, "age":50, "hw":87,  "bg":0.012},
                ],
            },
        },
    },

    "Tafilah": {
        "source":         "Tafilah-WAJ-Supply",
        "main_node":      "TFL-Main",
        "nrw_rate":        0.53,
        "flow_range":     (8, 150),
        "pressure_range": (22, 70),
        "env_code":        3,
        "pop_density":     48,
        "elevation_m":     1100,
        "branches": {
            "Tafilah-Main": {
                "split": 0.60,
                "segs": [
                    {"from":"TFL-Main",    "to":"Tafilah-City", "len":750, "age":25, "hw":122, "bg":0.006},
                    {"from":"Tafilah-City","to":"Busayra",      "len":680, "age":30, "hw":116, "bg":0.007},
                    {"from":"Busayra",     "to":"Ghrandal",     "len":600, "age":35, "hw":110, "bg":0.008},
                ],
            },
            "Tafilah-South": {
                "split": 0.40,
                "segs": [
                    {"from":"TFL-Main",    "to":"Ain-Al-Baida", "len":820, "age":28, "hw":116, "bg":0.007},
                    {"from":"Ain-Al-Baida","to":"Qadisiyya",    "len":750, "age":32, "hw":110, "bg":0.008},
                    {"from":"Qadisiyya",   "to":"Shawbak-Lnk",  "len":670, "age":38, "hw":104, "bg":0.009},
                ],
            },
        },
    },

    "Maan": {
        "source":         "Maan-Groundwater-Wells",
        "main_node":      "MAN-Main",
        "nrw_rate":        0.55,
        "flow_range":     (8, 170),
        "pressure_range": (20, 68),
        "env_code":        4,
        "pop_density":     4,
        "elevation_m":     1070,
        "branches": {
            "Maan-City": {
                "split": 0.60,
                "segs": [
                    {"from":"MAN-Main",    "to":"Maan-Center",  "len":950, "age":18, "hw":138, "bg":0.005},
                    {"from":"Maan-Center", "to":"Qatraneh",     "len":880, "age":22, "hw":132, "bg":0.006},
                    {"from":"Qatraneh",    "to":"Jafr",         "len":800, "age":28, "hw":125, "bg":0.007},
                ],
            },
            "Maan-South": {
                "split": 0.40,
                "segs": [
                    {"from":"MAN-Main",    "to":"Wadi-Musa",    "len":1050,"age":22, "hw":132, "bg":0.006},
                    {"from":"Wadi-Musa",   "to":"Shobak",       "len":980, "age":28, "hw":125, "bg":0.007},
                    {"from":"Shobak",      "to":"Ras-Al-Naqab", "len":900, "age":35, "hw":117, "bg":0.008},
                ],
            },
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def darcy_dp_psi(flow_lpm: float, length_m: float, hw: float,
                 diam_m: float = 0.05) -> float:
    """
    Pressure drop (PSI) using Darcy-Weisbach + Churchill (1977) friction factor.
    Returns a value clamped to [0.01, 60] PSI.
    """
    Q   = max(flow_lpm, 0.1) / 60_000.0        # m³/s
    A   = math.pi * (diam_m / 2) ** 2           # cross-section m²
    v   = Q / A                                  # velocity m/s
    Re  = max(v * diam_m / 1.004e-6, 1.0)       # Reynolds number
    eps = 0.26e-3 / diam_m                       # relative roughness (cast-iron)
    # Churchill (1977) Darcy friction factor — valid all Re regimes
    Ac  = (-2.457 * math.log((7.0/Re)**0.9 + 0.27*eps)) ** 16
    Bc  = (37_530.0 / Re) ** 16
    f   = 8.0 * ((8.0/Re)**12 + (Ac + Bc)**(-1.5)) ** (1.0/12.0)
    # Darcy-Weisbach head loss → PSI  (1 m H₂O = 1.422 PSI)
    dh  = f * (length_m / diam_m) * (v**2 / 19.62)   # metres of head
    dp  = dh * 1.422                                   # PSI
    return float(np.clip(dp, 0.01, 60.0))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DEMAND MODEL  (Jordan-specific)
# ══════════════════════════════════════════════════════════════════════════════

RAMADAN_MONTHS  = {3}       # approximate
EID_PROB_DAILY  = 10 / 365  # ~10 Eid days per year

def demand_factor(hour: int, dow: int, month: int,
                  is_ramadan: int, is_eid: int) -> float:
    """Combined demand multiplier relative to annual average."""
    # Diurnal pattern
    if   7  <= hour <= 9:   base = 1.42
    elif 18 <= hour <= 21:  base = 1.28
    elif hour < 5:           base = 0.38
    else:                    base = 1.00
    # Seasonal
    if   month in {6,7,8}:  base *= 1.38
    elif month in {12,1,2}: base *= 0.87
    # Jordan weekend (Thu–Fri)
    if   dow in {3,4}:      base *= 1.12
    # Islamic calendar
    if   is_eid:             base *= 1.42
    elif is_ramadan and hour in {3,4,5,20,21,22}:
        base *= 1.60    # Suhoor / Iftar peak
    elif is_ramadan and 9 <= hour <= 14:
        base *= 0.62    # midday fasting lull
    return float(np.clip(base, 0.35, 2.10))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ANOMALY INJECTION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_CFG = {
    "leak": {
        "severity": (0.04, 0.18),       # fraction of flow_in lost
        "pf":       (0.35, 0.75),       # pressure factor
        "label":    1,
    },
    "burst": {
        "severity": (0.25, 0.72),
        "pf":       (0.70, 0.92),
        "label":    2,
    },
    "theft": {
        "severity": (0.05, 0.14),
        "pf":       (0.04, 0.16),       # small orifice → minimal pressure loss
        "label":    3,
    },
}

ANOMALY_WEIGHTS = [0.45, 0.25, 0.30]   # leak, burst, theft


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BRANCH SCENARIO GENERATOR  (core function)
# ══════════════════════════════════════════════════════════════════════════════

def generate_branch(
    ts:          datetime,
    gov_name:    str,
    gov:         dict,
    branch_name: str,
    branch:      dict,
    fault_idx:   int,         # segment index with fault (-1 = normal)
    fault_type:  str | None,  # 'leak' / 'burst' / 'theft' / None
    rng:         np.random.Generator,
) -> list[dict]:
    """
    Return one row per segment in the branch.
    Flow propagates exactly: row[k].Flow_Out == row[k+1].Flow_In (no noise gap).
    """
    h, dow, mon = ts.hour, ts.weekday(), ts.month
    yr          = ts.year

    is_ramadan = int(mon in RAMADAN_MONTHS)
    is_eid     = int(rng.random() < EID_PROB_DAILY)
    is_summer  = int(mon in {6,7,8})
    is_winter  = int(mon in {12,1,2})
    is_weekend = int(dow in {3,4})
    supply_on  = int(rng.random() > 0.20)   # 80 % uptime

    dm = demand_factor(h, dow, mon, is_ramadan, is_eid)

    # Branch inlet flow
    flo, fhi   = gov["flow_range"]
    p_lo, p_hi = gov["pressure_range"]
    inlet_flow = float(
        rng.uniform(flo, fhi) * dm * branch["split"]
    )
    inlet_flow = max(inlet_flow, 1.5)
    if not supply_on:
        inlet_flow *= 0.14

    inlet_pressure = float(rng.uniform(p_lo, p_hi))

    rows       = []
    flow_in    = inlet_flow
    press_in   = inlet_pressure
    segs       = branch["segs"]
    n_segs     = len(segs)

    for si, seg in enumerate(segs):
        is_fault = (si == fault_idx and fault_type is not None)

        # Background (normal) loss — age-adjusted
        age_factor = 1.0 + max(seg["age"] - 20, 0) / 80.0
        bg_rate    = min(seg["bg"] * age_factor, 0.025)  # cap at 2.5 %

        # Predicted pressure drop (Darcy-Weisbach)
        dp_pred = darcy_dp_psi(flow_in, seg["len"], seg["hw"])

        # ── Anomaly or normal ──────────────────────────────────────────────
        if is_fault:
            cfg      = ANOMALY_CFG[fault_type]
            severity = float(rng.uniform(*cfg["severity"]))
            pf       = float(rng.uniform(*cfg["pf"]))

            # Flow: guaranteed < flow_in
            target   = flow_in * (1.0 - severity)
            noise_f  = float(rng.normal(0, max(0.008 * flow_in, 0.02)))
            flow_out = float(np.clip(
                target + noise_f,
                flow_in * 0.05,                          # at least 5 % remains
                flow_in * (1.0 - severity * 0.70) - 0.01 # well below target
            ))

            extra_dp   = severity * press_in * pf
            noise_p    = float(rng.normal(0, 0.25))
            press_out  = max(min(press_in - dp_pred - extra_dp + noise_p, press_in - 0.01), 0.5)

            a_type   = fault_type
            a_label  = cfg["label"]
            a_binary = 1
            fault_here = 1
        else:
            severity = 0.0
            # Normal loss: proportional noise
            noise_f  = float(rng.normal(0, max(0.008 * flow_in, 0.02)))
            flow_out = float(np.clip(
                flow_in * (1.0 - bg_rate) + noise_f,
                flow_in * 0.87,    # floor: background never > 13 %
                flow_in * 0.9999   # must not exceed inlet
            ))

            noise_p   = float(rng.normal(0, 0.25))
            press_out = max(min(press_in - dp_pred + noise_p, press_in - 0.005), 0.5)

            a_type   = "normal"
            a_label  = 0
            a_binary = 0
            fault_here = 0

        # ── Derived features ───────────────────────────────────────────────
        flow_loss     = max(flow_in - flow_out, 0.0)
        flow_loss_pct = flow_loss / max(flow_in, 1e-6) * 100.0
        excess_loss   = max(flow_loss_pct - bg_rate * 100.0, 0.0)
        dp_actual     = press_in - press_out
        dp_dev        = dp_actual - dp_pred
        flow_dp_ratio = flow_loss / max(abs(dp_actual), 0.10)
        velocity_prx  = flow_in / max(seg["len"] * 0.01, 0.01)

        rows.append({
            # ── Identifiers
            "Timestamp":          ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "Governorate":        gov_name,
            "Gov_Source":         gov["source"],
            "Branch":             branch_name,
            "Segment_From":       seg["from"],
            "Segment_To":         seg["to"],
            "Segment_ID":         f"{seg['from']}→{seg['to']}",
            # ── Pipe structure
            "Depth_Level":        si + 1,
            "Is_End_Segment":     int(si == n_segs - 1),
            "Pipe_Length_m":      seg["len"],
            "Pipe_Age_Years":     seg["age"],
            "HW_Coefficient":     seg["hw"],
            # ── Time / context
            "hour":               h,
            "day_of_week":        dow,
            "month":              mon,
            "year":               yr,
            "is_ramadan":         is_ramadan,
            "is_eid":             is_eid,
            "is_summer":          is_summer,
            "is_winter":          is_winter,
            "is_weekend":         is_weekend,
            "supply_on":          supply_on,
            # ── Geography
            "environment_code":   gov["env_code"],
            "population_density": gov["pop_density"],
            "nrw_rate":           gov["nrw_rate"],
            "elevation_m":        gov["elevation_m"],
            # ── Flow (PROPAGATING)
            "Flow_In":            round(flow_in,  4),
            "Flow_Out":           round(flow_out, 4),
            "Flow_Loss":          round(flow_loss, 4),
            "Flow_Loss_Pct":      round(flow_loss_pct, 4),
            "Bg_Loss_Rate_Pct":   round(bg_rate * 100.0, 4),
            "Excess_Loss_Pct":    round(excess_loss, 4),
            # ── Pressure
            "Pressure_In":        round(press_in,  4),
            "Pressure_Out":       round(press_out, 4),
            "DP_Actual":          round(dp_actual, 4),
            "DP_Predicted":       round(dp_pred,   4),
            "DP_Deviation":       round(dp_dev,    4),
            "Flow_DP_Ratio":      round(flow_dp_ratio, 4),
            # ── Derived
            "Velocity_Proxy":     round(velocity_prx, 4),
            # ── Targets
            "Anomaly_Type":       a_type,
            "Anomaly_Label":      a_label,
            "Anomaly_Binary":     a_binary,
            "Fault_Here":         fault_here,
            "Severity":           round(severity, 4),
        })

        # ══ PROPAGATE ══
        flow_in  = flow_out
        press_in = press_out

    return rows



# ══════════════════════════════════════════════════════════════════════════════
# 6.  FULL DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(n_timestamps: int, anomaly_rate: float, start_date: str,
                  rng: np.random.Generator) -> pd.DataFrame:
    base_dt  = datetime.strptime(start_date, "%Y-%m-%d")
    span_min = n_timestamps * 14          # 14-minute step (NOT multi-year)
    ts_list  = [base_dt + timedelta(minutes=round(i * span_min / n_timestamps))
                for i in range(n_timestamps)]
    branch_list = [(gn, gcfg, bn, bcfg)
                   for gn, gcfg in NETWORK.items()
                   for bn, bcfg in gcfg["branches"].items()]
    all_rows: list[dict] = []
    for ts in ts_list:
        for gn, gcfg, bn, bcfg in branch_list:
            n_segs = len(bcfg["segs"])
            if rng.random() < anomaly_rate:
                a_type    = rng.choice(["leak", "burst", "theft"], p=ANOMALY_WEIGHTS)
                fault_idx = int(rng.integers(0, n_segs))
            else:
                a_type, fault_idx = None, -1
            all_rows.extend(generate_branch(ts, gn, gcfg, bn, bcfg, fault_idx, a_type, rng))
    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    n_segs = sum(len(b["segs"]) for g in NETWORK.values() for b in g["branches"].values())  # 95
    n_ts_train = -(-150_000 // n_segs)   # 1579
    n_ts_test  = -(-30_000  // n_segs)   # 316
    train = build_dataset(n_ts_train, 0.20, "2019-01-01", np.random.default_rng(SEED))
    test  = build_dataset(n_ts_test,  0.05, "2022-07-01", np.random.default_rng(SEED + 1))

    ok = True
    for name, df in (("jordan_v4_train.csv", train), ("jordan_v4_test.csv", test)):
        p = os.path.join(a.out, name)
        df.to_csv(p, index=False)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"{name}: {len(df):,} x {df.shape[1]}  sha256={h}")
        if a.verify:
            good = h == EXPECTED_SHA256[name]
            ok &= good
            print("   ", "MATCH repo file" if good else "DIFFERENT from repo file")
    if a.verify and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
