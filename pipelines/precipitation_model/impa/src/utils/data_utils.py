# -*- coding: utf-8 -*-

NRAYS = 360

VARIABLES_DICT = {
    "TH": "data1",
    "CLASS": "data10",
    "DBZH": "data2",
    "VRADH": "data3",
    "WRADH": "data4",
    "ZDR": "data5",
    "KDP": "data6",
    "RHOHV": "data7",
    "SQIH": "data8",
    "PHIDP": "data9",
}

sat_dataframe = [
    "SAT-corrected_ABI-L2-RRQPEF-heavy_rain",
    "SAT-ABI-L2-RRQPEF-rain_events-sat-thr=10-radius=1h",
    "SAT-ABI-L2-RRQPEF-{location}-file=thr=0",
    "SAT-ABI-L2-RRQPEF-{location}-file=thr=0_split2",
]

data_modification_options = {
    "Elevation": False,
    "Hour_data": False,
    "No_context": False,
    "Lat_lon": False,
    "Lead_time_cond": False,
    "Pred_context": False,
    "Add_lead_to_input": False,
    "No_satellite": False,
    "Vector_field_after=before": False,
}
