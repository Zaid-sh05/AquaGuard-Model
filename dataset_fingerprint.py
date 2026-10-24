"""
Fingerprint a jordan_v4_*.csv so it can be compared against the reference values in the audit report.
Usage:  python3 dataset_fingerprint.py path/to/jordan_v4_train.csv
Closes the one provenance gap that cannot be closed from the chat alone: whether YOUR file (the one used for the
paper / Colab) is the same file the recovered generator produces.
"""
import sys, hashlib
import pandas as pd

path = sys.argv[1]
h = hashlib.sha256(open(path, "rb").read()).hexdigest()
df = pd.read_csv(path)
print("file            :", path)
print("sha256          :", h)
print("rows x cols     :", f"{len(df):,} x {df.shape[1]}")
print("timestamp range :", df['Timestamp'].min(), "->", df['Timestamp'].max())
print("class counts    :", df['Anomaly_Type'].value_counts().to_dict())
print("sum Flow_In     :", round(df['Flow_In'].sum(), 3))
print("sum Pressure_Out:", round(df['Pressure_Out'].sum(), 3))
print("sum DP_Predicted:", round(df['DP_Predicted'].sum(), 3))
print("sum Severity    :", round(df['Severity'].sum(), 3))
print("first row       :", df.iloc[0][['Timestamp','Segment_ID','Flow_In','Flow_Out','Pressure_In','Pressure_Out']].to_dict())
print("last row        :", df.iloc[-1][['Timestamp','Segment_ID','Flow_In','Flow_Out','Pressure_In','Pressure_Out']].to_dict())
