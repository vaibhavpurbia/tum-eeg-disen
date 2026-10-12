import pandas as pd

print("Downloading VitalDB track directory...")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

print("\n--- Available BIS-related tracks ---")
bis_tracks = df_trks[df_trks['tname'].str.contains('BIS', case=False, na=False)]['tname'].unique()
print(bis_tracks)

print("\n--- Available MAC / Volatile tracks ---")
mac_tracks = df_trks[df_trks['tname'].str.contains('MAC|SEVO', case=False, na=False)]['tname'].unique()
print(mac_tracks)