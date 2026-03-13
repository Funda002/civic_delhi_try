import pandas as pd
import os

class WardInfoProvider:
    def __init__(self, csv_path="ward_data.csv"):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            self.df = None
            print("Warning: ward_data.csv not found.")
        else:
            # Load English rows only
            self.df = pd.read_csv(self.csv_path).iloc[:250]
            self.df['ward_no'] = self.df['ward_no'].astype(int)

    def get_ward_details(self, ward_no):
        if self.df is None: return None
        try:
            row = self.df[self.df['ward_no'] == int(ward_no)]
            return row.iloc[0].to_dict() if not row.empty else None
        except: return None