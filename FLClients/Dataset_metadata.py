import json

from Data_operations.DTAT_LAKE import get_data_from_file
from pathlib import Path


dir_path = "/WriteAbleData/One_Day_Data_processed/merged_output_oct20_22"
file_path = "/WriteAbleData/One_Day_Data_processed/Python/5gDataProcessing_0/14days_merged_file_info_Nov2022.json"
phy_servers = ["Nation", "Niort", "Rochelle", "s137", "s147", "s155"]

layer = "physical"

fs = json.load(open(file_path))

info={}
for day in range(1, 15):
    info[day] = {"cpu":0, "memory":0, "network":0, "disk":0}
    for rsc in ["cpu", "memory", "network", "disk"]:
        clm_count=0
        for srv in phy_servers:
            clm_count += fs[str(day)][rsc][srv]["column_count"]

        print(f"{day}.{rsc}:{clm_count}")
        info[day][rsc] = clm_count


sfl = Path(f"{dir_path}/{layer}")

print(f"Writing info file\n{sfl}")
if not sfl.exists():
    sfl.mkdir(parents=True)
with open(f'{sfl}/dataset_metadata.json', 'w') as json_file:
    json.dump(info, json_file, indent=4)
