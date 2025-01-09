import csv
import os

def write_to_csv_file(file_name="csv_file.csv", column_names=['column_1','column_2', 'column_3'],data=[0,0,0]):
    
    csv_file_exists = False
    
    try:
            with open(file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                if any(row for row in reader):
                    csv_file_exists = True
    except FileNotFoundError:
        pass

        # Open the CSV file to save data
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_file_exists:
            writer.writerow(column_names)
        writer.writerow(data)

def delete_csv_file(file_name="csv_file.csv"):
     if os.path.exists(file_name):
         os.remove(file_name)
