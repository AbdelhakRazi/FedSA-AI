import os
import time
from threading import Thread

client_no =1

def load_file(file_name):
    print(file_name)

def check_file(f_name):
    i = 0
    while True:
        i+=1
        try:
            with open(f_name, "r") as fl:
                a = fl.readline()

                if a != "OK":
                    print("No response")
                    time.sleep(2)
                else:
                    print("got respose")
                    l = fl.writelines()
                    l = l[2:] #remove first two elements
                    split_no_file_naDCme = l[client_no]
                    load_file(split_no_file_name)
                    break
        except Exception as ex:
            print("no file ", i ,"\n", ex)
            time.sleep(2)

if __name__ == "__main__":

    f_name ="Salah"
    if os.path.exists(f_name):
        os.remove(f_name)
    Thread(target=check_file, args=(f_name,)).start()