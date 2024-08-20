import time

from My_Threads import my_thread


if __name__ == "__main__":

    f_name ="Salah"
    with open(f_name, "w") as fl:
        fl.write(f"OK\n")
        fl.write(f"TFN=5\n")
        fl.write(f"0ASD\n")
        fl.write(f"1ASD\n")
        fl.write(f"2ASD\n")
        fl.write(f"3ASD\n")
        fl.write(f"4ASD\n")
    print("file writen :)")



"""""
Produce UUID in linux bash    
echo $(cat/proc/sys/kernel/random/uuid)
OR
echo $(uuidgen)
"""
