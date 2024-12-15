from get_yelp_entropy import run
#from math import log2
import argparse
import warnings
import re
warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="", help="Name of the Dataset")
argparser.add_argument("--json_metis", type=str, default="", help="path of JSON file of the partition")
argparser.add_argument("--json_ew", type=str, default="", help="path of JSON file of the partition")
argparser.add_argument("--json_eb", type=str, default="", help="path of JSON file of the partition")
argparser.add_argument("--log", type=str, default="", help="path of log file of the partition")
argparser.add_argument("--no_of_part", type=int, default=4, help="No of partitions")
args = argparser.parse_args()


with open(args.log, 'r') as f: data = f.read()
metis = re.findall(r"Metis partitioning: \d*\.\d* seconds", data)
ew_eb = re.findall(r"Total Time :\s*\d*\.\d*", data)

if len(metis) < 1 or len(ew_eb) < 1:
        print("Incomplete run of partition generation codes or log file error.")
        exit()

time_metis = float(metis[0].split()[2])
time_ew = float(ew_eb[0].split()[3])


print(f"+{'-'*9}+{'-'*23}+")
print(f"|{' '*9}|{args.dataset:^23}|")
print(f"| Methods | H(P)  |  std |  time  |")
print(f"+{'-'*9}+{'-'*7}+{'-'*6}+{'-'*8}+")

out = run(args.json_metis, args.no_of_part)
print(f"|{'METIS':^9}|{round(out[0], 2):^7}|{round(out[1], 2):^6}|{round(time_metis, 2):^8}|")

out = run(args.json_ew, args.no_of_part)
print(f"|{'EW':^9}|{round(out[0], 2):^7}|{round(out[1], 2):^6}|{round(time_ew, 2):^8}|")

print(f"+{'-'*9}+{'-'*7}+{'-'*6}+{'-'*8}+")
