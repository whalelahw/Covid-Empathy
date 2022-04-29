import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--step", default=64, type=int)
args = parser.parse_args()
gold = pd.read_csv('./OurAnnotations.csv')
# new = pd.read_csv('./Newpred_addseeker_' + str(args.step) + '.csv')
# new = pd.read_csv('./Newpred_final_2.csv')
# new = pd.read_csv('./Newpred.csv')
new = pd.read_csv('./Oldpred_2.csv')
gold_er = gold['ER_label_annot'].tolist()
new_er = new['ER_label'].tolist()
print(accuracy_score(gold_er, new_er))
print(f1_score(gold_er, new_er, average='macro'))
print("")
gold_ex = gold['EX_label_annot'].tolist()
new_ex = new['EX_label'].tolist()
print(accuracy_score(gold_ex, new_ex))
print(f1_score(gold_ex, new_ex, average='macro'))
print("")
gold_ip = gold['IP_label_annot'].tolist()
new_ip = new['IP_label'].tolist()
print(accuracy_score(gold_ip, new_ip))
print(f1_score(gold_ip, new_ip, average='macro'))