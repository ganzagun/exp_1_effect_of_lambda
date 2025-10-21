import argparse, os, pandas as pd
import math

parser = argparse.ArgumentParser(description="Metrics Calculation")
parser.add_argument("--dataset", default="cora", type=str, help="Dataset")
parser.add_argument("--teacher", default="gcn", type=str, help="Teacher Model")
parser.add_argument("--student", default="mlp", type=str, help="Student Model")
parser.add_argument("--exp_setting", default="tran", type=str, help="Experiment setting, one of [tran, ind]")
parser.add_argument("--kd_method", default="glnn", type=str, help="KD method, one of [glnn, glnn_feat_distill, nosmog]" )
parser.add_argument("--teacher_output_dir", default=None, type=str, help="Teacher output directory")
parser.add_argument("--student_output_dir", default=None, type=str, help="Student output directory")
parser.add_argument("--hyperparameters", nargs="*", default=["Temperature"], type=str, help="List of hyperparameters that are varied")

args = parser.parse_args()

if args.exp_setting == "tran":
    args.exp_setting = "transductive"
elif args.exp_setting == "ind":
    args.exp_setting = "inductive"

if args.exp_setting == "transductive":
    teacher_file = os.path.join(args.teacher_output_dir, args.exp_setting, args.dataset, args.teacher, "output.csv")
    student_file = os.path.join(args.student_output_dir, args.exp_setting, args.dataset, f"{args.teacher}_{args.student}", "output.csv")
elif args.exp_setting == "inductive":
    teacher_file = os.path.join(args.teacher_output_dir, args.exp_setting, 'split_rate_0.2', args.dataset, args.teacher, "output.csv")
    student_file = os.path.join(args.student_output_dir, args.exp_setting, 'split_rate_0.2', args.dataset, f"{args.teacher}_{args.student}", "output.csv")

if not os.path.isfile(teacher_file):
    print(teacher_file, "not found")
    exit
if not os.path.isfile(student_file):
    print(student_file, "not found")
    exit


print("Teacher File: ", teacher_file)
print("Student File: ", student_file)

df1 = pd.read_csv(teacher_file)
df2 = pd.read_csv(student_file)


if args.exp_setting == "transductive":    
    fixed_pairs = {"Score": "Score_t", "ECE": "ECE_t"}
    dynamic_pairs = {item: item + "_t" for item in args.hyperparameters}
    df1 = df1[["Teacher Logits"] + args.hyperparameters + ["Score", "ECE"]].rename(columns={**fixed_pairs, **dynamic_pairs})

    fixed_pairs = {"Score": "Score_s", "ECE": "ECE_s"}
    dynamic_pairs = {item: item + "_s" for item in args.hyperparameters}
    df2 = df2[["Teacher Logits"] + args.hyperparameters + ["Score", "ECE"]].rename(columns={**fixed_pairs, **dynamic_pairs})
elif args.exp_setting == "inductive":
    fixed_pairs = {{"Score(test_ind)": "Score_t", "ECE (test_ind)": "ECE_t"}}
    dynamic_pairs = {item: item + "_t" for item in args.hyperparameters}
    df1 = df1[["Teacher Logits"] + args.hyperparameters + ["Score(test_ind)", "ECE (test_ind)"]].rename(columns={**fixed_pairs, **dynamic_pairs})

    fixed_pairs = {"Score (test_ind)": "Score_s", "ECE (test_ind)": "ECE_s"}
    dynamic_pairs = {item: item + "_s" for item in args.hyperparameters}
    df2 = df2[["Teacher Logits"] + args.hyperparameters + ["Score (test_ind)", "ECE (test_ind)"]].rename(columns={**fixed_pairs, **dynamic_pairs})

df = pd.merge(df1, df2, on="Teacher Logits")

df["Score_s"] = df["Score_s"] * 100
df["Score_t"] = df["Score_t"] * 100

# --- L2: Teacher Logits at which ece_t is minimum ---
idx_2 = df["ECE_t"].idxmin()
L2 = df.loc[idx_2, "Teacher Logits"]

# --- L3: Teacher Logits at which score_s is maximum ---
idx_3 = df["Score_s"].idxmax()
L3 = df.loc[idx_3, "Teacher Logits"]

# --- A1_s: score_s at uncalibrated teacher ---
A1_s = df.loc[df["Temperature_t"] == 1, "Score_s"].values[0]

# --- A2_s: score_s at L2 ---
A2_s = df.loc[df["Teacher Logits"] == L2, "Score_s"].values[0]

# --- A3_s: score_s at L3 ---
A3_s = df.loc[df["Teacher Logits"] == L3, "Score_s"].values[0]

# --- E1_s: ece_s at T = 1 ---
E1_s = df.loc[df["Temperature_s"] == 1, "ECE_s"].values[0]

# --- E2_s: ece_s at L2 ---
E2_s = df.loc[df["Teacher Logits"] == L2, "ECE_s"].values[0]

# --- E1_t: ece_t at T = 1 ---
E1_t = df.loc[df["Temperature_t"] == 1, "ECE_t"].values[0]

# --- E2_t: ece_t at L2 ---
E2_t = df.loc[df["Teacher Logits"] == L2, "ECE_t"].values[0]

if A3_s == A1_s:
    proximty_ratio = math.inf
else:
    proximty_ratio = ((A2_s - A1_s) / (A3_s - A1_s))

rank_strength = ((df["Score_s"] > A2_s).sum()) / len(df)

output_file = "output.csv"
new_row = pd.DataFrame(
    [[args.dataset, args.teacher, args.student, args.exp_setting, args.kd_method, proximty_ratio, rank_strength]],
    columns=["Dataset", "Teacher", "Student", "Experiment Setting", "KD Method", "Proximity Ratio", "Rank Strength"]
)
        

if os.path.isfile(output_file):
    new_row.to_csv(output_file, mode="a", header=False, index=False)
else:
    new_row.to_csv(output_file, mode="w", header=True, index=False)
