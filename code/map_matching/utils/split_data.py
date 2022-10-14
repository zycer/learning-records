import pandas as pd


f = pd.read_csv(r"C:\Users\11718\Desktop\train.csv", encoding="utf-8")
rows_num = f.shape[0]
file_num = 20
once = rows_num // file_num
current = 0
next_ = 0
i = 0

for j in range(file_num):
    if j < file_num:
        next_ += once
    else:
        next_ = rows_num

    print(current, next_)
    f_new = f[current:next_]
    f_new.to_csv(f"../data/gps_trajectory/train{i}.csv", index=False, header=True)
    current = next_
    i += 1
