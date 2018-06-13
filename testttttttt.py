import numpy as np



x = float('nan')


acc_list_all = np.array([x, x, x], dtype=np.float32)

test = [0.0, 0.0, 1.0]


print(np.nan_to_num(acc_list_all) + test)