import numpy as np

a = np.arange(5)
res = []
for i in range(10):
    res.append(a)
print(res)
print(np.concatenate(res))


a_res = np.array([])
for i in range(10): 
    a_res.append(a)
print(a_res)
