

```python
import pandas as pd
%pylab inline
import seaborn as sns
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
high_tmle = pd.read_csv('tmle_high_results.csv',delimiter = ',')
low_tmle = pd.read_csv('tmle_low_results.csv',delimiter = ',')
high_ctmle = pd.read_csv('ctmle_high_results.csv',delimiter = ',')
low_ctmle = pd.read_csv('ctmle_low_results.csv',delimiter = ',')
```


```python
dataset = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,25,26,27,28,45,46,47,48,49,50,51,52,53,54,55,56]
trueATE_list = []
trueATE_list_high = []
lb_list = []
ub_list = []
high_tmle_performance = []
high_ctmle_performance = []
for i in dataset:
    index = high_tmle[(high_tmle.DGPid == i)].index.tolist()
    trueATE_list = high_tmle['trueATE'][index]
    lb_list = high_tmle['lb'][index]
    ub_list = high_tmle['ub'][index]
    total = 0
    count = 0
    for j, k, l in zip(trueATE_list, lb_list, ub_list):
        total += 1
        if j >= k and j <= l:
            count += 1
    if count/total >= 0.85:
        high_tmle_performance.append(0)
    else:
        high_tmle_performance.append(1)
    trueATE = float(high_tmle['trueATE'][index][0:1,])
    index = high_ctmle[(high_ctmle.DGPid == i)].index.tolist()
    trueATE_list = high_ctmle['trueATE'][index]
    lb_list = high_ctmle['lb'][index]
    ub_list = high_ctmle['ub'][index]
    total = 0
    count = 0
    for j, k, l in zip(trueATE_list, lb_list, ub_list):
        total += 1
        if j >= k and j <= l:
            count += 1
    if count/total >= 0.85:
        high_ctmle_performance.append(0)
    else:
        high_ctmle_performance.append(1)
    trueATE_list_high.append(trueATE)
# for i,j in zip(dataset, high_tmle_performance):
#     print(i,j)
# for i,j in zip(dataset, high_ctmle_performance):
#     print(i,j)
print("Good-bad performance wrt DGPid ---- high-dim only")
plt.scatter(dataset, high_tmle_performance, s=25)
plt.scatter(dataset, high_ctmle_performance, s=10)
plt.show()
```

    Good-bad performance wrt DGPid ---- high-dim only
    


![png](output_2_1.png)



```python
dataset_low = [17,18,19,20,21,22,23,24,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,57,58,59,60,61,62,63,64]
trueATE_list = []
trueATE_list_low = []
lb_list = []
ub_list = []
low_tmle_performance = []
low_ctmle_performance = []
for i in dataset_low:
    index = low_tmle[(low_tmle.DGPid == i)].index.tolist()
    trueATE_list = low_tmle['trueATE'][index]
    lb_list = low_tmle['lb'][index]
    ub_list = low_tmle['ub'][index]
    total = 0
    count = 0
    for j, k, l in zip(trueATE_list, lb_list, ub_list):
        total += 1
        if j >= k and j <= l:
            count += 1
    if count == 0 or count/total >= 0.85:
        low_tmle_performance.append(0)
    else:
        low_tmle_performance.append(1)
    trueATE = float(low_tmle['trueATE'][index][0:1,])
    index = low_ctmle[(low_ctmle.DGPid == i)].index.tolist()
    trueATE_list = low_ctmle['trueATE'][index]
    lb_list = low_ctmle['lb'][index]
    ub_list = low_ctmle['ub'][index]
    total = 0
    count = 0
    for j, k, l in zip(trueATE_list, lb_list, ub_list):
        total += 1
        if j >= k and j <= l:
            count += 1
    if count == 0 or count/total >= 0.85:
        low_ctmle_performance.append(0)
    else:
        low_ctmle_performance.append(1)
    trueATE_list_low.append(trueATE)
# for i,j in zip(dataset_low, low_tmle_performance):
#     print(i,j)
# for i,j in zip(dataset_low, low_ctmle_performance):
#     print(i,j)
print("Good-bad performance wrt DGPid ---- low-dim only")
plt.scatter(dataset_low, low_tmle_performance, s=25)
plt.scatter(dataset_low, low_ctmle_performance, s=10)
plt.show()
```

    Good-bad performance wrt DGPid ---- low-dim only
    


![png](output_3_1.png)



```python
tmle_combined = []
ctmle_combined = []
count_high = 0
count_low = 0
dataset_combined = list(range(1,65))
for i in dataset_combined:
    if i in dataset:
        tmle_combined.append(high_tmle_performance[count_high])
        ctmle_combined.append(high_ctmle_performance[count_high])
        count_high += 1
    if i in dataset_low:
        tmle_combined.append(low_tmle_performance[count_low])
        ctmle_combined.append(low_ctmle_performance[count_low])
        count_low += 1
print("Good-bad performance wrt DGPid ---- combined")
plt.scatter(dataset_combined, tmle_combined, s=25)
plt.scatter(dataset_combined, ctmle_combined, s=10)
plt.show()
```

    Good-bad performance wrt DGPid ---- combined
    


![png](output_4_1.png)



```python
binary_list = list(range(17,29)) + list(range(33,37)) + list(range(41,57))
continuous_list = list(range(1,17)) + list(range(29, 33)) + list(range(37, 41)) + list(range(57, 65))
binary_continuous = []
for i in range(1,65):
    if i in binary_list:
        binary_continuous.append(1)
    else:
        binary_continuous.append(0)
tmle_count = [0, 0, 0, 0]
for i,j in zip(binary_continuous, tmle_combined):
    if i ==1 and j==0:
        tmle_count[0] += 1
    if i == 1 and j == 1:
        tmle_count[1] += 1
    if i == 0 and j==0:
        tmle_count[2] += 1
    if i == 0 and j == 1:
        tmle_count[3] += 1
print("Good-bad performance wrt Binary_continuous ---- tmle")
plt.bar(['binary_good', 'binary_bad', 'continuous_good', 'continuous_bad'], tmle_count)
```

    Good-bad performance wrt Binary_continuous ---- tmle
    




    <BarContainer object of 4 artists>




![png](output_5_2.png)



```python
ctmle_count = [0, 0, 0, 0]
for i,j in zip(binary_continuous, ctmle_combined):
    if i ==1 and j==0:
        ctmle_count[0] += 1
    if i == 1 and j == 1:
        ctmle_count[1] += 1
    if i == 0 and j==0:
        ctmle_count[2] += 1
    if i == 0 and j == 1:
        ctmle_count[3] += 1
print("Good-bad performance wrt Binary_continuous ---- ctmle")
plt.bar(['binary_good', 'binary_bad', 'continuous_good', 'continuous_bad'], ctmle_count)
```

    Good-bad performance wrt Binary_continuous ---- ctmle
    




    <BarContainer object of 4 artists>




![png](output_6_2.png)



```python
dataset_combined = list(range(1,65))
count_high = 0
count_low = 0
num_covariates = []
proportion_discrete = []
filename_list = []
for i in dataset_combined:
    discrete_count = 0
    if i in dataset:
        index = high_tmle[(high_tmle.DGPid == i)].index.tolist()
        filename = high_tmle['dataset'][index][0:1,].tolist()[0]
        filename_list.append(filename)
        temp_read = pd.read_csv(filename + '.csv',delimiter = ',')
        num_covariates.append(temp_read.shape[1] - 2)
        for k in list(temp_read)[2:]:
            if isinstance(temp_read[k].tolist()[0], int):
                discrete_count += 1
            else:
                if len(str(temp_read[k].tolist()[0]).split('.')[1]) <= 10:
                    discrete_count += 1
    if i in dataset_low:
        index = low_tmle[(low_tmle.DGPid == i)].index.tolist()
        filename = low_tmle['dataset'][index][0:1,].tolist()[0]
        filename_list.append(filename)
        temp_read = pd.read_csv(filename + '.csv',delimiter = ',')
        num_covariates.append(temp_read.shape[1] - 2)
        for k in list(temp_read)[2:]:
            if isinstance(temp_read[k].tolist()[0], int):
                discrete_count += 1
            else:
                if len(str(temp_read[k].tolist()[0]).split('.')[1]) <= 10:
                    discrete_count += 1
    proportion_discrete.append(discrete_count/(temp_read.shape[1] - 2))
print("Good-bad performance wrt number of covariates")
plt.scatter(num_covariates, tmle_combined, s=25)
plt.scatter(num_covariates, ctmle_combined, s=10)
plt.show()
```

    Good-bad performance wrt number of covariates
    


![png](output_7_1.png)



```python
print("Good-bad performance wrt proportion of discrete variables")
plt.scatter(proportion_discrete, tmle_combined, s=25)
plt.scatter(proportion_discrete, ctmle_combined, s=10)
plt.show()
```

    Good-bad performance wrt proportion of discrete variables
    


![png](output_8_1.png)



```python
trueATE_list_combined = []
count_high = 0
count_low = 0
dataset_combined = list(range(1,65))
for i in dataset_combined:
    if i in dataset:
        trueATE_list_combined.append(abs(trueATE_list_high[count_high]))
        count_high += 1
    if i in dataset_low:
        trueATE_list_combined.append(abs(trueATE_list_low[count_low]))
        count_low += 1
print("Good-bad performance wrt true value")
plt.scatter(trueATE_list_combined, tmle_combined, s=25)
plt.scatter(trueATE_list_combined, ctmle_combined, s=10)
plt.show()
```

    Good-bad performance wrt true value
    


![png](output_9_1.png)



```python
df = pd.DataFrame({'Binary_continuous': binary_continuous, 
                    'Number of Covariates': num_covariates, 
                    'Proportion of discrete variables': proportion_discrete,
                    'True Parameter Value': trueATE_list_combined,
                    'tmle': tmle_combined,
                    'ctmle': ctmle_combined}) 
```


```python
fig,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()
```


![png](output_11_0.png)

