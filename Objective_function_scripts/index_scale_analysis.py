import pandas as pd
from typing import Dict, Tuple, List


datas = []

columns = ['model', 'dataset', 'recall', 'query_latency', 'construction_time']

# First CSV
df = pd.read_csv("graph_new_datasets_standalone_results_index_v1.csv")

for index, row in df.iterrows():
    data = [
        row['model'],
        row['dataset'],
        row['recall with 2k refinement'],
        1/row['qps with 2k refienement'],
        row['construction time']
    ]
    datas.append(data)

# Second CSV
df = pd.read_csv("graph_new_datasets_augmented_results_index_v1.csv")

for index, row in df.iterrows():
    data = [
        row['model1'] + '+' + row['model2'],
        row['dataset'],
        row["augmented model's recall with refinement"],
        1/row["augmented model's qps with refinement in parallal settings"],
        row["augmented model's construction time in parallal settings"]
    ]
    datas.append(data)

# Convert to DataFrame
final_df = pd.DataFrame(datas, columns=columns)

# Save to CSV
final_df.to_csv("combined_graph_results.csv", index=False)

profiles_per_dataset: Dict[str, List[Tuple[float, float]]] = {}

for dataset, group in final_df.groupby("dataset"):
    g = group.copy()
    g = g[pd.notna(g["query_latency"]) & (g["query_latency"] > 0)]
    g = g[pd.notna(g["construction_time"]) & (g["construction_time"] > 0)]
    latencies = g["query_latency"].astype(float)
    ctimes = g["construction_time"].astype(float)
    lat_q = latencies.quantile([0.25, 0.5]).tolist()
    cst_q = ctimes.quantile([0.25, 0.5]).tolist()
    profiles: List[Tuple[float, float, float]] = []
    for (Tq, Tc) in zip(lat_q, cst_q):
        profiles.append((float(Tq), float(Tc)))

    profiles_per_dataset[dataset] = profiles
    
_500k_recalls=[]
_100k_recalls=[]
_50k_recalls=[]
_10k_recalls=[] 
recalls=[] 
    
for dataset in profiles_per_dataset.keys():
    base_best_recall=0
    augmented_best_recall=0
    dataset_filtered_df=final_df[final_df['dataset']==dataset]
    for index, row in dataset_filtered_df.iterrows():
        if '+' in row['model'] and row['query_latency']<=profiles_per_dataset[dataset][0][0] and row['construction_time']<=profiles_per_dataset[dataset][0][1]:
            augmented_best_recall=max(augmented_best_recall,row['recall'])
        elif (not '+' in row['model']) and row['query_latency']<=profiles_per_dataset[dataset][0][0] and row['construction_time']<=profiles_per_dataset[dataset][0][1]:
            base_best_recall=max(base_best_recall,row['recall'])
    # print(dataset, base_best_recall, augmented_best_recall)
    if '500k' in dataset:
        _500k_recalls.append((base_best_recall,augmented_best_recall))
    elif '100k' in dataset:
        _100k_recalls.append((base_best_recall,augmented_best_recall))
    elif '50k' in dataset:
        _50k_recalls.append((base_best_recall,augmented_best_recall))
    elif '10k' in dataset:
        _10k_recalls.append((base_best_recall,augmented_best_recall))
    else:
        recalls.append((base_best_recall,augmented_best_recall))
        
    
    
print("Hard Constraint")

for label, data in {
    "500k": _500k_recalls,
    "100k": _100k_recalls,
    "50k": _50k_recalls,
    "10k": _10k_recalls,
    "1M": recalls
}.items():
    if data:
        base = sum(x[0] for x in data)/len(data)
        aug  = sum(x[1] for x in data)/len(data)
        print(label, base, aug)

_500k_recalls=[]
_100k_recalls=[]
_50k_recalls=[]
_10k_recalls=[] 
recalls=[] 

for dataset in profiles_per_dataset.keys():
    base_best_recall=0
    augmented_best_recall=0
    dataset_filtered_df=final_df[final_df['dataset']==dataset]
    for index, row in dataset_filtered_df.iterrows():
        if '+' in row['model'] and row['query_latency']<=profiles_per_dataset[dataset][1][0] and row['construction_time']<=profiles_per_dataset[dataset][1][1]:
            augmented_best_recall=max(augmented_best_recall,row['recall'])
        elif (not '+' in row['model']) and row['query_latency']<=profiles_per_dataset[dataset][1][0] and row['construction_time']<=profiles_per_dataset[dataset][1][1]:
            base_best_recall=max(base_best_recall,row['recall'])
    # print(dataset, base_best_recall, augmented_best_recall)
    if '500k' in dataset:
        _500k_recalls.append((base_best_recall,augmented_best_recall))
    if '100k' in dataset:
        _100k_recalls.append((base_best_recall,augmented_best_recall))
    if '50k' in dataset:
        _50k_recalls.append((base_best_recall,augmented_best_recall))
    if '10k' in dataset:
        _10k_recalls.append((base_best_recall,augmented_best_recall))
    else:
        recalls.append((base_best_recall,augmented_best_recall))


print("Relaxed Constraint")

for label, data in {
    "500k": _500k_recalls,
    "100k": _100k_recalls,
    "50k": _50k_recalls,
    "10k": _10k_recalls,
    "1M": recalls
}.items():
    if data:
        base = sum(x[0] for x in data)/len(data)
        aug  = sum(x[1] for x in data)/len(data)
        print(label, base, aug)