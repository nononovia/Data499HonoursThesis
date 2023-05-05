import os
import pandas as pd

directory_path = "A_results"
list_of_df = []
for filename in os.listdir(directory_path):
   df = pd.read_csv(os.path.join(directory_path, filename))
   df = df.iloc[:, 1]
   list_of_df.append(df)

result_df = pd.concat(list_of_df, axis=1, ignore_index=True)
result_df = result_df.transpose()


print()