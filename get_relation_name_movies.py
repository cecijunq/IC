# import pandas module 
import pandas as pd 
  
# creating a data frame 
df = pd.read_csv("./data/oscar_full.csv", on_bad_lines='skip', sep=';')
df = df[["TITLE_EN", "TITLE_PT"]]
df = df.set_index('TITLE_PT')
print(df)