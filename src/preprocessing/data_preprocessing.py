import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import json
import pathlib
import src.preprocessing.data_preprocssing_funcs as dpf

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_PATH = SCRIPT_DIR / "data"

data = pd.read_csv(DATA_PATH / "train.csv")
test_data = pd.read_csv(DATA_PATH / "test.csv")


#Make lists actual lists, pandas defaults to strings when reading from CSV file
list_cols = ["prompt", "response_a", "response_b"]

for col in list_cols:
    data[col] = data[col].apply(json.loads)
    test_data[col] = test_data[col].apply(json.loads)
    
#Base Panda DF
data

#DF where winners represented in a single column "winner"
#winner: 0 = tie; 1 = model_a; 2 = model_b
base_df = dpf.get_single_winner_col(data)

#df where prompt/responses are split individually and respectively. 
#Ex: prompt: ["1", "2"] response_a: ["a1", "a2"] response_b: ["b1", "b2"]
#Becomes: prompt: ["1"] response_a: ["a1"] response_b: ["b1"]; prompt: ["2"] response_a: ["a2"] response_b: ["b2"]
#Other columns are kept and maintained
individual_response_df = dpf.expand_df(base_df)

#Option so you can see the whole prompt/responses
#pd.set_option("display.max_colwidth", None)

#original_df has columns: ['id', 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
original_df = data

#df has columns: ['id', 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'winner']
#Where winner = 0: tie, 1: model_a, 2: model_b 
#And each prompt and response contains a single string.
df = dpf.lowercase_df(individual_response_df)
#df values
column_names = list(df.columns)
unique_models = pd.unique(data[["model_a", "model_b"]].values.ravel())
ids = data["id"]

#test data has columns: ['id', 'prompt', 'response_a', 'response_b']
#each prompt and response are lists of strings
test_data

print(test_data.head())