#Combine winner into 1 column
#Winner: 0 = tie; 1 = model_a; 2 = model_b

def combine_winner(row):
    """
    Function to assign value to winner column
    """
    if row["winner_tie"] == 1:
        return 0  # tie
    elif row["winner_model_a"] == 1:
        return 1  # model A wins
    else:
        return 2  # model B wins

def get_single_winner_col(dataframe): 
    """
    Combines the 3 winner columns into a single column where 0 = tie; 1 = model_a; 2 = model_b

    Parameters:
        df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: new DataFrame with a single winner column
    """
    df = dataframe.copy()
    df["winner"] = df.apply(combine_winner, axis=1)
    df = df.drop(columns=["winner_model_a", "winner_model_b", "winner_tie"])
    return df


#Split prompts/response into individual entries. Can access through id
def expand_df(dataframe):
    """
    Splits lists of prompts and responses in the dataframe into individual entries.

    Parameters:
        df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: new DataFrame with each entry having a single prompt and pair of responses
    """
    df = dataframe.copy()
    df["triples"] = df.apply(
        lambda row: list(zip(row["prompt"], row["response_a"], row["response_b"])),
        axis=1
    )
    
    df_expanded = df.explode("triples").reset_index(drop=True)
    
    # Split back into separate columns
    df_expanded["prompt"] = df_expanded["triples"].apply(lambda x: x[0])
    df_expanded["response_a"] = df_expanded["triples"].apply(lambda x: x[1])
    df_expanded["response_b"] = df_expanded["triples"].apply(lambda x: x[2])
    
    df_expanded = df_expanded.drop(columns=["triples"])
    return df_expanded

def lowercase_df(dataframe):
    """
    Lowercase all string columns in a DataFrame.
    """
    df = dataframe.copy()
    
    # Select object/string columns
    str_cols = df.select_dtypes(include='object').columns
    
    # Apply str.lower() to each column
    for col in str_cols:
        df[col] = df[col].str.lower()
    
    return df


def update_ids(dataframe, keep_old = True):
    """
    Replace the `id` column with continuous integers starting from 0.

    Parameters:
        dataframe (pd.DataFrame): input DataFrame
        keep_old (bool): if True, keep original IDs in a new column 'old_id'

    Returns:
        pd.DataFrame: new DataFrame with regularized IDs
    """
    df = dataframe.copy()  # avoid modifying the original
    
    if keep_old:
        df["old_id"] = df["id"]
    
    df["id"] = range(len(df))
    
    return df