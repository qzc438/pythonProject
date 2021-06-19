# sklearn one-hot encoding
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
array = enc.transform([[0, 1, 3]]).toarray()
print(array)

# pandas get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)  # 属性
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)  #
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

import pandas as pd;
data = [['handsome', 'tall', 'Japan'],
        ['ugly', 'short', 'Japan'],
        ['handsome', 'middle', 'Chinese']]
df = pd.DataFrame(data, columns=['face', 'stature ', ' country '])
df, df_cat = one_hot_encoder(df)
print(df)
