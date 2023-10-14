from sklearn.model_selection import train_test_split
import numpy as np

# 定义变化函数
def biased_row(row, columns, df, prob):
    for col in columns:
        if np.random.rand() < prob:
            row[col] = np.random.choice(df[col])
    return row
    

def save_to_file(df, name, test_size=0.4):

    df_train, df_test = train_test_split(df, test_size=test_size)
    # train_size = int((1-test_size) * len(df))
    # df_train = df.iloc[:train_size]
    # df_test = df.iloc[train_size:]
    
    df.to_csv(f"results/{name}.csv", index=False)
    df_train.to_csv(f"results/{name}_train.csv", index=False)
    df_test.to_csv(f"results/{name}_test.csv", index=False)
    df_train.to_csv(f"results/{name}_train_index.csv", columns=[], header=False)
    df_test.to_csv(f"results/{name}_test_index.csv", columns=[], header=False)

    # prob = 1

    # df_test_bias = df_test.apply(biased_row, columns=df.columns, df=df, prob=prob, axis=1)
    # df_test_bias.to_csv(f"results/{name}_test_bias.csv", index=False)
    
    # import pandas as pd
    # import numpy as np
    # from sklearn.model_selection import train_test_split
    
    # # 生成示例数据
    # data = {
    #     'A': [1, 2, 3, 4, 5],
    #     'B': [5, 6, 7, 8, 9],
    #     'C': [9, 8, 7, 6, 5]
    # }
    # df = pd.DataFrame(data)
    
    # # 分割数据
    # test_size = 0.2
    # df_train, df_test = train_test_split(df, test_size=test_size)
    

    # prob = 0.5
    # df_test_bias = df_test.apply(biased_row, prob=prob, axis=1)
    
    # print(df_test_bias)
