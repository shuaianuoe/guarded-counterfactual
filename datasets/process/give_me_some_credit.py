import numpy as np
import pandas as pd
from scipy import stats
from saving import save_to_file


# def main():
# path = "raw/"
# Using only train file, because test file does not contain label information.
trainFile = "givemecredit.csv"

# Read Data from csv
train_df = pd.read_csv( trainFile, index_col=False)

# drop rows with missing values
train_df = train_df.dropna(axis=0)

continuous_cols = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
target = ["SeriousDlqin2yrs"]

# change labeling to be consistent with our notation
label_map = {0: 1, 1: 0}
train_df["SeriousDlqin2yrs"] = train_df["SeriousDlqin2yrs"].map(label_map)

# get rid of outliers
data_auxiliary = train_df[continuous_cols]
idx = (np.abs(stats.zscore(data_auxiliary)) < 3.0).all(axis=1)
train_df = train_df.iloc[list(idx)]
train_df = train_df[continuous_cols + target]
# train_df.to_csv("give_me_some_credit.csv", index=False)


train_df['RevolvingUtilizationOfUnsecuredLines'] = train_df['RevolvingUtilizationOfUnsecuredLines'].round(1)
train_df['DebtRatio'] = train_df['DebtRatio'].round(1)
train_df['MonthlyIncome'] = (train_df['MonthlyIncome'] / 1000).round() * 1000


zero_rows = train_df[train_df.iloc[:, -1] == 0]
one_rows_sample = train_df[train_df.iloc[:, -1] == 1].sample(n=1*len(zero_rows))

result_df = pd.concat([zero_rows, one_rows_sample], axis=0)


save_to_file(result_df, "give_me_some_credit")


# if __name__ == "__main__":
#     main()
