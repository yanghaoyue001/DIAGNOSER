import seaborn as sns
# sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.boxplot(x=tips["total_bill"])
# ax = sns.boxplot(x="day", y="total_bill", data=tips)