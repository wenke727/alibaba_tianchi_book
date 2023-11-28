#%%
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
ans = classification_report(y_true, y_pred, labels=target_names, output_dict=True)


# y_pred = [1, 1, 0]
# y_true = [1, 1, 1]
# print(ans)

#%%
import pandas as pd

pd.DataFrame(ans).T

# %%
