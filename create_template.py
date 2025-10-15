# create_template.py
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
X.to_csv("sample_input_template.csv", index=False)

print("✅ Sample CSV template saved as sample_input_template.csv")
