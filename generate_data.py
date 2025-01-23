import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
data = {
    "Machine_ID": range(1, 101),
    "Temperature": np.random.randint(50, 100, 100),
    "Run_Time": np.random.randint(60, 200, 100),
    "Downtime_Flag": np.random.choice([0, 1], 100, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv("data/sample_data.csv", index=False)
