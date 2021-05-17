import os

import numpy as np
import pandas as pd
import requests

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")

if __name__ == "__main__":
    data = pd.read_csv("data/raw/heart.csv")
    data = data.drop(columns="target")
    request_features = list(data.columns)
    for i in range(100):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            f"http://{host}:{port}/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())
