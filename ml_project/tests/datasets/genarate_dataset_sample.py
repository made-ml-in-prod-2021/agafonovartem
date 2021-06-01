import pandas as pd
import numpy as np


def generate_dataset_sample(size):
    age = np.random.normal(40, 10, size).astype(int)
    sex = np.random.randint(2, size=size)
    target = np.random.randint(2, size=size)
    df = pd.DataFrame({"Age": age, "Sex": sex, "Target": target})
    df.to_csv("generated_dataset_sample.csv")


if __name__ == "__main__":
    size = 11
    generate_dataset_sample(size)
