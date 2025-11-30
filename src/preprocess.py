import pandas as pd

TARGET = "Churn Label"

DROP_COLS = [
    "Customer ID",
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason"
]

def main():
    df = pd.read_csv("data/raw.csv")

    # 1) garder la target à part
    y = df[TARGET]

    # 2) features = tout sauf target + colonnes fuite
    X = df.drop(columns=[TARGET] + DROP_COLS)

    # 3) transformer la target en 0/1 si elle est Yes/No
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})

    processed = X.copy()
    processed[TARGET] = y

    processed.to_csv("data/processed.csv", index=False)

    print("processed.csv créé ✅")
    print("shape:", processed.shape)
    print("target distribution:\n", processed[TARGET].value_counts())

if __name__ == "__main__":
    main()
