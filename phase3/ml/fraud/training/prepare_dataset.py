from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "fraud_oracle_clean.csv"
)

OUTPUT_DIR = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "datasets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RANDOM_STATE = 42

TARGET = "FraudFound_P"

DROP_COLUMNS = [
    "PolicyNumber"
]


def main():

    print()
    print("==========================================")
    print("       FRAUD ML DATASET PREPARATION")
    print("==========================================")

    # ========================================================
    # LOAD
    # ========================================================

    print()
    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Total rows:",
        len(df)
    )

    print(
        "Total columns:",
        len(df.columns)
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column not found: {TARGET}"
        )

    print()
    print("Target distribution:")

    print(
        df[TARGET].value_counts()
    )

    # ========================================================
    # DROP IDENTIFIERS
    # ========================================================

    existing_drop_columns = [
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ]

    model_df = df.drop(
        columns=existing_drop_columns
    )

    print()
    print(
        "Dropped identifier columns:",
        existing_drop_columns
    )

    # ========================================================
    # TRAIN / TEMP
    # ========================================================

    train_df, temp_df = train_test_split(
        model_df,
        test_size=0.30,
        stratify=model_df[TARGET],
        random_state=RANDOM_STATE
    )

    # ========================================================
    # VALIDATION / TEST
    # ========================================================

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET],
        random_state=RANDOM_STATE
    )

    # ========================================================
    # SAVE
    # ========================================================

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("===== FINAL SPLIT =====")

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    print(
        "Total:",
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    print()
    print("===== FRAUD DISTRIBUTION =====")

    for name, split in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df)
    ]:

        fraud_count = int(
            split[TARGET].sum()
        )

        total = len(split)

        fraud_percentage = (
            fraud_count / total * 100
        )

        print()
        print(name)

        print(
            "Total:",
            total
        )

        print(
            "Fraud:",
            fraud_count
        )

        print(
            "Non-fraud:",
            total - fraud_count
        )

        print(
            "Fraud percentage:",
            round(
                fraud_percentage,
                2
            )
        )

    print()
    print("===== FILES =====")

    print(
        "Train:",
        train_path
    )

    print(
        "Validation:",
        val_path
    )

    print(
        "Test:",
        test_path
    )

    print()
    print("Dataset preparation complete.")

    print()
    print("==========================================")


if __name__ == "__main__":
    main()
