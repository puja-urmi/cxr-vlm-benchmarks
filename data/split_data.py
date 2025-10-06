import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your main dataset
csv_path = '/home/psaha03/scratch/complete_dataset/indiana_reports_complete.csv'
train_path = '/home/psaha03/scratch/complete_dataset/indiana_train.csv'
val_path = '/home/psaha03/scratch/complete_dataset/indiana_val.csv'
test_path = '/home/psaha03/scratch/complete_dataset/indiana_test.csv'

def split_and_save():
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    split_and_save()
