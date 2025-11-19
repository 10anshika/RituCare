import pandas as pd
import numpy as np
import os
from typing import List

def load_public_datasets() -> pd.DataFrame:
    """
    Load menstrual_cycle_data.csv and menstrual_cycle_dataset_with_factors.csv from dataset folder.
    Merge on ClientID if exists, otherwise concat.
    Returns the combined DataFrame.
    """
    file1 = os.path.join('dataset', 'menstrual_cycle_data.csv')
    file2 = os.path.join('dataset', 'menstrual_cycle_dataset_with_factors.csv')

    df1 = pd.read_csv(file1, low_memory=False) if os.path.exists(file1) else pd.DataFrame()
    df2 = pd.read_csv(file2, low_memory=False) if os.path.exists(file2) else pd.DataFrame()

    print(f"Loaded menstrual_cycle_data.csv: {len(df1)} rows")
    print(f"Loaded menstrual_cycle_dataset_with_factors.csv: {len(df2)} rows")

    if df1.empty and df2.empty:
        print("No datasets loaded.")
        return pd.DataFrame()

    if not df1.empty and not df2.empty and 'ClientID' in df1.columns and 'ClientID' in df2.columns:
        combined_df = pd.merge(df1, df2, on='ClientID', how='outer')
        print("Merged datasets on ClientID.")
    else:
        combined_df = pd.concat([df1, df2], ignore_index=True)
        print("Concatenated datasets.")

    return combined_df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame: standardize columns, drop unrelated, create features, handle missing.
    """
    if df.empty:
        print("DataFrame is empty, skipping preprocessing.")
        return df

    # Standardize column names
    column_mapping = {
        'User_ID': 'ClientID',
        'Client_ID': 'ClientID',
        'Cycle_Number': 'CycleNum',
        'Cycle_Num': 'CycleNum',
        'Cycle_Length': 'LengthofCycle',
        'Length_of_Cycle': 'LengthofCycle',
        'Mean_Cycle_Length': 'MeanCycle',
        'Mean_Cycle': 'MeanCycle',
        'Total_Number_of_Days': 'TotalDays',
        'Total_Days': 'TotalDays',
        'Start_Date_of_Cycle': 'FirstDayOfCycle',
        'First_Day_of_Cycle': 'FirstDayOfCycle',
        # Add more if needed
    }
    df.rename(columns=column_mapping, inplace=True)
    print("Standardized column names.")

    # Drop unrelated survey-type fields
    unrelated_fields = [
        'marriage', 'breastfeeding', 'education', 'age', 'weight', 'height', 'occupation', 'income', 'location',
        'ethnicity', 'religion', 'smoking', 'alcohol', 'exercise', 'diet', 'stress', 'anxiety', 'depression',
        'medication', 'hormonal_contraceptives', 'pcos_diagnosis', 'endometriosis', 'fibroids', 'thyroid',
        'diabetes', 'hypertension', 'heart_disease', 'cancer', 'other_conditions', 'family_history_pcos',
        'family_history_endometriosis', 'family_history_fibroids', 'family_history_thyroid', 'family_history_diabetes',
        'family_history_hypertension', 'family_history_heart_disease', 'family_history_cancer', 'menarche_age',
        'parity', 'abortion', 'miscarriage', 'live_birth', 'breastfeeding_duration', 'weaning_age',
        'contraceptive_use', 'contraceptive_type', 'contraceptive_duration', 'last_contraceptive_use',
        'menopause', 'menopause_age', 'hysterectomy', 'oophorectomy', 'other_surgeries', 'current_medication',
        'past_medication', 'allergy', 'chronic_illness', 'mental_health', 'sleep_disorder', 'eating_disorder',
        'substance_abuse', 'domestic_violence', 'sexual_assault', 'trauma', 'support_system', 'healthcare_access',
        'insurance', 'doctor_visits', 'hospital_visits', 'emergency_visits', 'screening_tests', 'vaccinations',
        'health_goals', 'lifestyle_changes', 'motivation', 'barriers', 'support_needed'
    ]
    df.drop(columns=[col for col in unrelated_fields if col in df.columns], inplace=True, errors='ignore')
    print("Dropped unrelated survey fields.")

    # Keep only useful columns if they exist
    useful_cols = ['ClientID', 'CycleNum', 'LengthofCycle', 'MeanCycle', 'TotalDays', 'FirstDayOfCycle']
    existing_useful = [col for col in useful_cols if col in df.columns]
    df = df[existing_useful]
    print(f"Kept useful columns: {existing_useful}")

    # Parse FirstDayOfCycle to datetime if exists
    if 'FirstDayOfCycle' in df.columns:
        df['FirstDayOfCycle'] = pd.to_datetime(df['FirstDayOfCycle'], errors='coerce')
        print("Parsed FirstDayOfCycle to datetime.")

    # Create engineered features
    if 'LengthofCycle' in df.columns:
        df['cycle_length'] = df['LengthofCycle']
        df['cycle_variation'] = df.groupby('ClientID')['LengthofCycle'].transform('std')
        # avg_length: rolling mean if user has >1 row, else the value
        df['avg_length'] = df.groupby('ClientID')['LengthofCycle'].transform(
            lambda x: x.rolling(window=3, min_periods=2).mean() if len(x) > 1 else x
        )
        print("Created engineered features: cycle_length, cycle_variation, avg_length.")

    # Handle missing values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    print("Handled missing values with median for numeric columns.")

    return df

def save_processed(df: pd.DataFrame):
    """
    Save the processed DataFrame to dataset/processed_cycles.csv.
    """
    output_path = os.path.join('dataset', 'processed_cycles.csv')
    os.makedirs('dataset', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}.")

if __name__ == "__main__":
    df = load_public_datasets()
    df = preprocess(df)
    save_processed(df)
    print("✅ Preprocessing complete, rows:", len(df))
    print("Sample rows:")
    print(df.head())
