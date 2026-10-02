import os
import pandas as pd
from pandas import DataFrame, Series

def generate_eda_report(dataframe: DataFrame, output_dir: str = 'outputs') -> None:
    """Generates consolidated statistics, consistency analysis, and exports tables."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Dataset Overview ===")
    total_syndromes: int = dataframe['syndrome_id'].nunique()
    total_subjects: int = dataframe['subject_id'].nunique()
    total_images: int = dataframe['image_id'].nunique()
    
    print(f"Total unique syndromes: {total_syndromes}")
    print(f"Total unique subjects: {total_subjects}")
    print(f"Total unique images: {total_images}")
    
    print("\n=== Syndrome Distribution ===")
    images_count: Series[int] = dataframe['syndrome_id'].value_counts()
    percentage: Series[float] = dataframe['syndrome_id'].value_counts(normalize=True) * 100
    subjects_count: Series[int] = dataframe.groupby('syndrome_id')['subject_id'].nunique()
    
    summary_df: DataFrame = pd.DataFrame({
        'images': images_count,
        '%': percentage,
        'subjects': subjects_count
    })
    
    summary_df['img/subject'] = summary_df['images'] / summary_df['subjects']
    summary_df = summary_df.sort_values(by='images', ascending=False)
    
    # Format for terminal display
    display_df: DataFrame = summary_df.copy()
    display_df['%'] = display_df['%'].map('{:.2f}'.format)
    display_df['img/subject'] = display_df['img/subject'].map('{:.2f}'.format)
    
    print(display_df.to_string())
    
    # Save to CSV
    csv_path: str = os.path.join(output_dir, 'syndrome_distribution.csv')
    summary_df.to_csv(csv_path, index_label='syndrome_id')
    print(f"\n[+] Table exported to: {csv_path}")

    print("\n=== Class Balancing ===")
    majority_images: int = summary_df['images'].max()
    minority_images: int = summary_df['images'].min()
    class_ratio: float = majority_images / minority_images
    print(f"Majority/minority class ratio: {class_ratio:.2f}")

    print("\n=== Subject Consistency ===")
    images_per_subject: Series[int] = dataframe['subject_id'].value_counts()
    multiple_images_count: int = (images_per_subject >= 2).sum()
    
    syndromes_per_subject: Series[int] = dataframe.groupby('subject_id')['syndrome_id'].nunique()
    multiple_syndromes_count: int = (syndromes_per_subject > 1).sum()
    
    print(f"Subjects with 2 or more images: {multiple_images_count}")
    print(f"Subjects associated with multiple syndromes: {multiple_syndromes_count}")
    print()