import os
import pandas as pd
from pandas import DataFrame, Series

def generate_eda_report(dataframe: DataFrame, output_dir: str = 'outputs') -> None:
    """Gera estatísticas consolidadas, análises de consistência e exporta tabelas."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Visão Geral do Dataset ===")
    total_syndromes: int = dataframe['syndrome_id'].nunique()
    total_subjects: int = dataframe['subject_id'].nunique()
    total_images: int = dataframe['image_id'].nunique()
    
    print(f"Total de síndromes únicas: {total_syndromes}")
    print(f"Total de sujeitos únicos: {total_subjects}")
    print(f"Total de imagens únicas: {total_images}")
    
    print("\n=== Distribuição por Síndrome ===")
    
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
    
    # Formatação para exibição no terminal
    display_df: DataFrame = summary_df.copy()
    display_df['%'] = display_df['%'].map('{:.2f}'.format)
    display_df['img/subject'] = display_df['img/subject'].map('{:.2f}'.format)
    
    print(display_df.to_string())
    
    # Salvando em CSV
    csv_path: str = os.path.join(output_dir, 'syndrome_distribution.csv')
    summary_df.to_csv(csv_path, index_label='syndrome_id')
    print(f"\n[+] Tabela exportada para: {csv_path}")

    print("\n=== Balanceamento de Classes ===")
    majority_images: int = summary_df['images'].max()
    minority_images: int = summary_df['images'].min()
    class_ratio: float = majority_images / minority_images
    print(f"Razão maior/menor classe: {class_ratio:.2f}")

    print("\n=== Consistência de Sujeitos ===")
    images_per_subject: Series[int] = dataframe['subject_id'].value_counts()
    multiple_images_count: int = (images_per_subject >= 2).sum()
    
    syndromes_per_subject: Series[int] = dataframe.groupby('subject_id')['syndrome_id'].nunique()
    multiple_syndromes_count: int = (syndromes_per_subject > 1).sum()
    
    print(f"Sujeitos com 2 ou mais imagens: {multiple_images_count}")
    print(f"Sujeitos associados a mais de uma síndrome: {multiple_syndromes_count}")
    print()