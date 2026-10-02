import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pandas import DataFrame
from src.data_processing import extract_features


def plot_tsne(dataframe: DataFrame, target_column: str = 'syndrome_id', output_dir: str = 'outputs') -> None:
    """Reduces dimensionality using t-SNE, plots the embeddings, and saves the figure."""
    print("=== Generating 2D Visualization with t-SNE ===")
    print("This might take a few seconds...")
    os.makedirs(output_dir, exist_ok=True)

    X = extract_features(dataframe)
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_reduced = tsne.fit_transform(X)

    df_plot = dataframe.copy()
    df_plot['Dimension 1'] = X_reduced[:, 0]
    df_plot['Dimension 2'] = X_reduced[:, 1]

    ordered_syndromes = dataframe[target_column].value_counts().index.tolist()
    num_classes = len(ordered_syndromes)
    distinct_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    if num_classes <= len(distinct_markers):
        markers = distinct_markers[:num_classes]
    else:
        markers = True

    plt.figure(figsize=(12, 8))

    # https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    sns.scatterplot(
        x='Dimension 1',
        y='Dimension 2',
        hue=target_column,
        hue_order=ordered_syndromes,
        style=target_column,
        style_order=ordered_syndromes,
        markers=markers,
        palette='tab10',
        data=df_plot,
        legend='full',
        alpha=0.7
    )

    plt.title('Embedding Patterns Visualization by Syndrome (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Syndrome ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, 'tsne_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[+] t-SNE plot exported to: {output_path}\n")
    plt.close()