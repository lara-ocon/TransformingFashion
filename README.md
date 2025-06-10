
# OutfitTransformer TFG Implementation

This section provides a high-level overview of the implementation for the Trabajo de Fin de Grado (TFG) project, which focuses on fashion outfit compatibility using the OutfitCLIPTransformer model. The project implements two tasks: Compatibility Prediction (CP) and Fill-In-The-Blank (FITB), leveraging the Polyvore dataset and Fashion-CLIP embeddings. Below, we detail the environment setup, dataset preparation, embedding generation, training procedures, and clustering for curriculum learning.

## Environment Setup and Dataset Preparation

The project uses two separate Python virtual environments to manage dependencies and avoid conflicts:

- **Main Environment (`outfit_transformer_env`)**: Contains libraries for training the OutfitCLIPTransformer model.
  ```bash
  python -m venv outfit_transformer_env
  source outfit_transformer_env/bin/activate
  pip install -r requirements/requirements_model.txt
  ```

- **Clustering Environment (`hdp_env`)**: Dedicated to K-means and Hierarchical Dirichlet Process (HDP) clustering experiments, due to compatibility issues with the `gensim` library.
  ```bash
  python -m venv hdp_env
  source hdp_env/bin/activate
  pip install -r requirements/requirements_clustering.txt
  ```

For `gensim` in `hdp_env`, a manual patch is required in `gensim/matutils.py` to fix a runtime error:
- Replace the import of triu in line `from scipy.linalg import get_blas_funcs, triu` with:
  ```python
  from numpy import triu
  ```

The Polyvore dataset is downloaded and extracted using `datasetutils/download_data.py` or manually:
```bash
mkdir -p datasets
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu -O polyvore.zip
unzip polyvore.zip -d ./datasets/polyvore
rm polyvore.zip
```

The `datasets/polyvore/` directory contains:
- `images/`: Item images named `item_id.jpg`.
- `item_metadata.json` and `item_title.json`: Item metadata including category, subcategory, descriptions, and titles.
- `disjoint/` and `nondisjoint/`: Data splits for training, validation, and test. This project uses the `nondisjoint` split, allowing item overlap for realistic evaluation.

Task-specific data (`compatibility/` and `fill_in_the_blank/`) is organized in `train.json`, `valid.json`, and `test.json`:
- **Compatibility**: Outfits with binary labels (0: incompatible, 1: compatible).
- **Fill-In-The-Blank (FITB)**: Incomplete outfits, four candidate items, the removed item’s position, and the correct candidate index.

## CLIP Embedding Generation

Fashion items are represented as 1024-dimensional multimodal embeddings using the pretrained `patrickjohncyh/fashion-clip` model (CLIP ViT-B/32), combining image and text features. This process is implemented in `notebooks/generate_clip_embeddings.ipynb`.

**Main Steps:**
- Load item metadata from `item_metadata.json`.
- Resize images (`images/item_id.jpg`) to 224x224 and preprocess them.
- Process images and text descriptions through FashionCLIP to obtain 512-dimensional image and text embeddings.
- Concatenate embeddings to form a 1024-dimensional vector per item.

Embeddings are computed in batches using PyTorch’s `DataLoader`, with optional distributed processing across GPUs via `torch.distributed`. Results are saved as `.pkl` files in `datasets/polyvore/precomputed_clip_embeddings`.

## Training Procedure for Compatibility Prediction (CP)

The CP task predicts whether an outfit is compatible (label = 1) or incompatible (label = 0) using a transformer-based architecture.

### Dataset and Dataloaders
The `nondisjoint/compatibility/train.json` dataset provides outfits (lists of item IDs) with binary labels. Precomputed FashionCLIP embeddings serve as item representations. The `PolyvoreCompatibilityDataset` class returns a list of `FashionItem` objects (containing metadata and 1024-dimensional embeddings) and their label. The dataloader’s `cp_collate_fn` returns batches of:
- `query`: List of outfits (each a list of `FashionItem`).
- `label`: List of binary labels.

### Model Architecture
The `OutfitCLIPTransformer` processes outfits as unordered sets:
- Projects 1024-dimensional CLIP embeddings to 128 dimensions using a linear layer.
- Prepends a learnable CLS token to the sequence.
- Applies 4 transformer encoder layers with 4 attention heads and 0.1 dropout.
- Feeds the CLS token embedding to a single-layer MLP classifier, outputting a logit for compatibility.

### Loss Function and Optimization
Training uses Focal Loss to focus on hard examples, with focusing parameter γ=2 and class balancing factor α=0.5. The AdamW optimizer with weight decay is paired with a OneCycleLR scheduler (max learning rate = 2e-5). Gradient clipping (max norm = 1.0) and 4-step gradient accumulation simulate a larger batch size.

### Training and Validation
Training runs for 200 epochs, saving checkpoints per epoch. Metrics (Accuracy, AUC, F1, Precision, Recall) are evaluated on the validation set, and the best AUC checkpoint is selected for testing and FITB pretraining.

## Cluster Generation for Curriculum Learning

The FITB task uses curriculum learning to sample increasingly difficult negatives, requiring item clustering based on similarity.

### K-Means Clustering
K-Means is applied per subcategory (`category_id`) within each semantic category (`semantic_category`), ensuring clusters group similar items (e.g., blouses or sneakers). The number of clusters \( k \) is set as:
\[ k = \max(5, \lfloor N / 300 \rfloor) \]
where \( N \) is the number of items in the subcategory. Subcategories with fewer than 10 items are skipped. Cluster assignments are saved as CSV files and parsed into dictionaries:
- `item_to_info`: Maps `item_id` to `semantic_category`, `category_id`, `cluster_id`.
- `cluster_to_items`: Maps `(semantic_category, category_id, cluster_id)` to item IDs.
- `category_to_clusters`: Maps `(semantic_category, category_id)` to cluster IDs.

These are serialized as `.pkl` files for use in FITB training.

### HDP Clustering
Hierarchical Dirichlet Process (HDP) clustering discovers latent topics from item text descriptions (concatenated metadata fields: title, description, categories, url_name, related). Outfit descriptions are aggregated into documents, preprocessed, and fed to a Gensim HDP model. Each item is assigned to the topic with the highest average probability across its outfits, with unassigned items grouped into a fallback cluster (-1). Assignments are saved as CSV files and converted to the same dictionary structure as K-Means.

## Training Procedure for Fill-In-The-Blank (FITB)

The FITB task ranks candidate items to complete an incomplete outfit, fine-tuning the best CP checkpoint with curriculum learning.

### Dataset and Dataloaders
Training uses `nondisjoint/train.json`, removing one item per outfit to create a query and positive answer. Validation and testing use `nondisjoint/fill_in_the_blank/valid.json` and `test.json`, providing queries, four candidates, and the correct index. `PolyvoreTripletDataset` (training) returns queries and positives, while `PolyvoreFillInTheBlankDataset` (validation) returns queries, candidates, and labels. Dataloaders return:
- Training: `query` (outfit items), `answer` (positive item).
- Validation: `query`, `candidates`, `label` (correct index).

### Model Architecture
The `OutfitCLIPTransformer` extracts 128-dimensional CLS token embeddings for queries and candidates, computing Euclidean distances to rank candidates (smallest distance wins). The classification head is unused.

### Loss Function and Optimization
Training uses in-batch triplet margin loss to ensure the positive item’s embedding is closer to the query than negatives, with margin = 2.0. AdamW optimizer, OneCycleLR scheduler (max learning rate = 2e-5), 4-step gradient accumulation, and gradient clipping (max norm = 1.0) are applied.

### Training and Validation
Training runs for 200 epochs, fine-tuning the CP checkpoint. Curriculum learning adjusts negative sampling:
- **Epochs 0–39**: In-batch negatives (other batch positives).
- **Epochs 40–49**: Negatives from the same semantic category.
- **Epochs ≥ 50**: Negatives from the same subcategory but different K-Means clusters, using `item_to_info`, `cluster_to_items`, and `category_to_clusters`.

The `sample_negatives` function dynamically samples negatives. Validation computes distances to candidates, selecting the minimum. Accuracy measures correct predictions, with the best validation accuracy checkpoint used for testing.

### Metrics
Accuracy is the primary metric, with precision and recall computed during testing to assess ranking quality.

## References
- [1] Wonjun Oh. *OutfitTransformer: Learning Outfit Representations for Fashion Recommendation*. GitHub repository, 2023. [Link](https://github.com/owj0421/outfit-transformer)
- [2] Han, X., et al. *Learning Fashion Compatibility with Bidirectional LSTMs*. arXiv:1707.05691, 2017. [Link](https://arxiv.org/abs/1707.05691)
- [3] Sarkar, P., et al. *OutfitTransformer: Learning Outfit Representations for Fashion Recommendation*. WACV, 2023.
- [4] Cucurull, G., et al. *Context-Aware Visual Compatibility Prediction*. CVPR, 2019.
- [5] Kocik, M. *Recommender Systems for Fashion Outfits*. Bachelor’s thesis, Charles University, 2020.
