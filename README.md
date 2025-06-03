# TransformingFashion
Research &amp; Improvement of the OutfitTransformer

# Enironment for Outfit Transformer

python -m venv outfit_transformer_env
source outfit_transformer_env/bin/activate
pip install -r requirements/requirements_model.txt

python -m venv hdp_env
source hdp_env/bin/activate
pip install -r requirements/requirements_clustering.txt
# hay que ir al archivo:
nano ~/Documentos/TFG_Lara/TransformingFashion/hdp_env/lib/python3.12/site-packages/gensim/matutils.py

y cambiar la linea:
from scipy.linalg import get_blas_funcs, triu (quitar triu)

y meter:
from numpy import triu
