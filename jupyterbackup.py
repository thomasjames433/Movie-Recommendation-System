import numpy as np 
import pandas as pd

movies=pd.read_csv('final.csv')
movies[movies.duplicated(subset='id', keep=False)]
movies.drop_duplicates(subset='id', keep='first', inplace=True)

import re

# Clean and lowercase keywords, keep as string
movies['keywords'] = movies['keywords'].fillna('').astype(str).apply(
    lambda x: re.sub(r'[^\w\s]', '', x).lower()
)

'''
# Same for overview
movies['overview'] = movies['overview'].fillna('').astype(str).apply(
    lambda x: re.sub(r'[^\w\s]', '', x).lower()
)

# Same for genres
movies['genres'] = movies['genres'].fillna('').astype(str).apply(
    lambda x: re.sub(r'[^\w\s]', '', x).lower()
)
'''
print("HIHI")

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    string=" ".join(l)
    return string
'''
title=movies[['id','title']]
# title['title']=title['title'].apply(lambda x: x.split())
overview=movies[['id','overview']]
genres=movies[['id','genres']]
'''
keywords=movies[['id','keywords']]
'''
director=movies[['id','director']]
cast=movies[['id','cast']]


print("Started Stemming")
genres.loc[:, 'genres'] = genres['genres'].apply(stem)

overview.loc[:, 'overview'] = overview['overview'].apply(stem)
'''
keywords.loc[:, 'keywords'] = keywords['keywords'].apply(stem)
print("Finished Stemming")


from sklearn.feature_extraction.text import CountVectorizer
'''

print("Start CV")
cv_genres=CountVectorizer(stop_words='english')
genres_vectors=cv_genres.fit_transform(genres['genres'])
'''

cv_keywords=CountVectorizer(stop_words='english')
keywords_vectors=cv_keywords.fit_transform(keywords['keywords'])

'''
cv_overview=CountVectorizer(stop_words='english')
overview_vectors=cv_overview.fit_transform(overview['overview'])

director = director.copy()
director['director'] = director['director'].fillna('unknown_director')

cv_director = CountVectorizer(stop_words='english')
director_vectors = cv_director.fit_transform(director['director'])
director_vectors = director_vectors.tolil()   #: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
unknown_index=cv_director.vocabulary_.get('unknown_director')

if unknown_index is not None:
    director_vectors=director_vectors.copy()
    director_vectors[:,unknown_index]=0
director_vectors = director_vectors.tocsr()       # Convert back to CSR for fast computations

cast=cast.copy()
cast['cast'] = cast['cast'].fillna('[]')
import ast

cast.loc[:, 'cast'] = cast['cast'].fillna('[]')

# Step 1: Convert cast strings to actual lists of tuples
cast.loc[:, 'cast'] = cast['cast'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# Step 2: Expand and convert to space-separated string
def expand_cast_to_string(cast_list):
    if not isinstance(cast_list, list):
        return ''
    expanded = [name for name, count in cast_list if isinstance(name, str) and isinstance(count, int) for _ in range(count)]
    return ' '.join(expanded)


# Step 3: Apply the function
cast.loc[:, 'cast'] = cast['cast'].apply(expand_cast_to_string)
cast.loc[:, 'cast'] = cast['cast'].str.replace('-', '', regex=False)


cv_cast=CountVectorizer(stop_words='english')
cast_vectors=cv_cast.fit_transform(cast['cast'])
'''
print("Finish Stemming")


import faiss
from sklearn.decomposition import TruncatedSVD
import gc
# ---- Helper Functions ----

def maybe_reduce_vector(x, name="vector", n_components=128):
    """Apply TruncatedSVD only if input is high-dimensional (and sparse)."""
    original_dim = x.shape[1]

    if n_components < original_dim:
        print(f"Reducing {name} from {original_dim} → {n_components} dimensions")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        x_reduced = svd.fit_transform(x)
        return x_reduced.astype(np.float32)
    else:
        print(f"Skipping reduction for {name} (dim = {original_dim})")
        return x.toarray().astype(np.float32)  # safe to convert small ones

def normalize(vectors):
    """Normalize vectors row-wise for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

# ---- STEP 1: Load and Reduce Vectors ----

keywords_vectors   = maybe_reduce_vector(keywords_vectors, 'keywords', 2000)

combined_norm = normalize(keywords_vectors)
combined_norm*=2.624
np.savez('/dev/shm/individual_keywords.npz', vectors=combined_norm)

import gzip
import shutil
import os

os.makedirs('keywords', exist_ok=True)

with open('/dev/shm/individual_keywords.npz', 'rb') as f_in:
    with gzip.open('keywords/individual_keywords.npz.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove('/dev/shm/individual_keywords.npz')
print("Normalized combined norm keywords")


# ---- STEP 5: Build FAISS index ----
index = faiss.IndexFlatIP(combined_norm.shape[1])
index.add(combined_norm)
print("Index added")
# ---- STEP 6: Search ----
k =3500
D, I = index.search(combined_norm, k)

# ---- STEP 7: Save ----
np.savez('/dev/shm/keywords_indices.npz', I)
np.savez('/dev/shm/keywords_scores.npz', D)

'''
with open('/dev/shm/keywords_scores.npz', 'rb') as f_in:
    with gzip.open('keywords/keywords_scores.npz.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with open('/dev/shm/keywords_indices.npz', 'rb') as f_in:
    with gzip.open('keywords/keywords_indices.npz.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# ---- Optional: remove uncompressed files ----
os.remove('/dev/shm/keywords_scores.npz')
os.remove('/dev/shm/keywords_indices.npz')
'''
print("✅  Done. Results saved.")
