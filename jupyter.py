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

# Same for overview
movies['overview'] = movies['overview'].fillna('').astype(str).apply(
    lambda x: re.sub(r'[^\w\s]', '', x).lower()
)

# Same for genres
movies['genres'] = movies['genres'].fillna('').astype(str).apply(
    lambda x: re.sub(r'[^\w\s]', '', x).lower()
)

print("HIHI")

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    string=" ".join(l)
    return string

title=movies[['id','title']]
# title['title']=title['title'].apply(lambda x: x.split())
overview=movies[['id','overview']]
genres=movies[['id','genres']]
keywords=movies[['id','keywords']]
director=movies[['id','director']]
cast=movies[['id','cast']]


print("Started Stemming")
genres.loc[:, 'genres'] = genres['genres'].apply(stem)

overview.loc[:, 'overview'] = overview['overview'].apply(stem)

keywords.loc[:, 'keywords'] = keywords['keywords'].apply(stem)
print("Finished Stemming")


from sklearn.feature_extraction.text import CountVectorizer

print("Start CV")
cv_genres=CountVectorizer(stop_words='english')
genres_vectors=cv_genres.fit_transform(genres['genres'])


cv_keywords=CountVectorizer(stop_words='english')
keywords_vectors=cv_keywords.fit_transform(keywords['keywords'])


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
# Get the feature names (i.e. the vocabulary)
feature_names = cv_cast.get_feature_names_out()

# Convert the first row to a dense array
first_row_vector = cast_vectors[0].toarray()[0]

# Combine feature names with counts
first_row_vocab = dict(zip(feature_names, first_row_vector))

print("Finish Stemming")

import numpy as np
import faiss
from sklearn.decomposition import TruncatedSVD

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

genres_vectors   = maybe_reduce_vector(genres_vectors, 'genres', 21)
keywords_vectors = maybe_reduce_vector(keywords_vectors, 'keywords', 13000)
overview_vectors = maybe_reduce_vector(overview_vectors, 'overview', 9000)
cast_vectors     = maybe_reduce_vector(cast_vectors, 'cast', 11000)
director_vectors = maybe_reduce_vector(director_vectors, 'director', 1000)

print("Finish reduce vectors")

# ---- STEP 2: Apply weights ----
genres_vectors   = genres_vectors * 10
keywords_vectors = keywords_vectors * 6
overview_vectors = overview_vectors * 6
# cast_vectors: unweighted
director_vectors = director_vectors * 4

# ---- STEP 3: Combine all vectors horizontally ----
combined_vectors = np.hstack([
    genres_vectors,
    keywords_vectors,
    overview_vectors,
    cast_vectors,
    director_vectors
])


# ---- STEP 4: Normalize for cosine similarity ----
combined_norm = normalize(combined_vectors)

print("Normalized combined norm")


# ---- STEP 5: Build FAISS index ----
index = faiss.IndexFlatIP(combined_norm.shape[1])
index.add(combined_norm)
print("Index added")
# ---- STEP 6: Search ----
k = 1500
D, I = index.search(combined_norm, k)

# ---- STEP 7: Save ----
np.save('similar_indices.npy', I)
np.save('similar_scores.npy', D)

print("✅ Done. Results saved.")
