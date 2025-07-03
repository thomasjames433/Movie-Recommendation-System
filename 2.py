import pandas as pd
from tqdm import tqdm # to track progress
df=pd.read_csv('filtered_movies.csv')

imdb_ids=set(df['imdb_id'])
jobs=['actress','actor','director','self']
output_file='filtered_principals.csv'
x=1
with open (output_file,'w',encoding='utf-8')as out_f:
    header_written=False

    for chunk in pd.read_csv('title.principals.csv', chunksize=100000, dtype=str):
        filtered_chunk = chunk[(chunk['tconst'].isin(imdb_ids)) & (chunk['category'].isin(jobs)) ]
        filtered_chunk=filtered_chunk[['tconst','ordering','nconst','category']]
        print(x)
        x+=1
        # Append to output
        if not filtered_chunk.empty:
            filtered_chunk.to_csv(out_f, index=False, header=not header_written, mode='a')
            header_written = True