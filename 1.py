import pandas as pd

df=pd.read_csv('data.csv')

target_languages = ['en',  'ml', 'hi']

# Filter rows where the original_language is in the target list
filtered_df = df[
                    (df['original_language'].isin(target_languages)) & 
                    (df['adult']==False) & 
                    (df['release_date'].str[0:4].astype('Int64')>1969)
                ]

# Select the desired columns
columns_to_keep = [
    'id', 'title',  'release_date',
    'runtime', 'original_language', 'overview',
    'poster_path', 'genres', 'keywords','imdb_id',
    'vote_average', 'vote_count','revenue','budget','popularity'
]

df['genres'] = df['genres'].fillna('').apply(lambda x: [genre.strip() for genre in x.split(',') if genre.strip()])
df['keywords'] = df['keywords'].fillna('').apply(lambda x: [kw.strip() for kw in x.split(',') if kw.strip()])

result_df = filtered_df[columns_to_keep]

# Save to new CSV
result_df.to_csv('filtered_movies.csv', index=False)

print(f"Filtered {len(result_df)} movies. Saved to 'filtered_movies.csv'.")

# import pandas as pd

# df=pd.read_csv('data.csv')
# x=0

# languages=[]
# for _,row in df.iterrows():
#     language=row['original_language']
#     if language not in languages:
#         x+=1
#         print(x)
#         print(language)
#         languages.append(language)
#     # if language=='en':
#     #     x+=1
# print(x)
# print (languages)

