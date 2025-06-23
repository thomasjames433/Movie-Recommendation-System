import pandas as pd

# Load the data
names_df = pd.read_csv("tconst_names.csv")
names_df.to_csv('names_df.csv',index=False)
movies_df = pd.read_csv("filtered_movies.csv")

# Filter categories
actors_df = names_df[names_df['category'].isin(['actor', 'actress','self'])]
directors_df = names_df[names_df['category'] == 'director']

# Only keep actors with ordering <= 8
actors_df = actors_df[actors_df['ordering'] <= 8].copy()
actors_df = actors_df[actors_df['name'].notna()]

# Map ordering to simplified values
def simplify_order(o):
    if 1 <= o <= 5:
        return 3
    elif 6 <= o <= 8:
        return 1
    return None

actors_df['order_group'] = actors_df['ordering'].apply(simplify_order)

# Remove any rows where order_group wasn't assigned
actors_df = actors_df[actors_df['order_group'].notnull()]

# Build (name, simplified_order) tuples
actors_df['name_order'] = list(zip(actors_df['name'].str.lower().str.replace(" ", "", regex=False), actors_df['order_group']))

# Group crew info
crew_info = actors_df.groupby('tconst')['name_order'].apply(list).rename('cast')

# Remove rows with missing director names
directors_df = directors_df[directors_df['name'].notna()]

# Clean and normalize director names (lowercase + remove spaces)
directors_df['name_clean'] = directors_df['name'].str.lower().str.replace(" ", "", regex=False)

# Get the first director per movie (lowest ordering)
first_director = directors_df.sort_values(['tconst', 'ordering']).drop_duplicates('tconst')

# Set as Series: tconst → cleaned name
first_director_info = first_director.set_index('tconst')['name_clean'].rename('director')


# Merge into final dataframe
final_df = movies_df.set_index('imdb_id').join(first_director_info).join(crew_info)

final_df = final_df.reset_index()

# Output result
final_df.to_csv("final.csv", index=False)
