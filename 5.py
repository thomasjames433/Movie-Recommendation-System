import pandas as pd

# Load dataset
movies = pd.read_csv('final.csv')

# Filter condition
zero_vote_count = movies[(movies['vote_count'] < 50) & (movies['vote_average'] < 2.0)].shape
print("Number of movies with vote_average < 2.0 and vote_count < 50:", zero_vote_count[0])

# Cleaned DataFrame
movies_cleaned = movies[~((movies['vote_count'] < 50) & (movies['vote_average'] < 2.0))]

# Select only desired columns
selected_columns = ['id', 'title','overview','genres', 'keywords','imdb_id','popularity','director','cast']  # modify as needed
movies_selected = movies_cleaned[selected_columns]

# Save only selected columns to CSV
movies_selected.to_csv('final.csv', index=False)

print("Filtered data with selected columns saved back to final.csv")
print(len(movies))