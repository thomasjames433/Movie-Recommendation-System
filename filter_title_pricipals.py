import pandas as pd
from collections import defaultdict

# File paths
input_file = 'unfiltered.title.principals.csv'
output_file = 'title.principals.csv'

# Columns to keep
columns_to_keep = ['tconst', 'ordering', 'nconst', 'category']

# Track seen (tconst, nconst) pairs
seen_pairs = set()

# Track ordering per tconst
tconst_order_counter = defaultdict(int)

# Chunk size
chunk_size = 100_00

# Control header write
write_header = True

i=0
# Process in chunks
for chunk in pd.read_csv(input_file, usecols=columns_to_keep + ['job', 'characters'], chunksize=chunk_size):
    # Drop unnecessary columns early
    print(i)
    i += 1
    chunk = chunk[columns_to_keep]
    
    # Filter out rows with already-seen (tconst, nconst) pairs
    def is_new_pair(row):
        pair = (row['tconst'], row['nconst'])
        if pair in seen_pairs:
            return False
        seen_pairs.add(pair)
        return True

    chunk = chunk[chunk.apply(is_new_pair, axis=1)]
    
    # Reassign ordering per tconst
    def assign_order(row):
        tconst = row['tconst']
        tconst_order_counter[tconst] += 1
        return tconst_order_counter[tconst]
    
    chunk['ordering'] = chunk.apply(assign_order, axis=1)
    
    # Write to output file
    chunk.to_csv(output_file, mode='a', index=False, header=write_header)
    write_header = False
