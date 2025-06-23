import pandas as pd

principals=pd.read_csv('filtered_principals.csv')
names=pd.read_csv('name.basics.csv')

merge_df=principals.merge(names,on='nconst',how='left')

merge_df['nconst']=merge_df['primaryName'].combine_first(merge_df['nconst'])

merge_df.drop(columns=['primaryName','job','birthYear','deathYear','primaryProfession','knownForTitles'],inplace=True)
merge_df.rename(columns={'nconst': 'name'}, inplace=True)
merge_df.to_csv('tconst_names.csv', index=False)
