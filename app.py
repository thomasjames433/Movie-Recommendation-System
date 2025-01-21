import streamlit as st
import pickle 
import requests

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=9efeaf04e2ff5b4f90e5c485641c4791&language=en-US'.format(movie_id))
    data= response.json()
    # st.text(data)
    # st.text('https://api.themoviedb.org/3/movie/{}?api_key=9efeaf04e2ff5b4f90e5c485641c4791&language=en-US')
    # print(data)
    return "https://image.tmdb.org/t/p/original/" +data['poster_path']


def recommend(movie):
    movie_index=movies_list[movies_list['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_got=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    recommend_movies=[]
    posters=[]
    for i in movies_got:
        recommend_movies.append(movies_list.iloc[i[0]].title)
        # fetch poster from API, movie id is i[0]
        posters.append(fetch_poster(movies_list.iloc[i[0]].movie_id))
    return recommend_movies,posters
    

movies_list= pickle.load(open('movies.pkl','rb'))
similarity=pickle.load(open('similarity.pkl','rb'))

st.title('Movie Recommender System')

chosen = st.selectbox(
    "Search a movie",
    (movies_list['title'].values),
)

if st.button("Reccomend"):
    names,posters=recommend(chosen)
    import streamlit as st

    col1, col2, col3, col4, col5 = st.columns(5)


    for col, poster, name in zip([col1, col2, col3, col4, col5], posters, names):
        with col:
            st.image(poster)
            # Custom font size and alignment using Markdown
            st.markdown(f"<h6 style='text-align: center; font-size:14px;'>{name}</h6>", unsafe_allow_html=True)