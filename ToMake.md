Keep adding cosine similarites to each other  
Append the first one to a final one then make the second, apend it to final, make the third append it to final adding weights as we append
if we dont find a matching one just append it to the end, or else leave it, we just need the top 100 matching max anyways

For user personal reomnedations
make a vector with constant magnitude keep averageing all teh new liked movies vectors
Try to store it as a json if not possible to have space for all users
If he likes a movie with say action and romance multiply the current vectors with the no he has already liked
add the value for action, if not romance add romance to th vector and then store it

Keep a vector like 
keyword1 -movie A
keyword1 movie B
keyword 2 movie A
keyword 2 movie C
If space allows or do the same as in prev step for user and  movie

Recommend movies of a language only if they have watched it before since its just ml and en set bool values in the database

