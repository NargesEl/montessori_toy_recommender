from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
cs = cosine_similarity

df=pd.read_csv("df_final.csv")

def get_recommender(user_input):
    """ This function produces the 5 top recommendations depending on user_input. 
    The user_input is the age group and skills selected from the webpage. 
    It calculates the similarity of each row in the dataframe and the user_input. """
    
    all_scores = []
    for i, row in df.iterrows():
        row = row[5:].to_numpy().reshape(1, -1)
        all_scores.append((cs(row,user_input)).flatten())

    all_scores = pd.Series(np.array(all_scores).flatten(), index=df.index)
    top_5 = all_scores.sort_values(ascending=False).head().index
    print (type(top_5))

    result=[]

    for t in top_5:
        print(t)
        toy = df.loc[df.index == t]
        link = toy["link"].values
        image = toy["main_image_link"].values
        price = toy["price_value"].values
        title = toy["title"].values
        entry ={"title":title, "link":link, "image":image, "price":price}
        result.append(entry)
    return result
    
    


