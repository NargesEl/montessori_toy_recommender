import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("amazon_data.csv")

df_cleaned = df.drop( axis=0, columns= ['bestsellers_rank_main_name', 'bestsellers_rank_main_rank', 'bestsellers_rank_sub_0_name', 'price_shipping', 'dimensions_cm_raw',
                           'price_per_unit',  'fulfillment_type', 'fulfillment_is_sold_by_amazon', 'attributes_variation_size_name', 'fulfillment_is_fulfilled_by_amazon', 'weight_tier',
                           'fulfillment_is_fulfilled_by_third_party','fulfillment_is_sold_by_third_party', 'price_buybox', 'price_rrp', 'weight_raw', 'weight_unit', 'weight_gramme',
                           'bestsellers_rank_sub_0_rank', 'labels_choice_keyword', 'labels_is_amazon_bestseller', 'labels_is_amazon_choice', 'offers_quantity', 'attributes_variation_style_name',
                           'bestsellers_rank_sub_1_name', 'bestsellers_rank_sub_1_rank', 'upc_number', 'ean_number', 'review_ratings_total', 'dimensions_type', 'dimensions_unit',
                           'dimensions_tier', 'part_number', 'parent_asin', 'variants_quantity', 'answered_questions_count', 'project_name', 'marketplace', 'attributes_variation_color_name',
                           'gtin_number','bestsellers_rank_sub_2_name', 'bestsellers_rank_sub_2_rank', 'estimations_sales_daily', 'dimensions_in_raw', 'attributes_variation_fit_type',
                           'estimations_sales_monthly', 'estimations_revenue_daily', 'model_number', 'keywords', 'review_rating', 'amazon_label', 'date_time', 'attributes_variation_number_of_items',
                           'estimations_revenue_monthly', 'estimations_revenue_per_review', 'price_symbol', 'aplus_feature', 'brand_name', 'seller_name', 'seller_id', 'is_used', 'asin',
                           'date_listed_iso', 'manufacturer', 'asins', 'images_count', 'review_by_stars_five_star', 'review_by_stars_four_star', 'review_by_stars_three_star', 'review_by_stars_two_star', 'review_by_stars_one_star'])

titles = df_cleaned["title"]
bullet1 = df_cleaned["features_bullet_point_0"].astype(str)
bullet2 = df_cleaned["features_bullet_point_1"].astype(str)
bullet3 = df_cleaned["features_bullet_point_2"].astype(str)
bullet4 = df_cleaned["features_bullet_point_3"].astype(str)
bullet5 = df_cleaned["features_bullet_point_4"].astype(str)
description = df_cleaned["features_description"].astype(str)

text_list = [titles, bullet1, bullet2, bullet3, bullet4, bullet5, description]


#Extracting Age:

def get_age_in_years(texts):
    """extracts numbers that occure with the words year(s), Year(s), Jahr(e), jahr(e) """
    age = r"(................[1234567890]................)"
    year_b = r"(................year................)"
    year_c = r"(................Year................)"
    year_d = r"(................jahr................)"
    year_e = r"(................Jahr................)"
    agey = []
    for n in texts:
        a = re.findall(age,n)
        text_y = ""
        if a:
            b = re.findall(year_b,n)
            if b:
                text_y = b[0]
            else:
                c = re.findall(year_c,n)
                if c:
                    text_y=c[0]
                else:
                    d = re.findall(year_d,n)
                    if d:
                        text_y = d[0]
                    else:
                        e = re.findall(year_e,n)
                        if e:
                            text_y = e[0]
        agey.append(text_y)
    return agey

def get_age_in_months(texts):
    """extracts numbers that occure with the words month(s), Month(s), monat(e), Monat(e) """
    age = r"..........[1234567890]............"
    month_b = r"(................month................)"
    month_c = r"(................Month................)"
    month_d = r"(................monat................)"
    month_e = r"(................Monat................)"
    agem = []
    for n in texts:
        a = re.findall(age,n)
        text = ""
        if a:
            b = re.findall(month_b,n)
            if b:
                text = b[0]
            else:
                c = re.findall(month_c,n)
                if c:
                    text=c[0]
                else:
                    d = re.findall(month_d,n)
                    if d:
                        text = d[0]
                    else:
                        e = re.findall(month_e,n)
                        if e:
                            text = e[0]
        agem.append(text)
    return agem

titles_agey = get_age_in_years(titles)
titles_agem = get_age_in_months(titles)
bullet1_agey = get_age_in_years(bullet1)
bullet1_agem = get_age_in_months(bullet1)
bullet2_agey = get_age_in_years(bullet2)
bullet2_agem = get_age_in_months(bullet2)
bullet3_agey = get_age_in_years(bullet3)
bullet3_agem = get_age_in_months(bullet3)
bullet4_agey = get_age_in_years(bullet4)
bullet4_agem = get_age_in_months(bullet4)
bullet5_agey = get_age_in_years(bullet5)
bullet5_agem = get_age_in_months(bullet5)
description_agey = get_age_in_years(description)
description_agem = get_age_in_months(description)

def get_age_in_numbers(texts):
    """extracts numbers without text """
    number = r"[^0-9]+(\b\d+\b)[^0-9]+(\b\d+\b)[^0-9]+(\b\d+\b)[^0-9]+(\b\d+\b).+"
    agen = []
    for n in texts:
        a = re.findall(number,n)
        text = ""
        if a:
            text = a[0]
        agen.append(set(text))
    return agen

titles_agey = get_age_in_numbers(titles_agey)
titles_agem = get_age_in_numbers(titles_agem)
bullet1_agey = get_age_in_numbers(bullet1_agey)
bullet1_agem = get_age_in_numbers(bullet1_agem)
bullet2_agey = get_age_in_numbers(bullet2_agey)
bullet2_agem = get_age_in_numbers(bullet2_agem)
bullet3_agey = get_age_in_numbers(bullet3_agey)
bullet3_agem = get_age_in_numbers(bullet3_agem)
bullet4_agey = get_age_in_numbers(bullet4_agey)
bullet4_agem = get_age_in_numbers(bullet4_agem)
bullet5_agey = get_age_in_numbers(bullet5_agey)
bullet5_agem = get_age_in_numbers(bullet5_agem)
description_agey = get_age_in_numbers(description_agey)
description_agem = get_age_in_numbers(description_agem)

df_cleaned["titles_agey"] = np.asarray(titles_agey)
df_cleaned["titles_agem"] = np.asarray(titles_agem)
df_cleaned["bullet1_agey"] = np.asarray(bullet1_agey)
df_cleaned["bullet1_agem"] = np.asarray(bullet1_agem)
df_cleaned["bullet2_agey"] = np.asarray(bullet2_agey)
df_cleaned["bullet2_agem"] = np.asarray(bullet2_agem)
df_cleaned["bullet3_agey"] = np.asarray(bullet3_agey)
df_cleaned["bullet3_agem"] = np.asarray(bullet3_agem)
df_cleaned["bullet4_agey"] = np.asarray(bullet4_agey)
df_cleaned["bullet4_agem"] = np.asarray(bullet4_agem)
df_cleaned["bullet5_agey"] = np.asarray(bullet5_agey)
df_cleaned["bullet5_agem"] = np.asarray(bullet5_agem)
df_cleaned["description_agey"] = np.asarray(description_agey)
df_cleaned["description_agem"] = np.asarray(description_agem)

df_age = df_cleaned.drop( axis=0, columns= ['features_bullet_point_0', 'features_bullet_point_1', 'features_bullet_point_2', 'features_bullet_point_3', 'features_bullet_point_4',
                                              'features_description', 'feature_bullets_count', 'images_row'])

list_years=[]

for i in range(len(df_age)):
    h = df_age.loc[i, "titles_agey"].union(df_age.loc[i, "bullet1_agey"]).union(df_age.loc[i, "bullet2_agey"]).union(df_age.loc[i, "bullet3_agey"]).union(df_age.loc[i, "bullet4_agey"]).union(df_age.loc[i, "bullet5_agey"]).union(df_age.loc[i, "description_agey"])
    list_years.append(h)
    
df_age["age_years"] = list_years

list_months=[]

for i in range(len(df_age)):
    h = df_age.loc[i, "titles_agem"].union(df_age.loc[i, "bullet1_agem"]).union(df_age.loc[i, "bullet2_agem"]).union(df_age.loc[i, "bullet3_agem"]).union(df_age.loc[i, "bullet4_agem"]).union(df_age.loc[i, "bullet5_agem"]).union(df_age.loc[i, "description_agem"])
    list_months.append(h)

df_age["age_years"] = list_years
df_age["age_months"] = list_months

df_age['new_age_in_months'] = df_age.age_years.apply(lambda z: set({int(x)*12 for x in z}))

df_age.age_months = df_age.age_months.apply(lambda y: set({int(x) for x in y}) )

new_row = []
for i,row in df_age.iterrows():
    new_row.append(row[-1].union(row[5]))

df_age['age'] = new_row

df_age = df_age.drop( axis=0, columns= ['bullet3_agey', 'bullet3_agem', 'bullet4_agey', 'bullet4_agem', 'bullet5_agey', 'bullet5_agem', 'bullet1_agey', 'bullet1_agem', 'titles_agey',
                                            'titles_agem','bullet2_agey', 'bullet2_agem', 'description_agey', 'description_agem'])

df_age['list_of_age'] = df_age.age.apply(lambda x: [int(y) for y in list(x)])

df_age['0_24'] = df_age.list_of_age.apply(lambda y : 1 if sum([1 if x >=0 and x <=24 else 0 for x in y])else 0)
df_age["25-60"]= df_age.list_of_age.apply(lambda y : 1 if sum([1 if x >=25 and x <=60 else 0 for x in y])else 0)
df_age['72+'] = df_age.list_of_age.apply(lambda y : 1 if sum([1 if x >=72 else 0 for x in y])else 0)

df_age.columns[5:]

df_age = df_age.drop(axis=0, columns=["list_of_age", "age_months", "new_age_in_months", "age", "age_years"])


# Extracting skills:

age = r"..........[1234567890]............"
motor = r"(motor)"
motor_c = r"(Motor)"
language = r"(language)"
language_c = r"(Language)"
math = r"(math)"
math_c = r"(Math)"
cognative = r"(cognative)"
cognative_c = r"(Cognative)"
numbers = r"(number)"
numbers_c = r"(Number)"
stem = r"(stem)"
stem_c = r"(Stem)"
stem_cc = r"(STEM)"
emotional = r"(emotional)"
emotional_c = r"(Emotional)"
social = r"(social)"
social_c = r"(Social)"

def get_skills(text,reg):
    skill = []
    for n in text:
        lines = ""
        a = re.findall(age,n)
        if a:
            b = re.findall(reg, n)
            if b:
                lines = b[0]
        skill.append(lines)
    return skill

for t in text_list:
    df_age["motor"] = get_skills(t, motor)
    df_age["motor_c"]= get_skills(t, motor_c)
    df_age["language"] = get_skills(t, language)
    df_age["language_c"]= get_skills(t, language_c)
    df_age["math"] = get_skills(t, math)
    df_age["math_c"]= get_skills(t, math_c)
    df_age["cognative"] = get_skills(t, cognative)
    df_age["cognative_c"]= get_skills(t, cognative_c)
    df_age["numbers"] = get_skills(t, numbers)
    df_age["numbers_c"]= get_skills(t, numbers_c)
    df_age["stem"] = get_skills(t, stem)
    df_age["stem_c"]= get_skills(t, stem_c)
    df_age["stem_cc"]= get_skills(t, stem_cc)
    df_age["emotional"] = get_skills(t, emotional)
    df_age["emotional_c"]= get_skills(t, emotional_c)
    df_age["social"] = get_skills(t, social)
    df_age["social_c"]= get_skills(t, social_c)


# Putting columns of same skill together

df_age_skills = df_age
df_age_skills["motor_skills"] = df_age_skills["motor"] + df_age_skills["motor_c"]
df_age_skills["language_skills"] = df_age_skills["language"] + df_age_skills["language_c"]
df_age_skills["math_skills"] = df_age_skills["math"] + df_age_skills["math_c"]
df_age_skills["cognative_skills"] = df_age_skills["cognative"] + df_age_skills["cognative_c"]
df_age_skills["numbers_skills"] = df_age_skills["numbers"] + df_age_skills["numbers_c"]
df_age_skills["stem_skills"] = df_age_skills["stem"] + df_age_skills["stem_c"] + df_age_skills["stem_cc"]
df_age_skills["emotional_skills"] = df_age_skills["emotional"] + df_age_skills["emotional_c"]
df_age_skills["social_skills"] = df_age_skills["social"] + df_age_skills["social_c"]
df_merged = df_age_skills.drop(axis=0, columns=["motor", "motor_c", "language", "language_c", "math", "math_c", "cognative", "cognative_c", "numbers", "numbers_c", "stem", "stem_c", "stem_cc", "emotional", "emotional_c", "social", "social_c"])


# Replacing empty strings with NaN

df_merged = df_merged.replace(r'^\s*$', np.NaN, regex=True)


# Unifying values in skills columns

df_merged["math_skills"].replace({"math": "stem", "Math": "stem"}, inplace=True)
df_merged["numbers_skills"].replace({"number": "stem", "Numbers": "stem"}, inplace=True)
df_merged["stem_skills"] = df_merged["math_skills"] + df_merged["numbers_skills"]
df_merged = df_merged.drop(axis=0, columns=["math_skills", "numbers_skills"])
df_merged["stem_skills"].replace({"stemstem": "stem", "stemnumberNumber": "stem", "stemNumber": "stem", "mathMathstem": "stem", "mathMathnumberNumber": "stem" }, inplace=True)
df_merged["motor_skills"].replace({"Motor": "motor", "motorMotor": "motor" }, inplace=True)
df_merged["language_skills"].replace({"Language": "language"}, inplace=True)
df_merged = df_merged.drop(axis=0, columns=["cognative_skills"])
df_merged["emotional_skills"].replace({"Emotional": "emotional", "emotionalEmotional": "emotional"}, inplace=True)
df_merged["social_skills"].replace({"Social": "social"}, inplace=True)

df_final = df_merged 


# One hot encoding

one_hot = OneHotEncoder(sparse=False, handle_unknown='ignore')

m = df_final['motor_skills'].to_numpy().reshape(-1, 1)
one_hot.fit(m)       
motor = one_hot.transform(m)
motor_df = pd.DataFrame(motor, columns= ["motor", "drop"]).drop( axis=0, columns='drop')

l = df_final['language_skills'].to_numpy().reshape(-1, 1)
one_hot.fit(l)       
language = one_hot.transform(l)
language_df = pd.DataFrame(language, columns= ["language", "drop"]).drop( axis=0, columns='drop')

st= df_final['stem_skills'].to_numpy().reshape(-1, 1)
one_hot.fit(st)       
stem = one_hot.transform(st)
stem_df = pd.DataFrame(stem, columns= ["stem", "drop"]).drop( axis=0, columns='drop')

e = df_final['emotional_skills'].to_numpy().reshape(-1, 1)
one_hot.fit(e)       
emotional = one_hot.transform(e)
emotional_df = pd.DataFrame(emotional, columns= ["emotional", "drop"]).drop( axis=0, columns='drop')

s = df_final['social_skills'].to_numpy().reshape(-1, 1)
one_hot.fit(s)       
social = one_hot.transform(s)
social_df = pd.DataFrame(social, columns= ["social", "drop"]).drop( axis=0, columns='drop')

df_final = pd.concat([df_final, motor_df, language_df, stem_df, emotional_df, social_df], axis=1)

df_final = df_final.drop( axis=0, columns=['motor_skills', "language_skills", "emotional_skills", "social_skills", "stem_skills"])


# Cosine Similarity

cs = cosine_similarity

all_scores = []
user_input = np.array([1, 0, 0, 1, 0, 0, 0, 0]).reshape(1, -1)

for i, row in df_final.iterrows():
    row = row[4:].to_numpy().reshape(1, -1)
    all_scores.append((cs(row,user_input)).flatten())

all_scores = pd.Series(np.array(all_scores).flatten(), index=df_final.index)



top_5 = all_scores.sort_values(ascending=False).head().index
