import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import gender_mapping, race_mapping, age_mapping, education_mapping, locale_mapping

def load_data():
    # Load dataset
    df = pd.read_csv("data/dices_990_full.csv")

    removed_raters_990 = [
        '296768824962581', '296740172100277', '296729424498056',
        '296835612486158', '297212198239077', '296740254381555',
        '296740049152799', '297212002905763', '296740761150184',
        '296708974347177', '296770194464643', '296768399729683'
    ]
    df = df[~df['rater_id'].isin(removed_raters_990)].reset_index(drop=True)
    df = df[df['Q_overall'].isin(['Yes','No'])].reset_index(drop=True)
    df['Q_overall'] = df['Q_overall'].replace({'Yes':1,'No':0})
    df['item_id'] += 351

    # Train/test split (Fold 1)
    test_df = df[(df['item_id'] >= 351) & (df['item_id'] <= 548)].reset_index(drop=True)
    train_df = df[df['item_id'] > 548].reset_index(drop=True)

    # Embeddings
    deepseek = pd.read_csv("data/D990_deepseek.csv")
    df_conv = pd.read_csv("data/convo_990.csv")
    split_safe = [[p.strip() for p in item.split(',')] for item in deepseek['safe_chars']]
    split_unsafe = [[p.strip() for p in item.split(',')] for item in deepseek['unsafe_chars']]
    df_conv['safe_chars'] = split_safe
    df_conv['unsafe_chars'] = split_unsafe

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    df_conv['safe_embeddings'] = df_conv['safe_chars'].apply(lambda x: model.encode(x).tolist())
    df_conv['unsafe_embeddings'] = df_conv['unsafe_chars'].apply(lambda x: model.encode(x).tolist())

    df_conv['final_emb'] = df_conv.apply(
        lambda row: [a+b for a,b in zip(row['safe_embeddings'][0], row['safe_embeddings'][1])]
                   if len(row['safe_embeddings']) > 1 else row['safe_embeddings'][0],
        axis=1
    )

    df_final_emb = df_conv[['item_id','final_emb']]
    train_df = train_df.merge(df_final_emb, on='item_id', how='left')
    test_df = test_df.merge(df_final_emb, on='item_id', how='left')

    # Sociocultural encodings
    def encode(df):
        out = pd.DataFrame()
        out['text_embedding'] = df['final_emb']
        out['rater_gender_encoded'] = df['rater_gender'].map(gender_mapping)
        out['rater_race_encoded'] = df['rater_race'].map(race_mapping)
        out['rater_age_encoded'] = df['rater_age'].map(age_mapping)
        out['rater_education_encoded'] = df['rater_education'].map(education_mapping)
        out['rater_locale_encoded'] = df['rater_locale'].map(locale_mapping)
        out['item_id'] = df['item_id']
        return out

    data_train = encode(train_df)
    data_test = encode(test_df)

    # Add Q_average
    grouped = train_df.groupby('item_id')['Q_overall'].mean().reset_index(name='Q_average')
    train_df = train_df.merge(grouped, on='item_id', how='left')
    data_train['Q_average'] = train_df['Q_average']
    data_train['Q_overall'] = train_df['Q_overall']

    grouped = test_df.groupby('item_id')['Q_overall'].mean().reset_index(name='Q_average')
    test_df = test_df.merge(grouped, on='item_id', how='left')
    data_test['Q_average'] = test_df['Q_average']
    data_test['Q_overall'] = test_df['Q_overall']

    return data_train, data_test
