import pandas as pd

df_allow_columns = [
    "encounter_id", "author_id", "image_ids", "responses", "query_title_en", "query_content_en"
]

def generate_image_paths(image_id):
    return f'2025_dataset/train/images_train/{image_id}'

def generate_training_dataframe(file_path):
    df = pd.read_json(file_path)
    df = df[df_allow_columns]
    df['responses'] = df['responses'].apply(lambda x: [values['content_en'] for values in x])
    df['image_ids'] = df['image_ids'].apply(lambda x: [generate_image_paths(values['image_ids'] for values in x['image_ids'])])
    

generate_training_dataframe('2025_dataset/train/train.json')