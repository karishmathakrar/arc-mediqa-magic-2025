import os
import json
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "2025_dataset", "train")

    train_json_path = os.path.join(data_dir, "train.json")
    train_df = pd.read_json(train_json_path)

    train_df = train_df[[
        "encounter_id", "author_id", "image_ids", "responses", 
        "query_title_en", "query_content_en"
    ]]

    def generate_image_paths(image_ids):
        return [os.path.normpath(os.path.join(data_dir, "images_train", img)) for img in image_ids]

    train_df["image_paths"] = train_df["image_ids"].apply(generate_image_paths)

    train_df["responses_en"] = train_df["responses"].apply(
        lambda resp_list: [r["content_en"] for r in resp_list]
    )

    cvqa_path = os.path.join(data_dir, "train_cvqa.json")
    with open(cvqa_path, "r", encoding="utf-8") as f:
        cvqa_data = json.load(f)
    cvqa_df = pd.json_normalize(cvqa_data)

    cvqa_long = cvqa_df.melt(id_vars=["encounter_id"], 
                             var_name="qid", 
                             value_name="answer_index")

    questions_path = os.path.join(data_dir, "closedquestions_definitions_imageclef2025.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions_df = pd.json_normalize(questions)[["qid", "question_en", "options_en"]]

    cvqa_merged = cvqa_long.merge(questions_df, on="qid", how="left")

    def get_answer_text(row):
        try:
            return row["options_en"][row["answer_index"]]
        except (IndexError, TypeError):
            return None

    cvqa_merged["answer_text"] = cvqa_merged.apply(get_answer_text, axis=1)

    final_df = cvqa_merged.merge(train_df, on="encounter_id", how="left")

    output_path = os.path.join(data_dir, "final_df.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Saved merged dataframe to {output_path}")

if __name__ == "__main__":
    main()