# MEDIQA-MAGIC-2024

Thist dataset includes clinical dermatology textual queries with an associated image, as well as the answers to the queries.

# Data Splits

The following table is the number of instances for each split for our original pull.

|Train|Valid|Test|
| -------- | ------- |------- |
| 435|50|100|

The input dataset is originally from Reddit. You will need to make use of the Reddit API's to download the input data.

The exact available number will depend on the time of your data pull.
At the test submission close time, we will reassess the test set encounters available and re-calculate scores accordingly.

# Data Download + Processing Instructions

1. Create a Reddit Developer Account

Please review Reddit's user terms.
https://www.reddit.com/wiki/api/
https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164

https://github.com/reddit-archive/reddit/wiki/OAuth2

https://www.honchosearch.com/blog/seo/how-to-use-praw-and-crawl-reddit-for-subreddit-post-data

2. Edit the a credentials.yaml file to use your own reddit developer account credentials. It should look something like the following:

credentials.yaml
```
CLIENT_ID: '<YOUR-CLIENT-ID>'
SECRET_TOKEN: '<YOUR-SECRET-TOKEN>'
```

3. Make sure to have praw installed

https://github.com/praw-dev/praw

4. Call the download and preprocess script using the encounter list mediqa-magic_{train,valid,test}_{answersonly,inputonly}.json

```
download_data.py <path-to-credentials> <path-to-postids-json> <output-file-json> <image-dir-output>
```

For example:
```
python download_data.py credentials.yaml train_answersonly.json train_downloaded.json images/train/
python download_data.py credentials.yaml valid_answersonly.json valid_downloaded.json images/valid/
python download_data.py credentials.yaml test_answersonly.json test_downloaded.json images/test/
```

# Data Description

This dataset includes English (en) only.

Input Content

(a) json list where each instance will be represented by a json object with the following attributes:

| attribute_id | description |
| -------- | ------- |
|encounter_id|unique identification string for the case|
|image_ids|list of strings of the image_id’s|
|query_title_{LANGUAGE}|a string representing the query title|
|query_content_{LANGUAGE}|a string representing the query content|

(b) image files with unique id’s

Images are stored in the following folders respectively:
images_{train,valid,test}/

Image naming convention is as follows: IMG_{ENCOUNTER_ID}_{IMAGENUM}.*

(c) Reference data will additionally have the field:

|attribute_id|description|
| -------- | ------- |
|responses|a list of json objects with the following keys ( author_id, content_{LANGUAGE} , completeness, contains_freq_ans)|


## Output Content

Output should be json list with at least the following content

|attribute_id|description|
| -------- | ------- |
|encounter_id|unique identification string for the case|
|responses|a list of json objects with the following keys ( “content_{LANGUAGE}” ) - put your answer in first object|


# Evaluation Script

**We use the following evaluation metrics:**
- **DeltaBLEU** and **bertscore** for general NLG evaluation. DeltaBLEU weighs n-grams according to a human-evaluated score. BERTSCORE takes the maximum score for any available reference. The evaluation scripts used are in evaluation/nlg.

- **MEDCON** for medical concept and assertion evaluation. QUICKUMLS is used to identify concepts (https://github.com/Georgetown-IR-Lab/QuickUMLS), assertion classification is obtained using an in-house llama classifier for English and gpt4 for Chinese. The assertion models are not released, please approximate using your own classifiers. Evaluation scripts are found in (
https://github.com/wyim/MEDIQA-M3G-2024/tree/main) evaluation/medcon.


# Contact

 MEDIQA-NLP mailing list: https://groups.google.com/g/mediqa-nlp 
 Email: mediqa.organizers@gmail.com 

# Organizers  
 
* Asma Ben Abacha, Microsoft, USA
* Wen-wai Yim, Microsoft, USA
* Meliha Yetisgen, University of Washington, USA
* Fei Xia, University of Washington, USA
