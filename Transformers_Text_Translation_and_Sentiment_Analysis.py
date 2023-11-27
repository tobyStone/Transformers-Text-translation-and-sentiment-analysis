import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline


# use the `pd.read_csv()` function to read the movie_review_*.csv files into 3 separate pandas dataframes

# Note: All the dataframes would have different column names. For testing purposes
# you should have the following column names/headers -> [Title, Year, Synopsis, Review]

def preprocess_data() -> pd.DataFrame:
    """
    Reads movie data from .csv files, map column names, add the "Original Language" column,
    and finally concatenate in one resultant dataframe called "df".
    """
    df_eng = pd.read_csv("data/movie_reviews_eng.csv")
    #TODO 1: Add your code here
    return df

df = preprocess_data()

df.sample(10)


# load translation models and tokenizers
# TODO 2: Update the code below
fr_en_model_name = None
es_en_model_name = None
fr_en_model = None
es_en_model = None
fr_en_tokenizer = None
es_en_tokenizer = None

# TODO 3: Complete the function below
def translate(text: str, model, tokenizer) -> str:
    """
    function to translate a text using a model and tokenizer
    """
    # encode the text using the tokenizer
    inputs = None

    # generate the translation using the model
    outputs = None

    # decode the generated output and return the translated text
    decoded = None
    return decoded





# TODO 4: Update the code below

# filter reviews in French and translate to English
fr_reviews = None
fr_reviews_en = None

# filter synopsis in French and translate to English
fr_synopsis = None
fr_synopsis_en = None

# filter reviews in Spanish and translate to English
es_reviews = None
es_reviews_en = None

# filter synopsis in Spanish and translate to English
es_synopsis = None
es_synopsis_en = None

# update dataframe with translated text
# add the translated reviews and synopsis - you can overwrite the existing data


df.sample(10)


# TODO 5: Update the code below
# load sentiment analysis model
model_name = None
sentiment_classifier = None

# TODO 6: Complete the function below
def analyze_sentiment(text, classifier):
    """
    function to perform sentiment analysis on a text using a model
    """
    return None


# TODO 7: Add code below for sentiment analysis
# perform sentiment analysis on reviews and store results in new column


df.sample(10)


# export the results to a .csv file
df.to_csv("result/reviews_with_sentiment.csv", index=False)





