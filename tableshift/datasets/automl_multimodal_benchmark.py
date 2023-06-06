"""
Datasets from the AutoML benchmark at https://github.com/sxjscience/automl_multimodal_benchmark .
"""
from typing import Sequence
from pandas import DataFrame
from tableshift.core.features import Feature, FeatureList, cat_dtype, \
    column_is_of_type

PROD_FEATURES = FeatureList(features=[
    Feature('Text_ID', int),
    Feature('Product_Description', cat_dtype,
            name_extended="product description"),
    Feature('Product_Type', cat_dtype, name_extended='product type'),
    Feature('Sentiment', int, is_target=True,
            value_mapping={
                0: "Cannot Say", 1: "Negative", 2: "Positive",
                3: "No Sentiment"})
],
    documentation='https://machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/overview')

AIRBNB_FEATURES = FeatureList(features=[
    Feature('id', int),
    Feature('listing_url', cat_dtype),
    Feature('scrape_id', float),
    Feature('last_scraped', cat_dtype,
            name_extended='date and time of last scrape'),
    Feature('name', cat_dtype, name_extended='listing name'),
    Feature('summary', cat_dtype),
    Feature('space', cat_dtype),
    Feature('description', cat_dtype,
            name_extended='Detailed description of the listing'),
    Feature('neighborhood_overview', cat_dtype,
            name_extended="Host's description of the neighbourhood"),
    Feature('notes', cat_dtype),
    Feature('transit', cat_dtype),
    Feature('access', cat_dtype),
    Feature('interaction', cat_dtype),
    Feature('house_rules', cat_dtype, name_extended='house rules'),
    Feature('picture_url', cat_dtype,
            'URL to the Airbnb hosted regular sized image for the listing',
            name_extended='listing image URL'),
    Feature('host_id', int, "Airbnb's unique identifier for the host/user",
            name_extended='host ID'),
    Feature('host_url', cat_dtype, "The Airbnb page for the host",
            name_extended='host URL'),
    Feature('host_name', cat_dtype, name_extended='host name'),
    Feature('host_since', cat_dtype,
            "The date the host/user was created. For hosts that are Airbnb guests this could be the date they registered as a guest.",
            name_extended='date of host registration'),
    Feature('host_location', cat_dtype, "The host's self reported location",
            name_extended='host location'),
    Feature('host_about', cat_dtype, "Description about the host",
            name_extended="description of host"),
    Feature('host_response_time', cat_dtype,
            name_extended='host response time'),
    Feature('host_response_rate', cat_dtype,
            name_extended='host response rate'),
    Feature('host_is_superhost', cat_dtype, name_extended='host is superhost'),
    Feature('host_thumbnail_url', cat_dtype,
            name_extended='host thumbnail URL'),
    Feature('host_picture_url', cat_dtype, name_extended='host picture URL'),
    Feature('host_neighborhood', cat_dtype, name_extended='host neighborhood'),
    Feature('host_verifications', cat_dtype,
            name_extended='host verifications'),
    Feature('host_has_profile_pic', cat_dtype,
            name_extended='host has profile pic'),
    Feature('host_identity_verified', cat_dtype,
            name_extended='host identitity verified'),
    Feature('street', cat_dtype),
    Feature('neighborhood', cat_dtype),
    Feature('city', cat_dtype),
    Feature('suburb', cat_dtype),
    Feature('state', cat_dtype),
    Feature('zipcode', cat_dtype),
    Feature('smart_location', cat_dtype, name_extended='smart location'),
    Feature('country_code', cat_dtype, name_extended='country code'),
    Feature('country', cat_dtype),
    Feature('latitude', float),
    Feature('longitude', float),
    Feature('is_location_exact', cat_dtype, name_extended='location is exact'),
    Feature('property_type', cat_dtype, name_extended='property type'),
    Feature('room_type', cat_dtype, name_extended='room type'),
    Feature('accommodates', int),
    Feature('bathrooms', float),
    Feature('bedrooms', float),
    Feature('beds', float),
    Feature('bed_type', cat_dtype, name_extended='bed type'),
    Feature('amenities', cat_dtype),
    Feature('price', int),
    Feature('weekly_price', float, name_extended='weekly price'),
    Feature('monthly_price', float, name_extended='monthly price'),
    Feature('security_deposit', float, name_extended='security deposit'),
    Feature('cleaning_fee', float, name_extended='cleaning fee'),
    Feature('guests_included', int, name_extended='guests included'),
    Feature('extra_people', int, name_extended='extra people'),
    Feature('minimum_nights', int, name_extended='minimum nights'),
    Feature('maximum_nights', int, name_extended='maximum nights'),
    Feature('calendar_updated', cat_dtype, name_extended='calendar updated'),
    Feature('has_availability', cat_dtype, name_extended='has availability'),
    Feature('availability_30', int,
            "avaliability_x. The availability of the listing x days in the "
            "future as determined by the calendar. Note a listing may not be "
            "available because it has been booked by a guest or blocked by "
            "the host.",
            name_extended='availability in next 30 days'),
    Feature('availability_60', int,
            "avaliability_x. The availability of the listing x days in the "
            "future as determined by the calendar. Note a listing may not be "
            "available because it has been booked by a guest or blocked by "
            "the host.",
            name_extended='availability in next 60 days'),
    Feature('availability_90', int,
            "avaliability_x. The availability of the listing x days in the "
            "future as determined by the calendar. Note a listing may not be "
            "available because it has been booked by a guest or blocked by "
            "the host.",
            name_extended='availability in next 90 days'),
    Feature('availability_365', int,
            "avaliability_x. The availability of the listing x days in the "
            "future as determined by the calendar. Note a listing may not be "
            "available because it has been booked by a guest or blocked by "
            "the host.",
            name_extended='availability in next 365 days'),
    Feature('calendar_last_scraped', cat_dtype,
            name_extended='last calendar scrape date'),
    Feature('number_of_reviews', int, name_extended='number of reviews'),
    Feature('first_review', cat_dtype,
            name_extended='date of the first/oldest review'),
    Feature('last_review', cat_dtype,
            name_extended='date of the last/newest review'),
    Feature('review_scores_rating', float,
            name_extended='average reviewer overall rating'),
    Feature('review_scores_accuracy', float,
            name_extended='average reviewer rating for accuracy'),
    Feature('review_scores_cleanliness', float,
            name_extended='average reviewer rating for cleanliness'),
    Feature('review_scores_checkin', float,
            name_extended='average reviewer rating for check-in'),
    Feature('review_scores_communication', float,
            name_extended='average reviewer rating for communication'),
    Feature('review_scores_location', float,
            name_extended='average reviewer rating for location'),
    Feature('review_scores_value', float,
            name_extended='average reviewer rating for value'),
    Feature('requires_license', cat_dtype, name_extended='requires license'),
    Feature('license', cat_dtype),
    Feature('instant_bookable', cat_dtype, name_extended='instant bookable'),
    Feature('cancellation_policy', cat_dtype,
            name_extended='cancellation policy'),
    Feature('require_guest_profile_picture', cat_dtype,
            name_extended='requires guest profile picture'),
    Feature('require_guest_phone_verification', cat_dtype,
            name_extended='requires guest phone verification'),
    Feature('calculated_host_listings_count', int,
            "The number of listings the host has in the current scrape, in the city/region geography.",
            name_extended='number of current listings for this host in the city/region'),
    Feature('reviews_per_month', float,
            "The number of reviews the listing has over the lifetime of the listing",
            name_extended='reviews per month'),
    Feature('price_label', int, is_target=True,
            name_extended='price per night',
            value_mapping={
                0: '[0,46)',
                1: "[46,60)",
                2: "[60,75)",
                3: "[75,90)",
                4: "[90,100)",
                5: "[100,120)",
                6: "[120,140)",
                7: "[140,159)",
                8: "[159,192)",
                9: "[192,inf)"}),
    Feature('host_verifications_jumio', bool),
    Feature('host_verifications_government_id', bool),
    Feature('host_verifications_kba', bool),
    Feature('host_verifications_zhima_selfie', bool),
    Feature('host_verifications_facebook', bool),
    Feature('host_verifications_work_email', bool),
    Feature('host_verifications_google', bool),
    Feature('host_verifications_sesame', bool),
    Feature('host_verifications_manual_online', bool),
    Feature('host_verifications_manual_offline', bool),
    Feature('host_verifications_offline_government_id', bool),
    Feature('host_verifications_selfie', bool),
    Feature('host_verifications_reviews', bool),
    Feature('host_verifications_identity_manual', bool),
    Feature('host_verifications_sesame_offline', bool),
    Feature('host_verifications_weibo', bool),
    Feature('host_verifications_email', bool),
    Feature('host_verifications_sent_id', bool),
    Feature('host_verifications_phone', bool),

], documentation="https://www.kaggle.com/tylerx/melbourne-airbnb-open-data ,"
                 "https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=1322284596")

WINE_REVIEWS_FEATURES = FeatureList([
    Feature('country', cat_dtype),
    Feature('description', cat_dtype),
    Feature('points', int, name_extended='wine enthusiast rating'),
    Feature('price', float),
    Feature('province', cat_dtype),
    Feature('variety', cat_dtype, is_target=True),
],
    documentation="https://www.kaggle.com/zynicide/wine-reviews")

IMDB_FEATURES = FeatureList([
    Feature('Rank', int),
    Feature('Title', cat_dtype, name_extended='title of the job ad'),
    Feature('Description', cat_dtype),
    Feature('Director', cat_dtype),
    Feature('Actors', cat_dtype, name_extended='List of actors'),
    Feature('Year', int),
    Feature('Runtime (Minutes)', int, name_extended='runtime in minutes'),
    Feature('Rating', float),
    Feature('Votes', int),
    Feature('Revenue (Millions)', float),
    Feature('Metascore', float),
    Feature('Genre_is_Drama', int, name_extended='genre is drama',
            is_target=True),
], documentation="https://www.kaggle.com/PromptCloudHQ/imdb-data")

JIGSAW_FEATURES = FeatureList([
    Feature('id', int),
    Feature('target', int, name_extended='is toxic', is_target=True),
    Feature('comment_text', cat_dtype, name_extended='comment text'),
    Feature('severe_toxicity', float, name_extended='severe toxicity score'),
    Feature('obscene', float, name_extended='obscene score'),
    Feature('identity_attack', float, name_extended='identity attack score'),
    Feature('insult', float, name_extended='insult score'),
    Feature('threat', float, name_extended='threat score'),
    Feature('asian', float, name_extended='asian score'),
    Feature('atheist', float, name_extended='atheist score'),
    Feature('bisexual', float, name_extended='bisexual score'),
    Feature('black', float, name_extended='black score'),
    Feature('buddhist', float, name_extended='buddhist score'),
    Feature('christian', float, name_extended='christian score'),
    Feature('female', float, name_extended='female score'),
    Feature('heterosexual', float, name_extended='heterosexual score'),
    Feature('hindu', float, name_extended='hindu score'),
    Feature('homosexual_gay_or_lesbian', float,
            name_extended='homosexual gay or lesbian score'),
    Feature('intellectual_or_learning_disability', float,
            name_extended='intellectual or learning disability score'),
    Feature('jewish', float, name_extended='jewish score'),
    Feature('latino', float, name_extended='latino score'),
    Feature('male', float, name_extended='male score'),
    Feature('muslim', float, name_extended='muslim score'),
    Feature('other_disability', float, name_extended='other disability score'),
    Feature('other_gender', float, name_extended='other gender score'),
    Feature('other_race_or_ethnicity', float,
            name_extended='other race or ethnicity score'),
    Feature('other_religion', float, name_extended='other religion score'),
    Feature('other_sexual_orientation', float,
            name_extended='other sexual orientation score'),
    Feature('physical_disability', float,
            name_extended='physical disability score'),
    Feature('psychiatric_or_mental_illness', float,
            name_extended='psychiatric or mental illness score'),
    Feature('transgender', float, name_extended='transgender score'),
    Feature('white', float, name_extended='white score'),
    Feature('created_date', cat_dtype),
    Feature('publication_id', int),
    Feature('parent_id', float),
    Feature('article_id', int),
    Feature('rating', cat_dtype),
    Feature('funny', int, name_extended='user votes for funny'),
    Feature('wow', int, name_extended='user votes for wow'),
    Feature('sad', int, name_extended='user votes for sad'),
    Feature('likes', int, name_extended='user likes'),
    Feature('disagree', int, name_extended='user votes for disagree'),
    Feature('sexual_explicit', float, name_extended='sexual explicit score'),
    Feature('identity_annotator_count', int,
            name_extended='count of identity annotators'),
    Feature('toxicity_annotator_count', int,
            name_extended='count of toxicity annotators'),
],
    documentation='https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification')

FAKE_JOBS_FEATURES = FeatureList([
    Feature('title', cat_dtype, name_extended='title of the job ad entry'),
    # Feature('location', cat_dtype, name_extended='geographical location of the job ad'),
    Feature('salary_range', cat_dtype, name_extended='salary range'),
    Feature('description', cat_dtype),
    Feature('required_experience', cat_dtype,
            name_extended='required experience'),
    Feature('required_education', cat_dtype,
            name_extended='required education'),
    Feature('fraudulent', int, is_target=True)

],
    documentation='https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction')

KICKSTARTER_FEATURES = FeatureList([
    Feature('name', cat_dtype, name_extended='name of the project'),
    Feature('desc', cat_dtype, name_extended='description of project'),
    Feature('goal', float, 'the goal (amount) required for the project'),
    Feature('keywords', cat_dtype),
    Feature('disable_communication', bool,
            name_extended='project authors have disabled communication'),
    Feature('country', cat_dtype),
    Feature('currency', cat_dtype),
    Feature('deadline', int,
            name_extended='unix timestamp of project deadline'),
    Feature('created_at', int,
            name_extended='unix timestamp of when the project was posted'),
    Feature('final_status', int, is_target=True,
            name_extended='funded successfully'),
],
    documentation='https://www.kaggle.com/codename007/funding-successful-projects')

NEWS_CHANNEL_FEATURES = FeatureList([
    Feature('n_tokens_content', float, name_extended='Number of words in the content'),
    Feature('n_unique_tokens', float, name_extended='Rate of unique words in the content'),
    Feature('n_non_stop_words', float, name_extended='Rate of non-stop words in the content'),
    Feature('n_non_stop_unique_tokens', float, name_extended='Rate of unique non-stop words in the content'),
    Feature('num_hrefs', float, name_extended='Number of links'),
    Feature('num_self_hrefs', float, name_extended='Number of links to other articles published by Mashable'),
    Feature('num_imgs', float, name_extended='Number of images'),
    Feature('num_videos', float, name_extended='Number of videos'),
    Feature('average_token_length', float, name_extended='Average length of the words in the content '),
    Feature('num_keywords', float, name_extended='Number of keywords in the metadata'),
    Feature('global_subjectivity', float, name_extended='Text subjectivity'),
    Feature('global_sentiment_polarity', float, name_extended='Text sentiment polarity'),
    Feature('global_rate_positive_words', float, name_extended='Rate of positive words in the content'),
    Feature('global_rate_negative_words', float, name_extended='Rate of negative words in the content'),
    Feature('rate_positive_words', float, name_extended='Rate of positive words among non-neutral tokens'),
    Feature('rate_negative_words', float, name_extended='Rate of negative words among non-neutral tokens'),
    Feature('article_title', cat_dtype, name_extended='article title'),
    Feature('channel', cat_dtype, is_target=True,
            value_mapping={
                ' data_channel_is_tech': 'tech',
                ' data_channel_is_entertainment': 'entertainment',
                ' data_channel_is_lifestyle': 'lifestyle',
                ' data_channel_is_bus': 'business',
                ' data_channel_is_world': 'world',
                ' data_channel_is_socmed': 'social media',
            }),
],
    documentation='https://archive.ics.uci.edu/ml/datasets/online+news+popularity')

SALARY_FEATURES = FeatureList([
    Feature('experience', cat_dtype),
    Feature('job_description', cat_dtype, name_extended='job description'),
    Feature('job_desig', cat_dtype, name_extended='job designation'),
    Feature('job_type', cat_dtype, name_extended="job type"),
    Feature('key_skills', cat_dtype, name_extended='key skills'),
    Feature('location', cat_dtype),
    Feature('salary', cat_dtype, is_target=True,
            name_extended='Salary in Rupees Lakhs',
            value_mapping={
                '25to50': '[25,50)', '3to6': '[3,6)', '15to25': '[15,25)',
                '10to15': '[10,15)', '6to10': '[6,10)', '0to3': '[0,3)', }),
],
    documentation='https://machinehack.com/hackathons/predict_the_data_scientists_salary_in_india_hackathon/overview')



def fill_missing(df: DataFrame, cols: Sequence[str]) -> DataFrame:
    for col in cols:
        df[col] = df[col].fillna('MISSING')
    return df


def preprocess_automl(df: DataFrame,
                      automl_benchmark_dataset_name: str) -> DataFrame:
    # For any string columns, we insert a 'missing' dummy value.
    object_cols = [c for c in df.columns if column_is_of_type(df[c], 'object')]
    df = fill_missing(df, object_cols)

    # remove leading spaces from column names
    for c in df.columns:
        if c.startswith(' '):
            df.rename(columns={c: c.lstrip()}, inplace=True)

    if automl_benchmark_dataset_name == "imdb_genre_prediction":
        df[IMDB_FEATURES.target] = df[IMDB_FEATURES.target].astype(int)
    elif automl_benchmark_dataset_name == "jigsaw_unintended_bias100K":
        df[JIGSAW_FEATURES.target] = df[JIGSAW_FEATURES.target].astype(int)

    return df
