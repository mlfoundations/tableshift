from datetime import datetime

import numpy as np
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

MOOC_FEATURES = FeatureList(features=[
    Feature('course_id', cat_dtype, """Administrative, string, identifies 
    institution (HarvardX or MITx), course name, and semester, 
    e.g. 'HarvardX/CB22x/2013_Spring'."""),
    Feature('certified', int, """Administrative, 0/1; anyone who earned a 
    certificate. Certificates are based on course grades, and depending on 
    the course, the cutoff for a certificate varies from 50% - 80%.""",
            is_target=True),
    Feature('viewed', int, """Administrative, 0/1; anyone who accessed the 
    ‘Courseware’ tab (the home of the videos, problem sets, and exams) within 
    the edX platform for the course. Note that there exist course materials 
    outside of the 'Courseware' tab, such as the Syllabus or the Discussion 
    forums."""),
    Feature('explored', int, """administrative, 0/1; anyone who accessed at 
    least half of the chapters in the courseware (chapters are the highest 
    level on the 'courseware' menu housing course content)."""),
    Feature('final_cc_cname_DI', cat_dtype, """mix of administrative (
    computed from IP address) and user- provided (filled in from student 
    address if available when IP was indeterminate); during 
    de-identification, some country names were replaced with the 
    corresponding continent/region name. Examples: 'Other South Asia' or 
    '“'Russian Federation'”'."""),
    Feature('LoE_DI', cat_dtype, """user-provided, highest level of education 
    completed. Possible values: 'Less than Secondary,' 'Secondary,
    ' 'Bachelor’s,' 'Master’s,' and 'Doctorate.'"""),
    Feature('YoB', int, """user-provided, year of birth. Example: '1980'."""),
    Feature('gender', cat_dtype, """user-provided. Possible values: m (male), 
    f (female) and o (other). Note that 'o' is dropped for this dataset,
    as it only contained 8 observations."""),
    Feature('nevents', int, """administrative, number of interactions with 
    the course, recorded in the tracking logs; blank if no interactions 
    beyond registration. Example: '502'."""),
    Feature('ndays_act', int, """administrative, number of unique days 
    student interacted with course. Example: '16'."""),
    Feature('nplay_video', int, """administrative, number of play video 
    events within the course. Example: '52'."""),
    Feature('nchapters', int, """administrative, number of chapters (within 
    the Courseware) with which the student interacted. Example: '12'."""),
    Feature('nforum_posts', int, """administrative, number of posts to the 
    Discussion Forum. Example: '8'."""),
    Feature('days_from_start_to_last_event', int, """Derived feature. 
    Computes the number of days from a students' first recorded interction 
    with the course platform to their last interaction."""),
],
    documentation="https://dataverse.harvard.edu/file.xhtml?persistentId=doi"
                  ":10.7910/DVN/26147/FD5IES&version=11.2")

_usecols = ['course_id', 'certified', 'viewed',
            'explored',
            'final_cc_cname_DI', 'LoE_DI', 'YoB',
            'gender',
            'start_time_DI', 'last_event_DI', 'nevents',
            'ndays_act',
            'nplay_video',
            'nchapters', 'nforum_posts',
            'incomplete_flag']


def preprocess_mooc(df: pd.DataFrame) -> pd.DataFrame:
    df = df[_usecols]

    # Drop incomplete records due to data processing issue
    # (see Person-Course Documentation PDF at link above)
    df = df[df.incomplete_flag != 1.0]

    for col in ('start_time_DI', 'last_event_DI'):
        df[col] = df[col].fillna('').apply(
            lambda x: datetime.strptime(x, '%m/%d/%y') if x else np.nan)

    df['days_from_start_to_last_event'] = (
            df['last_event_DI'] - df['start_time_DI']) \
        .apply(lambda timedelta: timedelta.days).fillna(0)

    df.drop(columns=['start_time_DI', 'last_event_DI', 'incomplete_flag'],
            inplace=True)

    # Drop students that did not declare gender; note 8 obs with gender
    # "other" are also dropped.
    df = df[df.gender.isin(['m', 'f'])]

    # Fill in zeros for numeric columns; they are coded as empty values
    for colname in (
            'nevents', 'ndays_act', 'nplay_video', 'nchapters',
            'nforum_posts'):
        df[colname] = df[colname].fillna(0)

    df['LoE_DI'] = df['LoE_DI'].fillna('NotProvided')

    df.dropna(inplace=True)  # drops users without YoB; appx 4392 obs.

    return df
