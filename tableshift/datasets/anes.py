"""
Utilities for ANES Time Series Cumulative Data File.

List of variables: https://electionstudies.org/wp-content/uploads/2019/09/anes_timeseries_cdf_codebook_Varlist.pdf
Codebook: https://electionstudies.org/wp-content/uploads/2022/09/anes_timeseries_cdf_codebook_var_20220916.pdf
"""
import pandas as pd

from tableshift.core.features import Feature, cat_dtype, FeatureList

# Note that "state" feature is named as VCF0901b; see below. Note that '99' is
# also a valid value, but it contains all missing targets .
ANES_STATES = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
               'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
               'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
               'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
               'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# Note that "year" feature is named as VCF0004; see below.
ANES_YEARS = [1948, 1952, 1954, 1956, 1958, 1960, 1962, 1964, 1966, 1968, 1970,
              1972, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992,
              1994, 1996, 1998, 2000, 2002, 2004, 2008, 2012, 2016, 2020]

# U.S. Census Regions; see 'VCF0112' feature below.
ANES_REGIONS = ['MISSING', '1.0', '2.0', '4.0', '3.0']

# This is a very preliminary feature list. We should probably
# try to find a good reference/principled heuristics for selecting these.
# I generally tried to select a few from each category, with an emphasis on
# questions that would be asked/relevant every year (i.e. not
# questions about Kennedy, Vietnam, Cold War, etc.).
# Only pre-election questions. Also dropped questions that were
# asked in only 3 or fewer years.

# We give the actual coding values for potential sensitive variables; for others
# we mostly give the question title; see the documentation linked above for
# details.

ANES_FEATURES = FeatureList(features=[
    Feature('VCF0702', int, "DID RESPONDENT VOTE IN THE NATIONAL "
                            "ELECTIONS 1. No, did not vote 2. Yes, "
                            "voted 0. DK; NA; no Post IW; refused to "
                            "say if voted; Washington D.C. ("
                            "presidential years only)",
            is_target=True,
            name_extended='voted in national election'),
    Feature("VCF0004", int, name_extended="election year"),
    Feature("VCF0901b", cat_dtype, """State of interview - state postal 
    abbreviation, 99. NA; wrong district identified (2000) INAP. question not 
    used""",
            name_extended='state'),

    # PARTISANSHIP AND ATTITUDES TOWARDS PARTIES
    Feature('VCF0218', float,
            name_extended="Democratic Party feeling thermometer"),
    Feature('VCF0224', float,
            name_extended="Republican Party feeling thermometer"),
    Feature('VCF0301', cat_dtype,
            """Generally speaking, do you usually think of yourself as a 
            Republican, a Democrat, an Independent, or what? (IF REPUBLICAN 
            OR DEMOCRAT) you call yourself a strong (REP/DEM) or a not very 
            strong (REP/DEM)? (IF INDEPENDENT, OTHER [1966 AND LATER:] OR NO 
            PREFERENCE; 2008: OR DK) Do you think of yourself as closer to 
            the Republican or Democratic party?""",
            name_extended="think of yourself as closer to the Republican or Democratic party",
            value_mapping={
                '0.0': "no answer",
                '1.0': "Strong Democrat",
                '2.0': "Weak Democrat",
                '3.0': "Independent - Democrat",
                '4.0': "Independent - Independent",
                '5.0': "Independent - Republican",
                '6.0': "Weak Republican",
                '7.0': "Strong Republican",
            }),
    Feature('VCF0302', cat_dtype,
            "Generally speaking, do you usually think of yourself as a "
            "Republican, a Democrat, an Independent, or what?",
            name_extended='party identification',
            value_mapping={
                '1.0': "Republican",
                '2.0': "Independent",
                '3.0': "No preference; none; neither",
                '4.0': "Other",
                '5.0': "Democrat",
                '8.0': "Don't know",
                '9.0': "no answer", }),
    Feature('VCF9008', cat_dtype,
            """Which party do you think would do a better job of handling the 
            problem of pollution and (1990,1994: protection of) the 
            environment?""",
            name_extended='party preference on pollution and environment',
            value_mapping={
                '1.0': "Better by Democrats",
                '3.0': "Same by both",
                '5.0': "Better by Republicans",
                '8.0': "Don't know",
                '9.0': "no answer"}),
    Feature('VCF9010', cat_dtype,
            """Do you think inflation would be handled better by the 
            Democrats, by the Republicans, or about the same by both?""",
            name_extended='party preference on inflation',
            value_mapping={
                '1.0': "Better by Democrats",
                '3.0': "Same by both",
                '5.0': "Better by Republicans",
                '8.0': "Don't know",
                '9.0': "no answer"}),
    Feature('VCF9011', cat_dtype,
            """Do you think the problems of unemployment would be handled 
            better by the Democrats, by the Republicans, or about the same by 
            both?""",
            name_extended='party preference on unemployment',
            value_mapping={
                '1.0': "Better by Democrats",
                '3.0': "Same by both",
                '5.0': "Better by Republicans",
                '8.0': "Don't know",
                '9.0': "no answer"}),
    Feature('VCF9201', cat_dtype,
            """(I’d like to know what you think about each of our political 
            parties. After I read the name of a political party, please rate 
            it on a scale from 0 to 10, where 0 means you strongly dislike 
            that party and 10 means that you strongly like that party. If I 
            come to a party you haven’t heard of or you feel you do not know 
            enough about, just say so.) [The first party is: / Using the same 
            scale where would you place:] the Democratic party {INTERVIEWER: 
            DO NOT PROBE DON’T KNOW}""",
            name_extended='like-dislike scale placement for democratic party (0-10)',
            na_values=(-7., -8., -9.)),
    Feature('VCF9202', cat_dtype,
            """(I’d like to know what you think about each of our political 
            parties. After I read the name of a political party, please rate 
            it on a scale from 0 to 10, where 0 means you strongly dislike 
            that party and 10 means that you strongly like that party. If I 
            come to a party you haven’t heard of or you feel you do not know 
            enough about, just say so.) [The first party is: / Using the same 
            scale where would you place:] the Republican party {INTERVIEWER: 
            DO NOT PROBE DON’T KNOW}""",
            name_extended='like-dislike scale placement for republican party (0-10)',
            na_values=(-7., -8., -9.)
            ),
    Feature('VCF9203', cat_dtype,
            """Would you say that any of the parties in the United States 
            represents your views reasonably well? {INTERVIEWER: DO NOT PROBE 
            DON’T KNOW}""",
            name_extended='do any of the parties in the U.S. represent views reasonably well',
            value_mapping={
                '1.0': "Yes",
                '2.0': "No",
                '-8.0': "no answer",
                '-9.0': "no answer"}),
    Feature('VCF9204', cat_dtype,
            """(Would you say that any of the parties in the United States 
            represents your views reasonably well?) Which party represents 
            your views best? {INTERVIEWER: DO NOT PROBE DON’T KNOW}""",
            name_extended='party in US that represents views best',
            value_mapping={
                '1.0': "Democratic",
                '2.0': "Republican",
                '7.0': "Other",
                '-8.0': "no answer",
                '-9.0': "no answer", }),
    Feature('VCF9205', cat_dtype,
            """Which party do you think would do a better job of handling the 
            nation’s economy, the Democrats, the Republicans, or wouldn’t 
            there be much difference between them? {1996: IF ‘NO DIFFERENCE’ 
            AND ‘NEITHER PARTY’ ARE VOLUNTEERED, DO NOT PROBE RESPONSES. 
            2000-later: IF ‘DK’ OR ‘NEITHER PARTY’ IS VOLUNTEERED, DO NOT 
            PROBE]}""",
            name_extended='which political party represents views best',
            value_mapping={
                '1.0': 'Democrats',
                '2.0': 'No difference',
                '3.0': 'Republican',
                '7.0': 'Neither party',
                '-8.0': "no answer",
                '-9.0': "no answer"}),
    Feature('VCF9206', cat_dtype,
            """Do you think it is better when one party controls both the 
            presidency and Congress; better when control is split between the 
            Democrats and Republicans, or doesn’t it matter?""",
            name_extended='better when one party controls both presidency and congress or when control is split',
            value_mapping={
                '1.0': "One party control both ",
                '2.0': "Control is split",
                '3.0': "It doesn’t matter",
                '-8.0': "no answer",
                '-9.0': "no answer"}),

    # PERCEIVED POSITIONS OF PARTIES
    Feature('VCF0521', cat_dtype,
            """Which party do you think is more likely to favor a stronger [
            1978,1980, 1984: more powerful; 1988,1992: a powerful] government 
            in Washington – the Democrats, the Republicans, or wouldn’t there 
            be any difference between them on this?""",
            name_extended='which party favors stronger government',
            value_mapping={
                '0.0': "no answer",
                '1.0': "Democrats",
                '2.0': "No difference",
                '3.0': "Republicans",
                '8.0': "DK which party",
            }),
    Feature('VCF0523', cat_dtype,
            """Which political party do you think is more in favor of cutting 
            military spending - the Democrats, the Republicans, or wouldn’t 
            there be much difference between them?""",
            name_extended='which party favors military spending cut',
            value_mapping={
                '1.0': "Democrats",
                '2.0': "Not much difference",
                '3.0': "Republicans",
                '8.0': "no answer",
                '0.0': "no answer"}),

    # CANDIDATE AND INCUMBENT EVALUATIONS
    Feature('VCF0428', float, name_extended="President thermometer",
            na_values=(98., 99.)),
    Feature('VCF0429', float, name_extended="Vice-president thermometer",
            na_values=(98., 99.)),

    # CANDIDATE/INCUMBENT PERFORMANCE EVALUATIONS
    Feature('VCF0875', cat_dtype, "MENTION 1: WHAT IS THE MOST IMPORTANT "
                                  "NATIONAL PROBLEM",
            name_extended='most important national problem',
            value_mapping={
                '1.0': "AGRICULTURAL",
                '2.0': "ECONOMICS; BUSINESS; CONSUMER ISSUES",
                '3.0': "FOREIGN AFFAIRS AND NATIONAL DEFENSE",
                '4.0': "GOVERNMENT FUNCTIONING",
                '5.0': "LABOR ISSUES",
                '6.0': "NATURAL RESOURCES",
                '7.0': "PUBLIC ORDER",
                '8.0': "RACIAL PROBLEMS",
                '9.0': "SOCIAL WELFARE",
                '97.0': "Other problems",
                '98.0': "no answer",
                '99.0': "no answer",
                '0.0': "no answer"}),
    Feature('VCF9052', cat_dtype,
            """Let’s talk about the country as a whole. Would you say that 
            things in the country are generally going very well, fairly well, 
            not too well or not well at all?""",
            name_extended='are things in U.S. going well or not',
            value_mapping={
                '1.0': "Very well",
                '2.0': "Fairly well",
                '4.0': "Not too well",
                '5.0': "Not well at all",
                '8.0': "no answer",
                '9.0': "no answer"}),
    # ISSUES
    Feature('VCF0809', cat_dtype, "GUARANTEED JOBS AND INCOME SCALE.",
            name_extended="guaranteed jobs and income scale (support/don't support)"),
    Feature('VCF0839', cat_dtype, "GOVERNMENT SERVICES-SPENDING SCALE",
            name_extended='government services and spending scale (fewer/more services)'),
    Feature('VCF0822', cat_dtype,
            """As to the economic policy of the government – I mean steps 
            taken to fight inflation or unemployment – would you say the 
            government is doing a good job, only fair, or a poor job?""",
            name_extended='rating of government economic policy',
            value_mapping={
                '1.0': "Poor job",
                '2.0': "Only fair",
                '3.0': "Good job",
                '0.0': "no answer",
                '9.0': "no answer"}),
    Feature('VCF0870', cat_dtype, "BETTER OR WORSE ECONOMY IN PAST YEAR",
            name_extended='better or worse economy in past year',
            value_mapping={
                '1.0': "Better",
                '3.0': "Stayed same",
                '5.0': "Worse",
                '0.0': "no answer",
                '8.0': "no answer"}),
    Feature('VCF0843', cat_dtype, "DEFENSE SPENDING SCALE",
            name_extended='defense spending scale (decrease/increase)',
            na_values=(0.,)
            ),
    Feature('VCF9045', cat_dtype,
            "POSITION OF THE U.S. WEAKER/STRONGER IN THE PAST YEAR",
            name_extended='position of the U.S. in past year',
            value_mapping={
                '1.0': "Weaker",
                '3.0': "Same",
                '5.0': "Stronger",
                '8.0': "Don't know",
                '9.0': "no answer"}),
    Feature('VCF0838', cat_dtype, "BY LAW, WHEN SHOULD ABORTION BE ALLOWED",
            name_extended='when should abortion be allowed by law',
            value_mapping={
                '1.0': "By law, abortion should never be permitted.",
                '2.0': "The law should permit abortion only in case of rape, incest, or when the woman’s life is in danger.",
                '3.0': "The law should permit abortion for reasons other than rape, incest, or danger to the woman’s life, but only after the need for the abortion has been clearly established.",
                '4.0': "By law, a woman should always be able to obtain an abortion as a matter of personal choice.",
                '9.0': "Don't know or other",
                '0.0': "no answer"}),
    Feature('VCF9239', cat_dtype, "HOW IMPORTANT IS GUN CONTROL ISSUE TO R",
            name_extended='importance of gun control',
            value_mapping={
                '1.0': "Extremely important ",
                '2.0': "Very important",
                '3.0': "Somewhat important ",
                '4.0': "Not too important",
                '5.0': "Not at all important",
                '-8.0': "no answer",
                '-9.0': "no answer"}),
    # IDEOLOGY AND VALUES
    Feature('VCF0803', cat_dtype, "LIBERAL-CONSERVATIVE SCALE",
            name_extended='liberal-conservative scale',
            value_mapping={
                '1.0': "Extremely liberal",
                '2.0': "Liberal",
                '3.0': "Slightly liberal",
                '4.0': "Moderate, middle of the road ",
                '5.0': "Slightly conservative",
                '6.0': "Conservative",
                '7.0': "Extremely conservative",
                '9.0': "Don't know; haven’t thought much about it",
                '0.0': "no answer"}),
    Feature('VCF0846', cat_dtype, "IS RELIGION IMPORTANT TO RESPONDENT",
            name_extended='importance of religion',
            value_mapping={
                '1.0': "Yes, important",
                '2.0': "Little to no importance",
                '0.0': "no answer",
                '8.0': "no answer"}),
    # SYSTEM SUPPORT
    Feature('VCF0601', cat_dtype, "APPROVE PARTICIPATION IN PROTESTS",
            name_extended='approve participation in protests',
            value_mapping={
                '1.0': "Disapprove",
                '2.0': "Pro-con, depends, don’t know",
                '3.0': "Approve",
                '0.0': "no answer"}),
    Feature('VCF0606', cat_dtype, "HOW MUCH DOES THE FEDERAL GOVERNMENT WASTE "
                                  "TAX MONEY",
            name_extended='how much does federal government waste tax money',
            value_mapping={
                '1.0': "A lot",
                '2.0': "Some",
                '3.0': "Not very much",
                '9.0': "Don't know",
                '0.0': "no answer"}),
    Feature('VCF0612', cat_dtype, "VOTING IS THE ONLY WAY TO HAVE A SAY IN "
                                  "GOVERNMENT",
            name_extended='voting is the only way to have a say in government',
            value_mapping=
            {
                '1.0': "Agree",
                '2.0': "Disagree",
                '9.0': "Don't know or not sure",
                '0.0': "no answer"}),
    Feature('VCF0615', cat_dtype, "MATTER WHETHER RESPONDENT VOTES OR NOT",
            name_extended='it matters whether I vote',
            value_mapping={
                '1.0': "Agree",
                '2.0': "Disagree",
                '3.0': "Neither agree nor disagree",
                '9.0': "Don't know or not sure",
                '0.0': "no answer"}),
    Feature('VCF0616', cat_dtype, "SHOULD THOSE WHO DON’T CARE ABOUT ELECTION "
                                  "OUTCOME VOTE",
            name_extended="those who don't care about election outcome should vote",
            value_mapping={
                '1.0': "Agree",
                '2.0': "Disagree",
                '3.0': "Neither agree nor disagree",
                '9.0': "Don't know or not sure",
                '0.0': "no answer"}),
    Feature('VCF0617', cat_dtype, "SHOULD SOMEONE VOTE IF THEIR PARTY CAN’T "
                                  "WIN",
            name_extended="someone should vote if their party can't win",
            value_mapping={
                '1.0': "Agree",
                '2.0': "Disagree",
                '9.0': "Don't know or not sure",
                '0.0': "no answer"}),
    Feature('VCF0310', cat_dtype, "INTEREST IN THE ELECTIONS",
            name_extended='interest in the elections',
            value_mapping={
                '1.0': "Not much interested ",
                '2.0': "Somewhat interested ",
                '3.0': "Very much interested ",
                '9.0': "Don't know",
                '0.0': "no answer"}),
    Feature('VCF0743', cat_dtype, "DOES R BELONG TO POLITICAL ORGANIZATION OR "
                                  "CLUB",
            name_extended='belongs to political organization or club',
            value_mapping={'1.0': "Yes", '5.0': "No", '9.0': "no answer"}),
    Feature('VCF0717', cat_dtype, "RESPONDENT TRY TO INFLUENCE THE VOTE OF "
                                  "OTHERS DURING THE CAMPAIGN",
            name_extended='tried to influence others during campaign',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),
    Feature('VCF0718', cat_dtype, "RESPONDENT ATTEND POLITICAL "
                                  "MEETINGS/RALLIES DURING THE CAMPAIGN",
            name_extended='attended political meetings/rallies during campaign',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),
    Feature('VCF0720', cat_dtype, "RESPONDENT DISPLAY CANDIDATE "
                                  "BUTTON/STICKER DURING THE CAMPAIGN",
            name_extended='displayed candidate button/sticker during campaign',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),
    Feature('VCF0721', cat_dtype, "RESPONDENT DONATE MONEY TO PARTY OR "
                                  "CANDIDATE DURING THE CAMPAIGN",
            name_extended='donated money to party or candidate during campaign',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),

    # REGISTRATION, TURNOUT, AND VOTE CHOICE
    Feature('VCF0701', cat_dtype, "REGISTERED TO VOTE PRE-ELECTION",
            name_extended='registered to vote pre-election',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),

    # MEDIA
    Feature('VCF0675', cat_dtype,
            "HOW MUCH OF THE TIME DOES RESPONDENT TRUST THE "
            "MEDIA TO REPORT FAIRLY",
            name_extended='how much of the time can you trust the media to report the news fairly',
            value_mapping={
                '1.0': "Just about always",
                '2.0': "Most of the time",
                '3.0': "Only some of the time",
                '4.0': "Almost never",
                '5.0': "Never",
                '8.0': "Don't know",
                '9.0': "no answer"}),
    Feature('VCF0724', cat_dtype, "WATCH TV PROGRAMS ABOUT THE ELECTION "
                                  "CAMPAIGNS",
            name_extended='watched TV programs about the election campaigns',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),
    Feature('VCF0725', cat_dtype, "HEAR PROGRAMS ABOUT CAMPAIGNS ON THE RADIO "
                                  "2- CATEGORY",
            name_extended='heard radio programs about the election campaigns',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),
    Feature('VCF0726', cat_dtype, "ARTICLES ABOUT ELECTION CAMPAIGNS IN "
                                  "MAGAZINES",
            name_extended='read about the election campaigns in magazines',
            value_mapping={
                '1.0': "No ",
                '2.0': "Yes",
                '0.0': "no answer"}),
    Feature('VCF0745', cat_dtype, "SAW ELECTION CAMPAIGN INFORMATION ON THE "
                                  "INTERNET",
            name_extended='saw election campaign information on the internet',
            value_mapping={
                '1.0': "Yes",
                '5.0': "No ",
                '9.0': "no answer"}),

    # PERSONAL AND DEMOGRAPHIC
    Feature('VCF0101', int, "RESPONDENT - AGE",
            name_extended='age'),
    Feature('VCF0104', cat_dtype, name_extended='gender',
            value_mapping={
                '1': "Male",
                '2': "Female",
                '3': "Other",
                '0': "no answer"}),
    Feature('VCF0105a', cat_dtype, """RACE-ETHNICITY SUMMARY, 7 CATEGORIES""",
            name_extended='race/ethnicity',
            value_mapping={
                '1.0': "White non-Hispanic",
                '2.0': "Black non-Hispanic",
                '3.0': "Asian or Pacific Islander, non-Hispanic",
                '4.0': "American Indian or Alaska Native non-Hispanic",
                '5.0': "Hispanic",
                '6.0': "Other or multiple races, non-Hispanic",
                '7.0': "Non-white and non-black",
                '9.0': "no answer"}),
    Feature('VCF0115', cat_dtype,
            """RESPONDENT - OCCUPATION GROUP 6-CATEGORY""",
            name_extended='occupation group',
            value_mapping={
                '1.0': "Professional and managerial",
                '2.0': "Clerical and sales workers",
                '3.0': "Skilled, semi-skilled and service workers",
                '4.0': "Laborers, except farm",
                '5.0': "Farmers, farm managers, farm laborers and foremen; forestry and fishermen",
                '6.0': "Homemakers",
                '0.0': "no answer"}),
    Feature('VCF0140a', cat_dtype, """RESPONDENT - EDUCATION 7-CATEGORY""",
            name_extended='education level',
            value_mapping={
                '1.0': "8 grades or less (‘grade school’)",
                '2.0': "9-12 grades (‘high school’), no diploma/equivalency; less than high school credential (2020)",
                '3.0': "12 grades, diploma or equivalency",
                '4.0': "12 grades, diploma or equivalency plus non-academic training",
                '5.0': "Some college, no degree; junior/community college level degree (AA degree)",
                '6.0': "BA level degrees",
                '7.0': "Advanced degrees incl. LLB",
                '8.0': "Don't know",
                '9.0': "no answer"}),
    Feature('VCF0112', cat_dtype, """Region - U.S. Census 1. Northeast (CT, 
    ME, MA, NH, NJ, NY, PA, RI, VT) 2. North Central (IL, IN, IA, KS, MI, MN, 
    MO, NE, ND, OH, SD, WI) 3. South (AL, AR, DE, D.C., FL, GA, KY, LA, MD, 
    MS, NC, OK, SC,TN, TX, VA, WV) 4. West (AK, AZ, CA, CO, HI, ID, MT, NV, 
    NM, OR, UT, WA, WY)""",
            name_extended='US census region',
            value_mapping={
                # (CT, ME, MA, NH, NJ, NY, PA, RI, VT)
                '1.0': "Northeast",
                # (IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI)
                '2.0': "North Central",
                # (AL, AR, DE, D.C., FL, GA, KY, LA, MD, MS, NC, OK, SC,TN, TX, VA, WV) 4. West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)
                '3.0': "South",
                # (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)
                '4.0': 'West', }),
],
    documentation="https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/")


def preprocess_anes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[ANES_FEATURES.target])
    df[ANES_FEATURES.target] = (
            df[ANES_FEATURES.target].astype(float) == 2.0).astype(int)
    for f in ANES_FEATURES.features:
        if f.kind == cat_dtype:
            df[f.name] = df[f.name].fillna("no answer").apply(str) \
                .astype("category")
    return df
