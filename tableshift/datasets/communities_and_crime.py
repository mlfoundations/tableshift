import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

NA_VALUES = ("?",)

CANDC_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/communities/communities.data"]

# Column names of the raw input features from UCI.
CANDC_INPUT_FEATURES = [
    'state', 'county', 'community', 'communityname', 'fold', 'population',
    'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian',
    'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf',
    'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc',
    'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
    'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov',
    'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu',
    'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
    'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
    'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
    'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
    'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
    'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup',
    'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos',
    'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart',
    'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ',
    'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
    'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn',
    'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
    'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps',
    'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
    'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite',
    'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor',
    'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea',
    'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
    'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn',
    'PolicBudgPerPop', 'ViolentCrimesPerPop'
]

CANDC_STATE_LIST = ['1', '10', '11', '12', '13', '16', '18', '19', '2', '20',
                    '21', '22', '23', '24', '25', '27', '28', '29', '32', '33',
                    '34', '35', '36', '37', '38', '39', '4', '40', '41', '42',
                    '44', '45', '46', '47', '48', '49', '5', '50', '51', '53',
                    '54', '55', '56', '6', '8', '9']

CANDC_FEATURES = FeatureList([
    Feature('Target', int,
            name_extended="Binary indicator for whether total number of violent crimes per 100K population exceeds threshold",
            is_target=True),
    Feature('PctKids2Par', float,
            name_extended='percentage of kids in family housing with two parents',
            na_values=NA_VALUES),  # importance: 0.3411
    Feature('pctUrban', float,
            name_extended='percentage of people living in areas classified as urban',
            na_values=NA_VALUES),  # importance: 0.1829
    Feature('NumIlleg', float,
            name_extended='number of kids born to never married',
            na_values=NA_VALUES),  # importance: 0.1176
    Feature('PctSameCity85', float,
            name_extended='percent of people living in the same city as in 1985 (5 years before)',
            na_values=NA_VALUES),  # importance: 0.1145
    Feature('numbUrban', float,
            name_extended='number of people living in areas classified as urban',
            na_values=NA_VALUES),  # importance: 0.0863
    Feature('PctNotHSGrad', float,
            name_extended='percentage of people 25 and over that are not high school graduates',
            na_values=NA_VALUES),  # importance: 0.0823
    Feature('MedRentPctHousInc', float,
            name_extended='median gross rent as a percentage of household income',
            na_values=NA_VALUES),  # importance: 0.0291
    Feature('HousVacant', float, name_extended='number of vacant households',
            na_values=NA_VALUES),  # importance: 0.0183
    Feature('agePct16t24', float,
            name_extended='percentage of population that is 16-24 in age',
            na_values=NA_VALUES),  # importance: 0.0147
    Feature('LemasPctOfficDrugUn', float,
            name_extended='percent of officers assigned to drug units',
            na_values=NA_VALUES),  # importance: 0.0133
    Feature('RentHighQ', float,
            name_extended='rental housing - upper quartile rent',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctRecImmig8', float,
            name_extended='percent of population who have immigrated within the last 8 years',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('indianPerCap', float,
            name_extended='per capita income for native americans',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('pctWInvInc', float,
            name_extended='percentage of households with investment/rent income in 1989',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctBornSameState', float,
            name_extended='percent of people born in the same state as currently living',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('medFamInc', float, name_extended='median family income',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctSameHouse85', float,
            name_extended='percent of people living in the same house as in 1985 (5 years before)',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctTeen2Par', float,
            name_extended='percent of kids age 12-17 in two parent households',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PersPerOccupHous', float,
            name_extended='mean persons per household', na_values=NA_VALUES),
    # importance: 0.0
    Feature('OtherPerCap', float,
            name_extended='per capita income for people with other heritage',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctOccupMgmtProf', float,
            name_extended='percentage of people 16 and over who are employed in management or professional occupations',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PolicAveOTWorked', float,
            name_extended='police average overtime worked', na_values=NA_VALUES),
    # importance: 0.0
    Feature('OwnOccLowQuart', float,
            name_extended='owner occupied housing - lower quartile value',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctLargHouseOccup', float,
            name_extended='percent of all occupied households that are large (6 or more)',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('MedNumBR', float, name_extended='median number of bedrooms',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PolicPerPop', float,
            name_extended='police officers per 100K population',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('agePct65up', float,
            name_extended='percentage of population that is 65 and over in age',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctRecentImmig', float,
            name_extended='percent of population who have immigrated within the last 3 years',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('LemasGangUnitDeploy', float, name_extended='gang unit deployed',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PolicCars', float, name_extended='number of police cars',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('PctForeignBorn', float,
            name_extended='percent of people foreign born', na_values=NA_VALUES),
    # importance: 0.0
    Feature('PctImmigRec8', float,
            name_extended='percentage of immigrants who immigated within last 8 years',
            na_values=NA_VALUES),  # importance: 0.0
    Feature('agePct12t21', float,
            name_extended='percentage of population that is 12-21 in age',
            na_values=NA_VALUES),  # importance: 0.0
    ##################################################
    ##################################################
    # Feature('PctLargHouseFam', float,
    #         name_extended='percent of family households that are large (6 or more)',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('racePctWhite', float,
    #         name_extended='percentage of population that is caucasian',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('state', int, name_extended='US state (by number)',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('racePctHisp', float,
    #         name_extended='percentage of population that is of hispanic heritage',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPopUnderPov', float,
    #         name_extended='percentage of people under the poverty level',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctSpeakEnglOnly', float,
    #         name_extended='percent of people who speak only English',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PopDens', float,
    #         name_extended='population density in persons per square mile',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctEmploy', float,
    #         name_extended='percentage of people 16 and over who are employed',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PersPerFam', float,
    #         name_extended='mean number of people per family', na_values=NA_VALUES),
    # # importance: 0.0
    # Feature('PolicReqPerOffic', float,
    #         name_extended='total requests for police per police officer',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctImmigRec5', float,
    #         name_extended='percentage of immigrants who immigated within last 5 years',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctVacantBoarded', float,
    #         name_extended='percent of vacant housing that is boarded up',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('population', float, name_extended='population', na_values=NA_VALUES),
    # # importance: 0.0
    # Feature('NumImmig', float,
    #         name_extended='total number of people known to be foreign born',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('OwnOccMedVal', float,
    #         name_extended='owner occupied housing - median value',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('MalePctNevMarr', float,
    #         name_extended='percentage of males who have never married',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasTotReqPerPop', float,
    #         name_extended='total requests for police per 100K popuation',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('NumUnderPov', float,
    #         name_extended='number of people under the poverty level',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctUnemployed', float,
    #         name_extended='percentage of people 16 and over, in the labor force, and unemployed',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPolicWhite', float,
    #         name_extended='percent of police that are caucasian',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasSwFTFieldOps', float,
    #         name_extended='number of sworn full time police officers in field operations',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PersPerOwnOccHous', float,
    #         name_extended='mean persons per owner occupied household',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('RentMedian', float, name_extended='rental housing - median rent',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPersDenseHous', float,
    #         name_extended='percent of persons in dense housing (more than 1 person per room)',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctWorkMom', float,
    #         name_extended='percentage of moms of kids under 18 in labor force',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('blackPerCap', float,
    #         name_extended='per capita income for african americans',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('OfficAssgnDrugUnits', float,
    #         name_extended='number of officers assigned to special drug units',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('RacialMatchCommPol', float,
    #         name_extended='measure of the racial match between the community and the police force',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctImmigRec10', float,
    #         name_extended='percentage of immigrants who immigated within last 10 years',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('racePctAsian', float,
    #         name_extended='percentage of population that is of asian heritage',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('county', float, name_extended='numeric code for county',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('OwnOccHiQuart', float,
    #         name_extended='owner occupied housing - upper quartile value',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctHousOwnOcc', float,
    #         name_extended='percent of households owner occupied',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctHousNoPhone', float,
    #         name_extended='percent of occupied housing units without phone',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('racepctblack', float,
    #         name_extended='percentage of population that is african american',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctSameState85', float,
    #         name_extended='percent of people living in the same state as in 1985 (5 years before)',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasTotalReq', float, name_extended='total requests for police',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('TotalPctDiv', float,
    #         name_extended='percentage of population who are divorced',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctVacMore6Mos', float,
    #         name_extended='percent of vacant housing that has been vacant more than 6 months',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('pctWRetire', float,
    #         name_extended='percentage of households with retirement income in 1989',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('pctWFarmSelf', float,
    #         name_extended='percentage of households with farm or self employment income in 1989',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('AsianPerCap', float,
    #         name_extended='per capita income for people with asian heritage',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('pctWWage', float,
    #         name_extended='percentage of households with wage or salary income in 1989',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctWorkMomYoungKids', float,
    #         name_extended='percentage of moms of kids 6 and under in labor force',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LandArea', float, name_extended='land area in square miles',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctNotSpeakEnglWell', float,
    #         name_extended='percent of people who do not speak English well',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPolicBlack', float,
    #         name_extended='percent of police that are african american',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasSwFTFieldPerPop', float,
    #         name_extended='sworn full time police officers in field operations per 100K population',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('pctWSocSec', float,
    #         name_extended='percentage of households with social security income in 1989',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('agePct12t29', float,
    #         name_extended='percentage of population that is 12-29 in age',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('HispPerCap', float,
    #         name_extended='per capita income for people with hispanic heritage',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('whitePerCap', float,
    #         name_extended='per capita income for caucasians', na_values=NA_VALUES),
    # # importance: 0.0
    # Feature('PctPolicHisp', float,
    #         name_extended='percent of police that are hispanic',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('communityname', cat_dtype, name_extended='community name',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasSwornFT', float,
    #         name_extended='number of sworn full time police officers',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('MedOwnCostPctIncNoMtg', float,
    #         name_extended='median owners cost as a percentage of household income',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PolicBudgPerPop', float,
    #         name_extended='police operating budget per population',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPolicMinor', float,
    #         name_extended='percent of police that are minority of any kind',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctYoungKids2Par', float,
    #         name_extended='percent of kids 4 and under in two parent households',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPolicAsian', float,
    #         name_extended='percent of police that are asian', na_values=NA_VALUES),
    # # importance: 0.0
    # Feature('medIncome', float, name_extended='median household income',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('FemalePctDiv', float,
    #         name_extended='percentage of females who are divorced',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasSwFTPerPop', float,
    #         name_extended='sworn full time police officers per 100K population',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctIlleg', float,
    #         name_extended='percentage of kids born to never married',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctRecImmig10', float,
    #         name_extended='percent of population who have immigrated within the last 10 years',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctImmigRecent', float,
    #         name_extended='percentage of immigrants who immigated within last 3 years',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctEmplManu', float,
    #         name_extended='percentage of people 16 and over who are employed in manufacturing',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('perCapInc', float, name_extended='per capita income',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('community', float, name_extended='numeric code for community',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('NumStreet', float,
    #         name_extended='number of homeless people counted in the street',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('MedOwnCostPctInc', float,
    #         name_extended='median owners cost as a percentage of household income',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctLess9thGrade', float,
    #         name_extended='percentage of people 25 and over with less than a 9th grade education',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('pctWPubAsst', float,
    #         name_extended='percentage of households with public assistance income in 1989',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('NumKindsDrugsSeiz', float,
    #         name_extended='number of different kinds of drugs seized',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PersPerRentOccHous', float,
    #         name_extended='mean persons per rental household',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctEmplProfServ', float,
    #         name_extended='percentage of people 16 and over who are employed in professional services',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctFam2Par', float,
    #         name_extended='percentage of families (with kids) that are headed by two parents',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('MalePctDivorce', float,
    #         name_extended='percentage of males who are divorced',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('MedYrHousBuilt', float,
    #         name_extended='median year housing units built', na_values=NA_VALUES),
    # # importance: 0.0
    # Feature('RentLowQ', float,
    #         name_extended='rental housing - lower quartile rent',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('MedRent', float,
    #         name_extended='median gross rent (including utilities)',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctHousLess3BR', float,
    #         name_extended='percent of housing units with less than 3 bedrooms',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctPersOwnOccup', float,
    #         name_extended='percent of people in owner occupied households',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctOccupManu', float,
    #         name_extended='percentage of people 16 and over who are employed in manufacturing',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PolicOperBudg', float, name_extended='police operating budget',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctWOFullPlumb', float,
    #         name_extended='percent of housing without complete plumbing facilities',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('LemasPctPolicOnPatr', float,
    #         name_extended='percent of sworn full time police officers on patrol',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('NumInShelters', float,
    #         name_extended='number of people in homeless shelters',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctRecImmig5', float,
    #         name_extended='percent of population who have immigrated within the last 5 years',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctHousOccup', float, name_extended='percent of housing occupied',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctUsePubTrans', float,
    #         name_extended='percent of people using public transit for commuting',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('householdsize', float, name_extended='mean people per household',
    #         na_values=NA_VALUES),  # importance: 0.0
    # Feature('PctBSorMore', float,
    #         name_extended='percentage of people 25 and over with a bachelors degree or higher education',
    #         na_values=NA_VALUES),  # importance: 0.0


],
    documentation='https://archive.ics.uci.edu/ml/datasets/communities+and+crime')


def preprocess_candc(df: pd.DataFrame,
                     target_threshold: float = 0.08) -> pd.DataFrame:
    df = df.rename(columns={'ViolentCrimesPerPop': "Target"})

    # The label of a community is 1 if that community is among the
    # 70% of communities with the highest crime rate and 0 otherwise,
    # following Khani et al. and Kearns et al. 2018.
    df["Target"] = (df["Target"] >= target_threshold).astype(int)

    return df
