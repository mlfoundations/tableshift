"""
Various feature mappings for ACS data.
"""

from frozendict import frozendict
from tableshift.datasets.acs_occp_mapping import ACS_OCCP_CODE_MAPPING

# Maps place of birth to coarse categories for state
# or global region (i.e. Africa, Middle East, South America).
POBP_MAPPING = {
    001.: "AL",  # .Alabama/AL
    002.: "AK",  # .Alaska/AK
    004.: "AZ",  # .Arizona/AZ
    005.: "AR",  # .Arkansas/AR
    006.: "CA",  # .California/CA
    008.: "CO",  # .Colorado/CO
    009.: "CT",  # .Connecticut/CT
    010.: "DE",  # .Delaware/DE
    011.: "DC",  # .District of Columbia/DC
    012.: "FL",  # .Florida/FL
    013.: "GA",  # .Georgia/GA
    015.: "HI",  # .Hawaii/HI
    016.: "ID",  # .Idaho/ID
    017.: "IL",  # .Illinois/IL
    018.: "IN",  # .Indiana/IN
    019.: "IA",  # .Iowa/IA
    020.: "KS",  # .Kansas/KS
    021.: "KY",  # .Kentucky/KY
    022.: "LA",  # .Louisiana/LA
    023.: "ME",  # .Maine/ME
    024.: "MD",  # .Maryland/MD
    025.: "MA",  # .Massachusetts/MA
    026.: "MI",  # .Michigan/MI
    027.: "MN",  # .Minnesota/MN
    028.: "MS",  # .Mississippi/MS
    029.: "MO",  # .Missouri/MO
    030.: "MT",  # .Montana/MT
    031.: "NE",  # .Nebraska/NE
    032.: "NV",  # .Nevada/NV
    033.: "NH",  # .New Hampshire/NH
    034.: "NJ",  # .New Jersey/NJ
    035.: "NM",  # .New Mexico/NM
    036.: "NY",  # .New York/NY
    037.: "NC",  # .North Carolina/NC
    038.: "ND",  # .North Dakota/ND
    039.: "OH",  # .Ohio/OH
    040.: "OK",  # .Oklahoma/OK
    041.: "OR",  # .Oregon/OR
    042.: "PA",  # .Pennsylvania/PA
    044.: "RI",  # .Rhode Island/RI
    045.: "SC",  # .South Carolina/SC
    046.: "SD",  # .South Dakota/SD
    047.: "TN",  # .Tennessee/TN
    048.: "TX",  # .Texas/TX
    049.: "UT",  # .Utah/UT
    050.: "VT",  # .Vermont/VT
    051.: "VA",  # .Virginia/VA
    053.: "WA",  # .Washington/WA
    054.: "WV",  # .West Virginia/WV
    055.: "WI",  # .Wisconsin/WI
    056.: "WY",  # .Wyoming/WY
    060.: "USTERR",  # .American Samoa
    066.: "USTERR",  # .Guam
    069.: "USTERR",  # .Commonwealth of the Northern Mariana Islands
    072.: "PR",  # .Puerto Rico
    078.: "USTERR",  # .US Virgin Islands
    100.: "EUROPE",  # .Albania
    102.: "EUROPE",  # .Austria
    103.: "EUROPE",  # .Belgium
    104.: "EUROPE",  # .Bulgaria
    105.: "EUROPE",  # .Czechoslovakia
    106.: "EUROPE",  # .Denmark
    108.: "EUROPE",  # .Finland
    109.: "EUROPE",  # .France
    110.: "EUROPE",  # .Germany
    116.: "EUROPE",  # .Greece
    117.: "EUROPE",  # .Hungary
    118.: "EUROPE",  # .Iceland
    119.: "EUROPE",  # .Ireland
    120.: "EUROPE",  # .Italy
    126.: "EUROPE",  # .Netherlands
    127.: "EUROPE",  # .Norway
    128.: "EUROPE",  # .Poland
    129.: "EUROPE",  # .Portugal
    130.: "EUROPE",  # .Azores Islands
    132.: "EUROPE",  # .Romania
    134.: "EUROPE",  # .Spain
    136.: "EUROPE",  # .Sweden
    137.: "EUROPE",  # .Switzerland
    138.: "EUROPE",  # .United Kingdom, Not Specified
    139.: "EUROPE",  # .England
    140.: "EUROPE",  # .Scotland
    142.: "EUROPE",  # .Northern Ireland (201.: "", #7 or later)
    147.: "EUROPE",  # .Yugoslavia
    148.: "EUROPE",  # .Czech Republic
    149.: "EUROPE",  # .Slovakia
    150.: "EUROPE",  # .Bosnia and Herzegovina
    151.: "EUROPE",  # .Croatia
    152.: "EUROPE",  # .Macedonia
    154.: "EUROPE",  # .Serbia
    156.: "EUROPE",  # .Latvia
    157.: "EUROPE",  # .Lithuania
    158.: "EUROPE",  # .Armenia
    159.: "EUROPE",  # .Azerbaijan
    160.: "EUROPE",  # .Belarus
    161.: "EUROPE",  # .Georgia
    162.: "EUROPE",  # .Moldova
    163.: "ASIA",  # .Russia
    164.: "ASIA",  # .Ukraine
    165.: "ASIA",  # .USSR
    166.: "EUROPE",  # .Europe (201.: "", #7 or later)
    167.: "EUROPE",  # .Kosovo (201.: "", #7 or later)
    168.: "EUROPE",  # .Montenegro
    169.: "EUROPE",  # .Other Europe, Not Specified
    200.: "MIDDLEEAST",  # .Afghanistan
    202.: "ASIA",  # .Bangladesh
    203.: "ASIA",  # .Bhutan
    205.: "ASIA",  # .Myanmar
    206.: "ASIA",  # .Cambodia
    207.: "ASIA",  # .China
    208.: "EUROPE",  # .Cyprus (201.: "", #6 or earlier)
    209.: "ASIA",  # .Hong Kong
    210.: "ASIA",  # .India
    211.: "ASIA",  # .Indonesia
    212.: "MIDDLEEAST",  # .Iran
    213.: "MIDDLEEAST",  # .Iraq
    214.: "MIDDLEEAST",  # .Israel
    215.: "ASIA",  # .Japan
    216.: "MIDDLEEAST",  # .Jordan
    217.: "ASIA",  # .Korea
    218.: "MIDDLEEAST",  # .Kazakhstan
    219.: "MIDDLEEAST",  # .Kyrgyzstan (201.: "", #7 or later)
    222.: "MIDDLEEAST",  # .Kuwait
    223.: "ASIA",  # .Laos
    224.: "MIDDLEEAST",  # .Lebanon
    226.: "ASIA",  # .Malaysia
    228.: "ASIA",  # .Mongolia (201.: "", #7 or later)
    229.: "ASIA",  # .Nepal
    231.: "ASIA",  # .Pakistan
    233.: "ASIA",  # .Philippines
    235.: "ASIA",  # .Saudi Arabia
    236.: "ASIA",  # .Singapore
    238.: "ASIA",  # .Sri Lanka
    239.: "ASIA",  # .Syria
    240.: "ASIA",  # .Taiwan
    242.: "ASIA",  # .Thailand
    243.: "EUROPE",  # .Turkey
    245.: "MIDDLEEAST",  # .United Arab Emirates
    246.: "MIDDLEEAST",  # .Uzbekistan
    247.: "ASIA",  # .Vietnam
    248.: "MIDDLEEAST",  # .Yemen
    249.: "ASIA",  # .Asia
    253.: "ASIA",  # .South Central Asia, Not Specified
    254.: "ASIA",  # .Other Asia, Not Specified
    300.: "CARIBBEAN",  # .Bermuda
    301.: "CANADA",  # .Canada
    303.: "MEXICO",  # .Mexico
    310.: "CENTRALAMERICA",  # .Belize
    311.: "CENTRALAMERICA",  # .Costa Rica
    312.: "CENTRALAMERICA",  # .El Salvador
    313.: "CENTRALAMERICA",  # .Guatemala
    314.: "CENTRALAMERICA",  # .Honduras
    315.: "CENTRALAMERICA",  # .Nicaragua
    316.: "CENTRALAMERICA",  # .Panama
    321.: "CARIBBEAN",  # .Antigua and Barbuda
    323.: "CARIBBEAN",  # .Bahamas
    324.: "CARIBBEAN",  # .Barbados
    327.: "CARIBBEAN",  # .Cuba
    328.: "CARIBBEAN",  # .Dominica
    329.: "CARIBBEAN",  # .Dominican Republic
    330.: "CARIBBEAN",  # .Grenada
    332.: "CARIBBEAN",  # .Haiti
    333.: "CARIBBEAN",  # .Jamaica
    338.: "CARIBBEAN",  # .St. Kitts-Nevis (201.: "", #7 or later)
    339.: "CARIBBEAN",  # .St. Lucia
    340.: "CARIBBEAN",  # .St. Vincent and the Grenadines
    341.: "CARIBBEAN",  # .Trinidad and Tobago
    343.: "CARIBBEAN",  # .West Indies
    344.: "CARIBBEAN",  # .Caribbean, Not Specified
    360.: "SOUTHAMERICA",  # .Argentina
    361.: "SOUTHAMERICA",  # .Bolivia
    362.: "SOUTHAMERICA",  # .Brazil
    363.: "SOUTHAMERICA",  # .Chile
    364.: "SOUTHAMERICA",  # .Colombia
    365.: "SOUTHAMERICA",  # .Ecuador
    368.: "SOUTHAMERICA",  # .Guyana
    369.: "SOUTHAMERICA",  # .Paraguay
    370.: "SOUTHAMERICA",  # .Peru
    372.: "SOUTHAMERICA",  # .Uruguay
    373.: "SOUTHAMERICA",  # .Venezuela
    374.: "SOUTHAMERICA",  # .South America
    399.: "SOUTHAMERICA",  # .Americas, Not Specified
    400.: "AFRICA",  # .Algeria
    407.: "AFRICA",  # .Cameroon
    408.: "AFRICA",  # .Cabo Verde
    412.: "AFRICA",  # .Congo
    414.: "AFRICA",  # .Egypt
    416.: "AFRICA",  # .Ethiopia
    417.: "AFRICA",  # .Eritrea
    420.: "AFRICA",  # .Gambia
    421.: "AFRICA",  # .Ghana
    423.: "AFRICA",  # .Guinea
    425.: "AFRICA",  # .Ivory Coast (201.: "", #7 or later)
    427.: "AFRICA",  # .Kenya
    429.: "AFRICA",  # .Liberia
    430.: "AFRICA",  # .Libya
    436.: "AFRICA",  # .Morocco
    440.: "AFRICA",  # .Nigeria
    442.: "AFRICA",  # .Rwanda (201.: "", #7 or later)
    444.: "AFRICA",  # .Senegal
    447.: "AFRICA",  # .Sierra Leone
    448.: "AFRICA",  # .Somalia
    449.: "AFRICA",  # .South Africa
    451.: "AFRICA",  # .Sudan
    453.: "AFRICA",  # .Tanzania
    454.: "AFRICA",  # .Togo
    456.: "AFRICA",  # .Tunisia (201.: "", #7 or later)
    457.: "AFRICA",  # .Uganda
    459.: "AFRICA",  # .Democratic Republic of Congo (Zaire)
    460.: "AFRICA",  # .Zambia
    461.: "AFRICA",  # .Zimbabwe
    462.: "AFRICA",  # .Africa
    463.: "AFRICA",  # .South Sudan (201.: "", #7 or later)
    464.: "AFRICA",  # .Northern Africa, Not Specified
    467.: "AFRICA",  # .Western Africa, Not Specified
    468.: "AFRICA",  # .Other Africa, Not Specified
    469.: "AFRICA",  # .Eastern Africa, Not Specified
    501.: "OCEANIA",  # .Australia
    508.: "OCEANIA",  # .Fiji
    511.: "OCEANIA",  # .Marshall Islands
    512.: "OCEANIA",  # .Micronesia
    515.: "OCEANIA",  # .New Zealand
    523.: "OCEANIA",  # .Tonga
    527.: "OCEANIA",  # .Samoa
    554.: "OCEANIA",
    # .Other US Island Areas, Oceania, Not Specified, or at Sea
}

# List of 630 unique codes that occur in the data
ALL_CODES = [10.0, 20.0, 40.0, 50.0, 51.0, 52.0, 60.0, 100.0, 101.0, 102.0,
             110.0, 120.0, 135.0, 136.0, 137.0, 140.0, 150.0, 160.0, 205.0,
             220.0, 230.0, 300.0, 310.0, 330.0, 335.0, 340.0, 350.0, 360.0,
             410.0, 420.0, 425.0, 430.0, 440.0, 500.0, 510.0, 520.0, 530.0,
             540.0, 565.0, 600.0, 630.0, 640.0, 650.0, 700.0, 705.0, 710.0,
             725.0, 726.0, 735.0, 740.0, 750.0, 800.0, 810.0, 820.0, 830.0,
             840.0, 845.0, 850.0, 860.0, 900.0, 910.0, 930.0, 940.0, 950.0,
             960.0, 1005.0, 1006.0, 1007.0, 1010.0, 1020.0, 1021.0, 1022.0,
             1030.0, 1031.0, 1032.0, 1050.0, 1060.0, 1065.0, 1105.0, 1106.0,
             1107.0, 1108.0, 1200.0, 1220.0, 1240.0, 1300.0, 1305.0, 1306.0,
             1310.0, 1320.0, 1340.0, 1350.0, 1360.0, 1400.0, 1410.0, 1420.0,
             1430.0, 1440.0, 1450.0, 1460.0, 1520.0, 1530.0, 1540.0, 1541.0,
             1545.0, 1550.0, 1551.0, 1555.0, 1560.0, 1600.0, 1610.0, 1640.0,
             1650.0, 1700.0, 1710.0, 1720.0, 1740.0, 1745.0, 1750.0, 1760.0,
             1800.0, 1820.0, 1821.0, 1822.0, 1825.0, 1840.0, 1860.0, 1900.0,
             1910.0, 1920.0, 1930.0, 1935.0, 1965.0, 1970.0, 1980.0, 2000.0,
             2001.0, 2002.0, 2003.0, 2004.0, 2005.0, 2006.0, 2010.0, 2011.0,
             2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2025.0, 2040.0, 2050.0,
             2060.0, 2100.0, 2105.0, 2145.0, 2160.0, 2170.0, 2180.0, 2200.0,
             2205.0, 2300.0, 2310.0, 2320.0, 2330.0, 2340.0, 2350.0, 2360.0,
             2400.0, 2430.0, 2435.0, 2440.0, 2540.0, 2545.0, 2550.0, 2555.0,
             2600.0, 2630.0, 2631.0, 2632.0, 2633.0, 2634.0, 2635.0, 2636.0,
             2640.0, 2700.0, 2710.0, 2720.0, 2721.0, 2722.0, 2723.0, 2740.0,
             2750.0, 2751.0, 2752.0, 2755.0, 2760.0, 2770.0, 2800.0, 2805.0,
             2810.0, 2825.0, 2830.0, 2840.0, 2850.0, 2860.0, 2861.0, 2862.0,
             2865.0, 2900.0, 2905.0, 2910.0, 2920.0, 3000.0, 3010.0, 3030.0,
             3040.0, 3050.0, 3060.0, 3090.0, 3100.0, 3110.0, 3120.0, 3140.0,
             3150.0, 3160.0, 3200.0, 3210.0, 3220.0, 3230.0, 3245.0, 3250.0,
             3255.0, 3256.0, 3258.0, 3260.0, 3261.0, 3270.0, 3300.0, 3310.0,
             3320.0, 3321.0, 3322.0, 3323.0, 3324.0, 3330.0, 3400.0, 3401.0,
             3402.0, 3420.0, 3421.0, 3422.0, 3423.0, 3424.0, 3430.0, 3500.0,
             3510.0, 3515.0, 3520.0, 3535.0, 3540.0, 3545.0, 3550.0, 3600.0,
             3601.0, 3602.0, 3603.0, 3605.0, 3610.0, 3620.0, 3630.0, 3640.0,
             3645.0, 3646.0, 3647.0, 3648.0, 3649.0, 3655.0, 3700.0, 3710.0,
             3720.0, 3725.0, 3730.0, 3740.0, 3750.0, 3800.0, 3801.0, 3802.0,
             3820.0, 3840.0, 3850.0, 3870.0, 3900.0, 3910.0, 3930.0, 3940.0,
             3945.0, 3946.0, 3955.0, 3960.0, 4000.0, 4010.0, 4020.0, 4030.0,
             4040.0, 4050.0, 4055.0, 4060.0, 4110.0, 4120.0, 4130.0, 4140.0,
             4150.0, 4160.0, 4200.0, 4210.0, 4220.0, 4230.0, 4240.0, 4250.0,
             4251.0, 4252.0, 4255.0, 4300.0, 4320.0, 4330.0, 4340.0, 4350.0,
             4400.0, 4410.0, 4420.0, 4430.0, 4435.0, 4460.0, 4461.0, 4465.0,
             4500.0, 4510.0, 4520.0, 4521.0, 4522.0, 4525.0, 4530.0, 4540.0,
             4600.0, 4610.0, 4620.0, 4621.0, 4622.0, 4640.0, 4650.0, 4655.0,
             4700.0, 4710.0, 4720.0, 4740.0, 4750.0, 4760.0, 4800.0, 4810.0,
             4820.0, 4830.0, 4840.0, 4850.0, 4900.0, 4920.0, 4930.0, 4940.0,
             4950.0, 4965.0, 5000.0, 5010.0, 5020.0, 5030.0, 5040.0, 5100.0,
             5110.0, 5120.0, 5130.0, 5140.0, 5150.0, 5160.0, 5165.0, 5200.0,
             5220.0, 5230.0, 5240.0, 5250.0, 5260.0, 5300.0, 5310.0, 5320.0,
             5330.0, 5340.0, 5350.0, 5360.0, 5400.0, 5410.0, 5420.0, 5500.0,
             5510.0, 5520.0, 5521.0, 5522.0, 5530.0, 5540.0, 5550.0, 5560.0,
             5600.0, 5610.0, 5620.0, 5630.0, 5700.0, 5710.0, 5720.0, 5730.0,
             5740.0, 5800.0, 5810.0, 5820.0, 5840.0, 5850.0, 5860.0, 5900.0,
             5910.0, 5920.0, 5940.0, 6005.0, 6010.0, 6040.0, 6050.0, 6100.0,
             6115.0, 6120.0, 6130.0, 6200.0, 6210.0, 6220.0, 6230.0, 6240.0,
             6250.0, 6260.0, 6300.0, 6305.0, 6320.0, 6330.0, 6355.0, 6360.0,
             6400.0, 6410.0, 6420.0, 6440.0, 6441.0, 6442.0, 6460.0, 6515.0,
             6520.0, 6530.0, 6540.0, 6600.0, 6660.0, 6700.0, 6710.0, 6720.0,
             6730.0, 6740.0, 6765.0, 6800.0, 6820.0, 6825.0, 6830.0, 6835.0,
             6840.0, 6850.0, 6940.0, 6950.0, 7000.0, 7010.0, 7020.0, 7030.0,
             7040.0, 7100.0, 7110.0, 7120.0, 7130.0, 7140.0, 7150.0, 7160.0,
             7200.0, 7210.0, 7220.0, 7240.0, 7260.0, 7300.0, 7315.0, 7320.0,
             7330.0, 7340.0, 7350.0, 7360.0, 7410.0, 7420.0, 7430.0, 7510.0,
             7540.0, 7560.0, 7610.0, 7630.0, 7640.0, 7700.0, 7710.0, 7720,
             7730.0, 7740.0, 7750.0, 7800.0, 7810.0, 7830.0, 7840.0, 7850.0,
             7855.0, 7900.0, 7905.0, 7920.0, 7925.0, 7930.0, 7940.0, 7950.0,
             8000.0, 8025.0, 8030.0, 8040.0, 8100.0, 8130.0, 8140.0, 8220.0,
             8225.0, 8250.0, 8255.0, 8256.0, 8300.0, 8310.0, 8320.0, 8330.0,
             8335.0, 8350.0, 8365.0, 8400.0, 8410.0, 8420.0, 8450.0, 8460.0,
             8465.0, 8500.0, 8510.0, 8530.0, 8540.0, 8550.0, 8555.0, 8600.0,
             8610.0, 8620.0, 8630.0, 8640.0, 8650.0, 8710.0, 8720.0, 8730.0,
             8740.0, 8750.0, 8760.0, 8800.0, 8810.0, 8830.0, 8850.0, 8910.0,
             8920.0, 8930.0, 8940.0, 8950.0, 8965.0, 8990.0, 9000.0, 9005.0,
             9030.0, 9040.0, 9050.0, 9110.0, 9120.0, 9121.0, 9122.0, 9130.0,
             9140.0, 9141.0, 9142.0, 9150.0, 9200.0, 9210.0, 9240.0, 9260.0,
             9265.0, 9300.0, 9310.0, 9350.0, 9360.0, 9365.0, 9410.0, 9415.0,
             9420.0, 9430.0, 9510.0, 9520.0, 9560.0, 9570.0, 9600.0, 9610.0,
             9620.0, 9630.0, 9640.0, 9645.0, 9650.0, 9720.0, 9750.0, 9760.0,
             9800.0, 9810.0, 9820.0, 9825.0, 9830.0, 9920.0]

OCCP_MAPPING_IDENTITY = {
    k: str(int(k)) for k in ALL_CODES
}


def _float_to_string_mapping(minval, maxval):
    return {float(x): f"{x:02d}" for x in range(minval, maxval + 1)}


DEFAULT_ACS_FEATURE_MAPPINGS = {
    # Ancestry recode (included to match folktables feature sets)
    'ANC': _float_to_string_mapping(1, 8),
    # Citizenship
    'CIT': _float_to_string_mapping(1, 5),
    # "Class of worker"
    'COW': _float_to_string_mapping(1, 9),
    # Hearing difficulty
    'DEAR': _float_to_string_mapping(1, 2),
    # Vision difficulty
    'DEYE': _float_to_string_mapping(1, 2),
    # Cognitive difficulty
    'DREM': _float_to_string_mapping(1, 2),
    # Division code based on 2010 Census
    'DIVISION': _float_to_string_mapping(0, 9),
    # Ability to speak English
    'ENG': _float_to_string_mapping(0, 4),
    # Employment status of parents
    'ESP': _float_to_string_mapping(0, 8),
    # Employment Status Recode
    'ESR': _float_to_string_mapping(0, 6),
    # Gave birth to child within the past 12 months
    'FER': _float_to_string_mapping(0, 2),
    # Insurance through a current or former employer or union
    'HINS1': _float_to_string_mapping(1, 2),
    # Insurance purchased directly from an insurance company
    'HINS2': _float_to_string_mapping(1, 2),
    # Medicare, for people 65 and older, or people with certain disabilities
    'HINS3': _float_to_string_mapping(1, 2),
    # Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability
    'HINS4': _float_to_string_mapping(1, 2),
    # Marital status.
    'MAR': _float_to_string_mapping(0, 5),
    # Mobility status
    'MIG': _float_to_string_mapping(1, 3),
    # Military service
    'MIL': _float_to_string_mapping(0, 4),
    # Nativity
    'NATIVITY': _float_to_string_mapping(1, 2),
    # On layoff from work (Unedited-See "Employment Status Recode" (ESR))
    'NWLA': _float_to_string_mapping(1, 3),
    # Looking for work (Unedited-See "Employment Status Recode" (ESR))
    'NWLK': _float_to_string_mapping(1, 3),
    # # Occupation recode for 2018 and later based on 2018 OCC codes.
    # 'OCCP': OCCP_MAPPING_FINE,
    # Place of birth.
    'POBP': POBP_MAPPING,
    # Relationship
    'RELP': _float_to_string_mapping(0, 17),
    # Educational attainment
    'SCHL': _float_to_string_mapping(1, 24),
    'ST': _float_to_string_mapping(1, 72),
    # Worked last week
    'WRK': _float_to_string_mapping(0, 2)
}

OCCP_MAPPINGS = {
    'identity': OCCP_MAPPING_IDENTITY,
    'coarse': {k: v[0] for k, v in ACS_OCCP_CODE_MAPPING.items()},
    'fine': {k: v[1] for k, v in ACS_OCCP_CODE_MAPPING.items()},
}


def get_feature_mapping(occp_mapping='coarse') -> frozendict:
    """Helper function to fetch feature mapping dict.

    Returns a nested dict mapping feature names to a mapping;
    the mapping assigns each possible value of the feature
    to a new set of values (in most cases this is either
    a 1:1 mapping or a many:1 mapping to reduce cardinality).
    """
    assert occp_mapping in ('identity', 'coarse', 'fine')
    mapping = DEFAULT_ACS_FEATURE_MAPPINGS
    mapping['OCCP'] = OCCP_MAPPINGS[occp_mapping]
    return frozendict(mapping)
