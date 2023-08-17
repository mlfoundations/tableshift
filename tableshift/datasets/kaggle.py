"""
Utilities for Kaggle competition data sources.

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift
"""
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

OTTO_FEATURES = FeatureList(features=[
    Feature('id', int),
    Feature('feat_1', int),
    Feature('feat_2', int),
    Feature('feat_3', int),
    Feature('feat_4', int),
    Feature('feat_5', int),
    Feature('feat_6', int),
    Feature('feat_7', int),
    Feature('feat_8', int),
    Feature('feat_9', int),
    Feature('feat_10', int),
    Feature('feat_11', int),
    Feature('feat_12', int),
    Feature('feat_13', int),
    Feature('feat_14', int),
    Feature('feat_15', int),
    Feature('feat_16', int),
    Feature('feat_17', int),
    Feature('feat_18', int),
    Feature('feat_19', int),
    Feature('feat_20', int),
    Feature('feat_21', int),
    Feature('feat_22', int),
    Feature('feat_23', int),
    Feature('feat_24', int),
    Feature('feat_25', int),
    Feature('feat_26', int),
    Feature('feat_27', int),
    Feature('feat_28', int),
    Feature('feat_29', int),
    Feature('feat_30', int),
    Feature('feat_31', int),
    Feature('feat_32', int),
    Feature('feat_33', int),
    Feature('feat_34', int),
    Feature('feat_35', int),
    Feature('feat_36', int),
    Feature('feat_37', int),
    Feature('feat_38', int),
    Feature('feat_39', int),
    Feature('feat_40', int),
    Feature('feat_41', int),
    Feature('feat_42', int),
    Feature('feat_43', int),
    Feature('feat_44', int),
    Feature('feat_45', int),
    Feature('feat_46', int),
    Feature('feat_47', int),
    Feature('feat_48', int),
    Feature('feat_49', int),
    Feature('feat_50', int),
    Feature('feat_51', int),
    Feature('feat_52', int),
    Feature('feat_53', int),
    Feature('feat_54', int),
    Feature('feat_55', int),
    Feature('feat_56', int),
    Feature('feat_57', int),
    Feature('feat_58', int),
    Feature('feat_59', int),
    Feature('feat_60', int),
    Feature('feat_61', int),
    Feature('feat_62', int),
    Feature('feat_63', int),
    Feature('feat_64', int),
    Feature('feat_65', int),
    Feature('feat_66', int),
    Feature('feat_67', int),
    Feature('feat_68', int),
    Feature('feat_69', int),
    Feature('feat_70', int),
    Feature('feat_71', int),
    Feature('feat_72', int),
    Feature('feat_73', int),
    Feature('feat_74', int),
    Feature('feat_75', int),
    Feature('feat_76', int),
    Feature('feat_77', int),
    Feature('feat_78', int),
    Feature('feat_79', int),
    Feature('feat_80', int),
    Feature('feat_81', int),
    Feature('feat_82', int),
    Feature('feat_83', int),
    Feature('feat_84', int),
    Feature('feat_85', int),
    Feature('feat_86', int),
    Feature('feat_87', int),
    Feature('feat_88', int),
    Feature('feat_89', int),
    Feature('feat_90', int),
    Feature('feat_91', int),
    Feature('feat_92', int),
    Feature('feat_93', int),
    Feature('target', cat_dtype, is_target=True,
            value_mapping={
                'Class_1': '1',
                'Class_2': '2',
                'Class_3': '3',
                'Class_4': '4',
                'Class_5': '5',
                'Class_6': '6',
                'Class_7': '7',
                'Class_8': '8',
                'Class_9': '9',
            }),
],
    documentation="https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data")

SF_CRIME_FEATURES = FeatureList(features=[
    Feature('Dates', cat_dtype,
            name_extended="date and time of crime incident"),
    Feature('Category', cat_dtype, is_target=True),
    Feature('Descript', cat_dtype,
            name_extended="description of the crime incident"),
    Feature('DayOfWeek', cat_dtype, name_extended="day of week"),
    Feature('PdDistrict', cat_dtype,
            name_extended="Police Department District"),
    Feature('Resolution', cat_dtype),
    Feature('Address', cat_dtype,
            name_extended="approximate street address of the crime incident"),
    Feature('X', float, name_extended="longitude"),
    Feature('Y', float, name_extended="latitude"),
], documentation='https://www.kaggle.com/competitions/sf-crime/data')

PLASTICC_FEATURES = FeatureList(features=[
    Feature('object_id', int, name_extended="unique object identifier"),
    Feature('mjd', float,
            "the time in Modified Julian Date (MJD) of the observation. Can "
            "be read as days since November 17, 1858. Can be converted to "
            "Unix epoch time with the formula unix_time = (MJD−40587)×86400",
            name_extended="observation time in Modified Julian Date (MJD)"),
    Feature('passband', int,
            "The specific LSST passband integer, such that u, g, r, i, z, "
            "Y = 0, 1, 2, 3, 4, 5 in which it was viewed",
            name_extended="observation LSST passband"),
    Feature('flux', float,
            "the measured flux (brightness) in the passband of observation as "
            "listed in the passband column. These values have already been "
            "corrected for dust extinction (mwebv), though heavily extincted "
            "objects will have larger uncertainties (flux_err) in spite of "
            "the correction.",
            name_extended="flux (brightness)"),
    Feature('flux_err', float,
            "the uncertainty on the measurement of the flux listed above.",
            name_extended="uncertainty of flux measurement"),
    Feature('detected', int,
            "If 1, the object's brightness is significantly different at the "
            "3-sigma level relative to the reference template. Only objects "
            "with at least 2 detections are included in the dataset."),
    Feature('ra', float,
            "right ascension, sky coordinate: co-longitude in degrees",
            name_extended="right ascension, sky coordinate: co-longitude in degrees"),
    Feature('decl', float,
            "declination, sky coordinate: co-latitude in degrees",
            name_extended="declination, sky coordinate: co-latitude in degrees"),
    Feature('gal_l', float,
            name_extended="galactic longitude in degrees"),
    Feature('gal_b', float,
            name_extended="galactic latitude in degrees"),
    Feature('ddf', int,
            "A flag to identify the object as coming from the DDF survey area "
            "(with value DDF = 1 for the DDF, DDF = 0 for the WFD survey). "
            "Note that while the DDF fields are contained within the full WFD "
            "survey area, the DDF fluxes have significantly smaller "
            "uncertainties",
            name_extended="indocator for object is from DDF survey area"),
    Feature('hostgal_specz', float,
            "the spectroscopic redshift of the source. This is an extremely "
            "accurate measure of redshift, available for the training set and "
            "a small fraction of the test set.",
            name_extended="spectroscopic redshift"),
    Feature('hostgal_photoz', float,
            "The photometric redshift of the host galaxy of the astronomical "
            "source. While this is meant to be a proxy for hostgal_specz, "
            "there can be large differences between the two and should be "
            "regarded as a far less accurate version of hostgal_specz.",
            name_extended="photometric redshift of the host galaxy of the astronomical source"),
    Feature('hostgal_photoz_err', float,
            "The uncertainty on the hostgal_photoz based on LSST survey projections",
            name_extended="uncertainty of the photometric redshift"),
    Feature('distmod', float,
            "The distance to the source calculated from hostgal_photoz and using general relativity",
            name_extended="distance to the source calculated from photometric "
                          "redshift and using general relativity"),
    Feature('mwebv', float,
            "this ‘extinction’ of light is a property of the Milky Way (MW) "
            "dust along the line of sight to the astronomical source, and is "
            "thus a function of the sky coordinates of the source ra, "
            "decl. This is used to determine a passband dependent dimming and "
            "redenning of light from astronomical sources as described in "
            "subsection 2.1, and based on the Schlafly et al. (2011) and "
            "Schlegel et al. (1998) dust models."),
    Feature('target', int, is_target=True,
            name_extended="class of the astronomical source"),
], documentation="https://www.kaggle.com/competitions/PLAsTiCC-2018/data")

WALMART_FEATURES = FeatureList(features=[
    Feature('TripType', int, is_target=True,
            name_extended='id representing the type of shopping trip the customer made'),
    Feature('VisitNumber', int,
            'an id corresponding to a single trip by a single customer',
            name_extended='unique visit ID'),
    Feature('Weekday', cat_dtype, name_extended="weekday of the trip"),
    Feature('Upc', float, name_extended="UPC number of the product purchased"),
    Feature('ScanCount', int,
            "the number of the given item that was purchased. A negative value indicates a product return.",
            name_extended="number of the given item purchased"),
    Feature('DepartmentDescription', object,
            name_extended="product department description"),
    Feature('FinelineNumber', float, name_extended="product fine line number"),
],
    documentation="https://www.kaggle.com/competitions/walmart-recruiting-trip-type-classification/data")

TRADESHIFT_FEATURES = FeatureList(features=[
    Feature('y33', int, is_target=True),
    Feature('x140', cat_dtype),  # importance: 0.2476
    Feature('x131', float),  # importance: 0.1324
    Feature('x24', cat_dtype),  # importance: 0.1102
    Feature('x126', cat_dtype),  # importance: 0.091
    Feature('x115', cat_dtype),  # importance: 0.0714
    Feature('x9', float),  # importance: 0.042
    Feature('x54', float),  # importance: 0.0418
    Feature('x114', float),  # importance: 0.0321
    Feature('x61', cat_dtype),  # importance: 0.0242
    Feature('x31', cat_dtype),  # importance: 0.022
    Feature('x91', cat_dtype),  # importance: 0.0159
    Feature('x109', float),  # importance: 0.0114
    Feature('x4', cat_dtype),  # importance: 0.0101
    Feature('x141', cat_dtype),  # importance: 0.0101
    Feature('x136', float),  # importance: 0.01
    Feature('x132', float),  # importance: 0.0083
    Feature('x130', cat_dtype),  # importance: 0.0082
    Feature('x95', cat_dtype),  # importance: 0.0081
    Feature('x94', cat_dtype),  # importance: 0.0078
    Feature('x3', cat_dtype),  # importance: 0.0077
    Feature('x118', float),  # importance: 0.0072
    Feature('x143', float),  # importance: 0.0062
    Feature('x27', float),  # importance: 0.0059
    Feature('x120', float),  # importance: 0.0053
    Feature('x18', float),  # importance: 0.005
    Feature('x133', float),  # importance: 0.005
    Feature('x23', float),  # importance: 0.0047
    Feature('x34', cat_dtype),  # importance: 0.0046
    Feature('x84', float),  # importance: 0.0046
    Feature('x64', cat_dtype),  # importance: 0.0037
    Feature('x100', float),  # importance: 0.003
    Feature('x135', float),  # importance: 0.0029
    Feature('x106', float),  # importance: 0.0022
    ##################################################
    ##################################################
    # Feature('x145', float),  # importance: 0.0022
    # Feature('x134', float),  # importance: 0.0017
    # Feature('x65', cat_dtype),  # importance: 0.0016
    # Feature('x89', float),  # importance: 0.0015
    # Feature('x50', float),  # importance: 0.0015
    # Feature('x56', cat_dtype),  # importance: 0.0013
    # Feature('x123', float),  # importance: 0.0012
    # Feature('x40', float),  # importance: 0.0011
    # Feature('x30', cat_dtype),  # importance: 0.0011
    # Feature('x17', float),  # importance: 0.001
    # Feature('x35', cat_dtype),  # importance: 0.0009
    # Feature('x112', float),  # importance: 0.0008
    # Feature('x60', float),  # importance: 0.0005
    # Feature('x116', cat_dtype),  # importance: 0.0004
    # Feature('x21', float),  # importance: 0.0004
    # Feature('x83', float),  # importance: 0.0004
    # Feature('x137', float),  # importance: 0.0004
    # Feature('x28', float),  # importance: 0.0004
    # Feature('x5', float),  # importance: 0.0004
    # Feature('x90', float),  # importance: 0.0004
    # Feature('x38', float),  # importance: 0.0003
    # Feature('x22', float),  # importance: 0.0003
    # Feature('x113', float),  # importance: 0.0002
    # Feature('x6', float),  # importance: 0.0002
    # Feature('x16', float),  # importance: 0.0002
    # Feature('x104', cat_dtype),  # importance: 0.0002
    # Feature('x79', float),  # importance: 0.0002
    # Feature('x117', cat_dtype),  # importance: 0.0002
    # Feature('x70', float),  # importance: 0.0002
    # Feature('x82', float),  # importance: 0.0002
    # Feature('x144', float),  # importance: 0.0002
    # Feature('x2', cat_dtype),  # importance: 0.0001
    # Feature('x47', float),  # importance: 0.0001
    # Feature('x53', float),  # importance: 0.0001
    # Feature('x129', cat_dtype),  # importance: 0.0001
    # Feature('x62', cat_dtype),  # importance: 0.0001
    # Feature('x119', float),  # importance: 0.0001
    # Feature('x101', cat_dtype),  # importance: 0.0001
    # Feature('x49', float),  # importance: 0.0001
    # Feature('x110', float),  # importance: 0.0001
    # Feature('x105', cat_dtype),  # importance: 0.0001
    # Feature('x68', float),  # importance: 0.0001
    # Feature('x46', float),  # importance: 0.0001
    # Feature('x14', cat_dtype),  # importance: 0.0001
    # Feature('x15', float),  # importance: 0.0001
    # Feature('x37', float),  # importance: 0.0001
    # Feature('x121', float),  # importance: 0.0001
    # Feature('x29', float),  # importance: 0.0001
    # Feature('x125', float),  # importance: 0.0001
    # Feature('x66', float),  # importance: 0.0001
    # Feature('x92', cat_dtype),  # importance: 0.0001
    # Feature('x128', cat_dtype),  # importance: 0.0001
    # Feature('x20', float),  # importance: 0.0001
    # Feature('x76', float),  # importance: 0.0001
    # Feature('x48', float),  # importance: 0.0001
    # Feature('x63', cat_dtype),  # importance: 0.0001
    # Feature('x77', float),  # importance: 0.0001
    # Feature('x1', cat_dtype),  # importance: 0.0001
    # Feature('x98', float),  # importance: 0.0001
    # Feature('x8', float),  # importance: 0.0001
    # Feature('x43', cat_dtype),  # importance: 0.0001
    # Feature('x139', float),  # importance: 0.0001
    # Feature('x96', float),  # importance: 0.0001
    # Feature('x51', float),  # importance: 0.0001
    # Feature('x86', cat_dtype),  # importance: 0.0001
    # Feature('x107', float),  # importance: 0.0001
    # Feature('x59', float),  # importance: 0.0001
    # Feature('x80', float),  # importance: 0.0001
    # Feature('x36', float),  # importance: 0.0001
    # Feature('x13', cat_dtype),  # importance: 0.0001
    # Feature('x32', cat_dtype),  # importance: 0.0001
    # Feature('x138', float),  # importance: 0.0001
    # Feature('x67', float),  # importance: 0.0001
    # Feature('x19', float),  # importance: 0.0001
    # Feature('x69', float),  # importance: 0.0001
    # Feature('x7', float),  # importance: 0.0001
    # Feature('x10', cat_dtype),  # importance: 0.0001
    # Feature('x39', float),  # importance: 0.0001
    # Feature('x78', float),  # importance: 0.0001
    # Feature('x88', float),  # importance: 0.0001
    # Feature('x81', float),  # importance: 0.0001
    # Feature('x55', cat_dtype),  # importance: 0.0001
    # Feature('x103', cat_dtype),  # importance: 0.0001
    # Feature('x97', float),  # importance: 0.0001
    # Feature('x11', cat_dtype),  # importance: 0.0
    # Feature('x58', float),  # importance: 0.0
    # Feature('x142', cat_dtype),  # importance: 0.0
    # Feature('x108', float),  # importance: 0.0
    # Feature('x93', cat_dtype),  # importance: 0.0
    # Feature('x99', float),  # importance: 0.0
    # Feature('x52', float),  # importance: 0.0
    # Feature('id', float),  # importance: 0.0
    # Feature('x124', float),  # importance: 0.0
    # Feature('x41', cat_dtype),  # importance: 0.0
    # Feature('x111', float),  # importance: 0.0
    # Feature('x75', cat_dtype),  # importance: 0.0
    # Feature('x87', cat_dtype),  # importance: 0.0
    # Feature('x73', cat_dtype),  # importance: 0.0
    # Feature('x45', cat_dtype),  # importance: 0.0
    # Feature('x25', cat_dtype),  # importance: 0.0
    # Feature('x72', cat_dtype),  # importance: 0.0
    # Feature('x71', cat_dtype),  # importance: 0.0
    # Feature('x26', cat_dtype),  # importance: 0.0
    # Feature('x57', cat_dtype),  # importance: 0.0
    # Feature('x127', cat_dtype),  # importance: 0.0
    # Feature('x122', float),  # importance: 0.0
    # Feature('x12', cat_dtype),  # importance: 0.0
    # Feature('x85', cat_dtype),  # importance: 0.0
    # Feature('x102', cat_dtype),  # importance: 0.0
    # Feature('x33', cat_dtype),  # importance: 0.0
    # Feature('x74', cat_dtype),  # importance: 0.0
    # Feature('x44', cat_dtype),  # importance: 0.0
    # Feature('x42', cat_dtype),  # importance: 0.0
],
    documentation="https://www.kaggle.com/competitions/tradeshift-text-classification/data")

SCHIZOPHRENIA_FEATURES = FeatureList(features=[
    Feature('Class', int, is_target=True,
                value_mapping={0: 'Healthy Control',
                               1: 'Schizophrenic Patient'}
            ),
    Feature('FNC233', float, 'functional MRI feature 233'),  # importance: 0.1325
    Feature('FNC237', float, 'functional MRI feature 237'),  # importance: 0.0963
    Feature('FNC226', float, 'functional MRI feature 226'),  # importance: 0.0812
    Feature('FNC194', float, 'functional MRI feature 194'),  # importance: 0.0578
    Feature('SBM_map7', float, 'structural ICA map feature 7'),  # importance: 0.0549
    Feature('FNC33', float, 'functional MRI feature 33'),  # importance: 0.0538
    Feature('SBM_map36', float, 'structural ICA map feature 36'),  # importance: 0.0468
    Feature('FNC105', float, 'functional MRI feature 105'),  # importance: 0.0465
    Feature('FNC161', float, 'functional MRI feature 161'),  # importance: 0.038
    Feature('FNC353', float, 'functional MRI feature 353'),  # importance: 0.0371
    Feature('FNC290', float, 'functional MRI feature 290'),  # importance: 0.0353
    Feature('FNC80', float, 'functional MRI feature 80'),  # importance: 0.0299
    Feature('FNC243', float, 'functional MRI feature 243'),  # importance: 0.0277
    Feature('FNC345', float, 'functional MRI feature 345'),  # importance: 0.025
    Feature('FNC293', float, 'functional MRI feature 293'),  # importance: 0.022
    Feature('FNC158', float, 'functional MRI feature 158'),  # importance: 0.0201
    Feature('FNC48', float, 'functional MRI feature 48'),  # importance: 0.0196
    Feature('FNC110', float, 'functional MRI feature 110'),  # importance: 0.0175
    Feature('FNC270', float, 'functional MRI feature 270'),  # importance: 0.017
    Feature('FNC295', float, 'functional MRI feature 295'),  # importance: 0.0158
    Feature('FNC301', float, 'functional MRI feature 301'),  # importance: 0.0136
    Feature('SBM_map67', float, 'structural ICA map feature 67'),  # importance: 0.0134
    Feature('FNC75', float, 'functional MRI feature 75'),  # importance: 0.0113
    Feature('FNC337', float, 'functional MRI feature 337'),  # importance: 0.0079
    Feature('FNC67', float, 'functional MRI feature 67'),  # importance: 0.0076
    Feature('SBM_map17', float, 'structural ICA map feature 17'),  # importance: 0.0074
    Feature('FNC165', float, 'functional MRI feature 165'),  # importance: 0.0068
    Feature('FNC61', float, 'functional MRI feature 61'),  # importance: 0.0068
    Feature('FNC278', float, 'functional MRI feature 278'),  # importance: 0.0067
    Feature('FNC208', float, 'functional MRI feature 208'),  # importance: 0.0066
    Feature('FNC136', float, 'functional MRI feature 136'),  # importance: 0.0065
    Feature('FNC5', float, 'functional MRI feature 5'),  # importance: 0.0049
    Feature('FNC189', float, 'functional MRI feature 189'),  # importance: 0.0047
    ##################################################
    ##################################################
    # Feature('FNC232', float, 'functional MRI feature 232'),  # importance: 0.0043
    # Feature('FNC376', float, 'functional MRI feature 376'),  # importance: 0.0041
    # Feature('SBM_map13', float, 'structural ICA map feature 13'),  # importance: 0.0035
    # Feature('FNC167', float, 'functional MRI feature 167'),  # importance: 0.0032
    # Feature('FNC263', float, 'functional MRI feature 263'),  # importance: 0.0025
    # Feature('FNC20', float, 'functional MRI feature 20'),  # importance: 0.0025
    # Feature('FNC357', float, 'functional MRI feature 357'),  # importance: 0.0007
    # Feature('FNC352', float, 'functional MRI feature 352'),  # importance: 0.0
    # Feature('FNC39', float, 'functional MRI feature 39'),  # importance: 0.0
    # Feature('FNC229', float, 'functional MRI feature 229'),  # importance: 0.0
    # Feature('FNC129', float, 'functional MRI feature 129'),  # importance: 0.0
    # Feature('SBM_map71', float, 'structural ICA map feature 71'),  # importance: 0.0
    # Feature('FNC296', float, 'functional MRI feature 296'),  # importance: 0.0
    # Feature('FNC209', float, 'functional MRI feature 209'),  # importance: 0.0
    # Feature('FNC305', float, 'functional MRI feature 305'),  # importance: 0.0
    # Feature('SBM_map26', float, 'structural ICA map feature 26'),  # importance: 0.0
    # Feature('FNC211', float, 'functional MRI feature 211'),  # importance: 0.0
    # Feature('FNC144', float, 'functional MRI feature 144'),  # importance: 0.0
    # Feature('FNC362', float, 'functional MRI feature 362'),  # importance: 0.0
    # Feature('FNC22', float, 'functional MRI feature 22'),  # importance: 0.0
    # Feature('FNC182', float, 'functional MRI feature 182'),  # importance: 0.0
    # Feature('FNC2', float, 'functional MRI feature 2'),  # importance: 0.0
    # Feature('FNC107', float, 'functional MRI feature 107'),  # importance: 0.0
    # Feature('FNC98', float, 'functional MRI feature 98'),  # importance: 0.0
    # Feature('FNC145', float, 'functional MRI feature 145'),  # importance: 0.0
    # Feature('FNC43', float, 'functional MRI feature 43'),  # importance: 0.0
    # Feature('FNC162', float, 'functional MRI feature 162'),  # importance: 0.0
    # Feature('FNC84', float, 'functional MRI feature 84'),  # importance: 0.0
    # Feature('FNC9', float, 'functional MRI feature 9'),  # importance: 0.0
    # Feature('FNC283', float, 'functional MRI feature 283'),  # importance: 0.0
    # Feature('FNC146', float, 'functional MRI feature 146'),  # importance: 0.0
    # Feature('FNC60', float, 'functional MRI feature 60'),  # importance: 0.0
    # Feature('FNC1', float, 'functional MRI feature 1'),  # importance: 0.0
    # Feature('FNC184', float, 'functional MRI feature 184'),  # importance: 0.0
    # Feature('FNC176', float, 'functional MRI feature 176'),  # importance: 0.0
    # Feature('SBM_map51', float, 'structural ICA map feature 51'),  # importance: 0.0
    # Feature('FNC222', float, 'functional MRI feature 222'),  # importance: 0.0
    # Feature('FNC56', float, 'functional MRI feature 56'),  # importance: 0.0
    # Feature('FNC254', float, 'functional MRI feature 254'),  # importance: 0.0
    # Feature('FNC282', float, 'functional MRI feature 282'),  # importance: 0.0
    # Feature('FNC78', float, 'functional MRI feature 78'),  # importance: 0.0
    # Feature('FNC50', float, 'functional MRI feature 50'),  # importance: 0.0
    # Feature('FNC6', float, 'functional MRI feature 6'),  # importance: 0.0
    # Feature('FNC186', float, 'functional MRI feature 186'),  # importance: 0.0
    # Feature('FNC331', float, 'functional MRI feature 331'),  # importance: 0.0
    # Feature('FNC168', float, 'functional MRI feature 168'),  # importance: 0.0
    # Feature('FNC45', float, 'functional MRI feature 45'),  # importance: 0.0
    # Feature('FNC69', float, 'functional MRI feature 69'),  # importance: 0.0
    # Feature('FNC73', float, 'functional MRI feature 73'),  # importance: 0.0
    # Feature('FNC117', float, 'functional MRI feature 117'),  # importance: 0.0
    # Feature('FNC173', float, 'functional MRI feature 173'),  # importance: 0.0
    # Feature('FNC109', float, 'functional MRI feature 109'),  # importance: 0.0
    # Feature('FNC319', float, 'functional MRI feature 319'),  # importance: 0.0
    # Feature('FNC7', float, 'functional MRI feature 7'),  # importance: 0.0
    # Feature('FNC215', float, 'functional MRI feature 215'),  # importance: 0.0
    # Feature('FNC302', float, 'functional MRI feature 302'),  # importance: 0.0
    # Feature('FNC36', float, 'functional MRI feature 36'),  # importance: 0.0
    # Feature('FNC174', float, 'functional MRI feature 174'),  # importance: 0.0
    # Feature('FNC90', float, 'functional MRI feature 90'),  # importance: 0.0
    # Feature('FNC225', float, 'functional MRI feature 225'),  # importance: 0.0
    # Feature('FNC212', float, 'functional MRI feature 212'),  # importance: 0.0
    # Feature('FNC77', float, 'functional MRI feature 77'),  # importance: 0.0
    # Feature('SBM_map45', float, 'structural ICA map feature 45'),  # importance: 0.0
    # Feature('FNC371', float, 'functional MRI feature 371'),  # importance: 0.0
    # Feature('FNC370', float, 'functional MRI feature 370'),  # importance: 0.0
    # Feature('FNC157', float, 'functional MRI feature 157'),  # importance: 0.0
    # Feature('SBM_map1', float, 'structural ICA map feature 1'),  # importance: 0.0
    # Feature('FNC3', float, 'functional MRI feature 3'),  # importance: 0.0
    # Feature('FNC164', float, 'functional MRI feature 164'),  # importance: 0.0
    # Feature('FNC128', float, 'functional MRI feature 128'),  # importance: 0.0
    # Feature('FNC231', float, 'functional MRI feature 231'),  # importance: 0.0
    # Feature('FNC286', float, 'functional MRI feature 286'),  # importance: 0.0
    # Feature('FNC138', float, 'functional MRI feature 138'),  # importance: 0.0
    # Feature('FNC193', float, 'functional MRI feature 193'),  # importance: 0.0
    # Feature('SBM_map8', float, 'structural ICA map feature 8'),  # importance: 0.0
    # Feature('FNC369', float, 'functional MRI feature 369'),  # importance: 0.0
    # Feature('FNC299', float, 'functional MRI feature 299'),  # importance: 0.0
    # Feature('SBM_map2', float, 'structural ICA map feature 2'),  # importance: 0.0
    # Feature('FNC130', float, 'functional MRI feature 130'),  # importance: 0.0
    # Feature('FNC126', float, 'functional MRI feature 126'),  # importance: 0.0
    # Feature('FNC197', float, 'functional MRI feature 197'),  # importance: 0.0
    # Feature('FNC111', float, 'functional MRI feature 111'),  # importance: 0.0
    # Feature('FNC312', float, 'functional MRI feature 312'),  # importance: 0.0
    # Feature('FNC142', float, 'functional MRI feature 142'),  # importance: 0.0
    # Feature('FNC300', float, 'functional MRI feature 300'),  # importance: 0.0
    # Feature('FNC85', float, 'functional MRI feature 85'),  # importance: 0.0
    # Feature('FNC104', float, 'functional MRI feature 104'),  # importance: 0.0
    # Feature('FNC242', float, 'functional MRI feature 242'),  # importance: 0.0
    # Feature('FNC114', float, 'functional MRI feature 114'),  # importance: 0.0
    # Feature('FNC342', float, 'functional MRI feature 342'),  # importance: 0.0
    # Feature('SBM_map55', float, 'structural ICA map feature 55'),  # importance: 0.0
    # Feature('FNC21', float, 'functional MRI feature 21'),  # importance: 0.0
    # Feature('FNC15', float, 'functional MRI feature 15'),  # importance: 0.0
    # Feature('FNC235', float, 'functional MRI feature 235'),  # importance: 0.0
    # Feature('FNC99', float, 'functional MRI feature 99'),  # importance: 0.0
    # Feature('FNC47', float, 'functional MRI feature 47'),  # importance: 0.0
    # Feature('FNC274', float, 'functional MRI feature 274'),  # importance: 0.0
    # Feature('FNC304', float, 'functional MRI feature 304'),  # importance: 0.0
    # Feature('FNC100', float, 'functional MRI feature 100'),  # importance: 0.0
    # Feature('FNC344', float, 'functional MRI feature 344'),  # importance: 0.0
    # Feature('FNC310', float, 'functional MRI feature 310'),  # importance: 0.0
    # Feature('FNC377', float, 'functional MRI feature 377'),  # importance: 0.0
    # Feature('FNC248', float, 'functional MRI feature 248'),  # importance: 0.0
    # Feature('FNC292', float, 'functional MRI feature 292'),  # importance: 0.0
    # Feature('FNC323', float, 'functional MRI feature 323'),  # importance: 0.0
    # Feature('FNC249', float, 'functional MRI feature 249'),  # importance: 0.0
    # Feature('FNC124', float, 'functional MRI feature 124'),  # importance: 0.0
    # Feature('FNC332', float, 'functional MRI feature 332'),  # importance: 0.0
    # Feature('FNC350', float, 'functional MRI feature 350'),  # importance: 0.0
    # Feature('FNC64', float, 'functional MRI feature 64'),  # importance: 0.0
    # Feature('FNC375', float, 'functional MRI feature 375'),  # importance: 0.0
    # Feature('Id', float),  # importance: 0.0
    # Feature('FNC76', float, 'functional MRI feature 76'),  # importance: 0.0
    # Feature('FNC70', float, 'functional MRI feature 70'),  # importance: 0.0
    # Feature('FNC351', float, 'functional MRI feature 351'),  # importance: 0.0
    # Feature('FNC218', float, 'functional MRI feature 218'),  # importance: 0.0
    # Feature('FNC199', float, 'functional MRI feature 199'),  # importance: 0.0
    # Feature('FNC55', float, 'functional MRI feature 55'),  # importance: 0.0
    # Feature('FNC59', float, 'functional MRI feature 59'),  # importance: 0.0
    # Feature('FNC267', float, 'functional MRI feature 267'),  # importance: 0.0
    # Feature('FNC178', float, 'functional MRI feature 178'),  # importance: 0.0
    # Feature('FNC143', float, 'functional MRI feature 143'),  # importance: 0.0
    # Feature('FNC269', float, 'functional MRI feature 269'),  # importance: 0.0
    # Feature('FNC32', float, 'functional MRI feature 32'),  # importance: 0.0
    # Feature('FNC281', float, 'functional MRI feature 281'),  # importance: 0.0
    # Feature('FNC373', float, 'functional MRI feature 373'),  # importance: 0.0
    # Feature('FNC303', float, 'functional MRI feature 303'),  # importance: 0.0
    # Feature('FNC236', float, 'functional MRI feature 236'),  # importance: 0.0
    # Feature('FNC185', float, 'functional MRI feature 185'),  # importance: 0.0
    # Feature('SBM_map32', float, 'structural ICA map feature 32'),  # importance: 0.0
    # Feature('FNC285', float, 'functional MRI feature 285'),  # importance: 0.0
    # Feature('FNC10', float, 'functional MRI feature 10'),  # importance: 0.0
    # Feature('SBM_map52', float, 'structural ICA map feature 52'),  # importance: 0.0
    # Feature('FNC359', float, 'functional MRI feature 359'),  # importance: 0.0
    # Feature('FNC53', float, 'functional MRI feature 53'),  # importance: 0.0
    # Feature('FNC150', float, 'functional MRI feature 150'),  # importance: 0.0
    # Feature('FNC205', float, 'functional MRI feature 205'),  # importance: 0.0
    # Feature('FNC82', float, 'functional MRI feature 82'),  # importance: 0.0
    # Feature('FNC72', float, 'functional MRI feature 72'),  # importance: 0.0
    # Feature('FNC37', float, 'functional MRI feature 37'),  # importance: 0.0
    # Feature('FNC35', float, 'functional MRI feature 35'),  # importance: 0.0
    # Feature('FNC246', float, 'functional MRI feature 246'),  # importance: 0.0
    # Feature('FNC361', float, 'functional MRI feature 361'),  # importance: 0.0
    # Feature('FNC333', float, 'functional MRI feature 333'),  # importance: 0.0
    # Feature('FNC374', float, 'functional MRI feature 374'),  # importance: 0.0
    # Feature('FNC207', float, 'functional MRI feature 207'),  # importance: 0.0
    # Feature('FNC135', float, 'functional MRI feature 135'),  # importance: 0.0
    # Feature('FNC95', float, 'functional MRI feature 95'),  # importance: 0.0
    # Feature('FNC160', float, 'functional MRI feature 160'),  # importance: 0.0
    # Feature('FNC42', float, 'functional MRI feature 42'),  # importance: 0.0
    # Feature('FNC31', float, 'functional MRI feature 31'),  # importance: 0.0
    # Feature('FNC201', float, 'functional MRI feature 201'),  # importance: 0.0
    # Feature('FNC358', float, 'functional MRI feature 358'),  # importance: 0.0
    # Feature('FNC206', float, 'functional MRI feature 206'),  # importance: 0.0
    # Feature('FNC24', float, 'functional MRI feature 24'),  # importance: 0.0
    # Feature('FNC87', float, 'functional MRI feature 87'),  # importance: 0.0
    # Feature('FNC336', float, 'functional MRI feature 336'),  # importance: 0.0
    # Feature('FNC13', float, 'functional MRI feature 13'),  # importance: 0.0
    # Feature('FNC297', float, 'functional MRI feature 297'),  # importance: 0.0
    # Feature('FNC288', float, 'functional MRI feature 288'),  # importance: 0.0
    # Feature('FNC147', float, 'functional MRI feature 147'),  # importance: 0.0
    # Feature('FNC329', float, 'functional MRI feature 329'),  # importance: 0.0
    # Feature('FNC202', float, 'functional MRI feature 202'),  # importance: 0.0
    # Feature('FNC325', float, 'functional MRI feature 325'),  # importance: 0.0
    # Feature('FNC343', float, 'functional MRI feature 343'),  # importance: 0.0
    # Feature('FNC321', float, 'functional MRI feature 321'),  # importance: 0.0
    # Feature('FNC378', float, 'functional MRI feature 378'),  # importance: 0.0
    # Feature('FNC28', float, 'functional MRI feature 28'),  # importance: 0.0
    # Feature('FNC311', float, 'functional MRI feature 311'),  # importance: 0.0
    # Feature('FNC308', float, 'functional MRI feature 308'),  # importance: 0.0
    # Feature('FNC102', float, 'functional MRI feature 102'),  # importance: 0.0
    # Feature('FNC307', float, 'functional MRI feature 307'),  # importance: 0.0
    # Feature('FNC250', float, 'functional MRI feature 250'),  # importance: 0.0
    # Feature('FNC188', float, 'functional MRI feature 188'),  # importance: 0.0
    # Feature('SBM_map6', float, 'structural ICA map feature 6'),  # importance: 0.0
    # Feature('SBM_map61', float, 'structural ICA map feature 61'),  # importance: 0.0
    # Feature('FNC91', float, 'functional MRI feature 91'),  # importance: 0.0
    # Feature('FNC347', float, 'functional MRI feature 347'),  # importance: 0.0
    # Feature('FNC262', float, 'functional MRI feature 262'),  # importance: 0.0
    # Feature('FNC169', float, 'functional MRI feature 169'),  # importance: 0.0
    # Feature('FNC118', float, 'functional MRI feature 118'),  # importance: 0.0
    # Feature('FNC190', float, 'functional MRI feature 190'),  # importance: 0.0
    # Feature('FNC68', float, 'functional MRI feature 68'),  # importance: 0.0
    # Feature('FNC314', float, 'functional MRI feature 314'),  # importance: 0.0
    # Feature('FNC348', float, 'functional MRI feature 348'),  # importance: 0.0
    # Feature('FNC210', float, 'functional MRI feature 210'),  # importance: 0.0
    # Feature('FNC217', float, 'functional MRI feature 217'),  # importance: 0.0
    # Feature('FNC4', float, 'functional MRI feature 4'),  # importance: 0.0
    # Feature('FNC94', float, 'functional MRI feature 94'),  # importance: 0.0
    # Feature('FNC132', float, 'functional MRI feature 132'),  # importance: 0.0
    # Feature('FNC172', float, 'functional MRI feature 172'),  # importance: 0.0
    # Feature('FNC83', float, 'functional MRI feature 83'),  # importance: 0.0
    # Feature('FNC356', float, 'functional MRI feature 356'),  # importance: 0.0
    # Feature('FNC317', float, 'functional MRI feature 317'),  # importance: 0.0
    # Feature('FNC239', float, 'functional MRI feature 239'),  # importance: 0.0
    # Feature('FNC272', float, 'functional MRI feature 272'),  # importance: 0.0
    # Feature('SBM_map43', float, 'structural ICA map feature 43'),  # importance: 0.0
    # Feature('FNC341', float, 'functional MRI feature 341'),  # importance: 0.0
    # Feature('FNC192', float, 'functional MRI feature 192'),  # importance: 0.0
    # Feature('FNC113', float, 'functional MRI feature 113'),  # importance: 0.0
    # Feature('FNC58', float, 'functional MRI feature 58'),  # importance: 0.0
    # Feature('FNC221', float, 'functional MRI feature 221'),  # importance: 0.0
    # Feature('FNC71', float, 'functional MRI feature 71'),  # importance: 0.0
    # Feature('FNC334', float, 'functional MRI feature 334'),  # importance: 0.0
    # Feature('FNC79', float, 'functional MRI feature 79'),  # importance: 0.0
    # Feature('FNC251', float, 'functional MRI feature 251'),  # importance: 0.0
    # Feature('FNC141', float, 'functional MRI feature 141'),  # importance: 0.0
    # Feature('FNC200', float, 'functional MRI feature 200'),  # importance: 0.0
    # Feature('FNC177', float, 'functional MRI feature 177'),  # importance: 0.0
    # Feature('FNC322', float, 'functional MRI feature 322'),  # importance: 0.0
    # Feature('FNC220', float, 'functional MRI feature 220'),  # importance: 0.0
    # Feature('FNC93', float, 'functional MRI feature 93'),  # importance: 0.0
    # Feature('FNC8', float, 'functional MRI feature 8'),  # importance: 0.0
    # Feature('FNC170', float, 'functional MRI feature 170'),  # importance: 0.0
    # Feature('FNC134', float, 'functional MRI feature 134'),  # importance: 0.0
    # Feature('FNC44', float, 'functional MRI feature 44'),  # importance: 0.0
    # Feature('FNC234', float, 'functional MRI feature 234'),  # importance: 0.0
    # Feature('FNC277', float, 'functional MRI feature 277'),  # importance: 0.0
    # Feature('FNC223', float, 'functional MRI feature 223'),  # importance: 0.0
    # Feature('SBM_map28', float, 'structural ICA map feature 28'),  # importance: 0.0
    # Feature('FNC25', float, 'functional MRI feature 25'),  # importance: 0.0
    # Feature('FNC152', float, 'functional MRI feature 152'),  # importance: 0.0
    # Feature('FNC196', float, 'functional MRI feature 196'),  # importance: 0.0
    # Feature('FNC115', float, 'functional MRI feature 115'),  # importance: 0.0
    # Feature('FNC155', float, 'functional MRI feature 155'),  # importance: 0.0
    # Feature('FNC309', float, 'functional MRI feature 309'),  # importance: 0.0
    # Feature('FNC338', float, 'functional MRI feature 338'),  # importance: 0.0
    # Feature('FNC180', float, 'functional MRI feature 180'),  # importance: 0.0
    # Feature('FNC367', float, 'functional MRI feature 367'),  # importance: 0.0
    # Feature('FNC255', float, 'functional MRI feature 255'),  # importance: 0.0
    # Feature('FNC137', float, 'functional MRI feature 137'),  # importance: 0.0
    # Feature('FNC62', float, 'functional MRI feature 62'),  # importance: 0.0
    # Feature('FNC51', float, 'functional MRI feature 51'),  # importance: 0.0
    # Feature('FNC14', float, 'functional MRI feature 14'),  # importance: 0.0
    # Feature('FNC54', float, 'functional MRI feature 54'),  # importance: 0.0
    # Feature('FNC224', float, 'functional MRI feature 224'),  # importance: 0.0
    # Feature('FNC198', float, 'functional MRI feature 198'),  # importance: 0.0
    # Feature('FNC363', float, 'functional MRI feature 363'),  # importance: 0.0
    # Feature('FNC203', float, 'functional MRI feature 203'),  # importance: 0.0
    # Feature('FNC227', float, 'functional MRI feature 227'),  # importance: 0.0
    # Feature('SBM_map64', float, 'structural ICA map feature 64'),  # importance: 0.0
    # Feature('FNC291', float, 'functional MRI feature 291'),  # importance: 0.0
    # Feature('FNC258', float, 'functional MRI feature 258'),  # importance: 0.0
    # Feature('FNC256', float, 'functional MRI feature 256'),  # importance: 0.0
    # Feature('FNC213', float, 'functional MRI feature 213'),  # importance: 0.0
    # Feature('FNC204', float, 'functional MRI feature 204'),  # importance: 0.0
    # Feature('FNC57', float, 'functional MRI feature 57'),  # importance: 0.0
    # Feature('FNC16', float, 'functional MRI feature 16'),  # importance: 0.0
    # Feature('FNC238', float, 'functional MRI feature 238'),  # importance: 0.0
    # Feature('FNC74', float, 'functional MRI feature 74'),  # importance: 0.0
    # Feature('FNC191', float, 'functional MRI feature 191'),  # importance: 0.0
    # Feature('FNC276', float, 'functional MRI feature 276'),  # importance: 0.0
    # Feature('FNC346', float, 'functional MRI feature 346'),  # importance: 0.0
    # Feature('FNC52', float, 'functional MRI feature 52'),  # importance: 0.0
    # Feature('FNC97', float, 'functional MRI feature 97'),  # importance: 0.0
    # Feature('FNC372', float, 'functional MRI feature 372'),  # importance: 0.0
    # Feature('FNC316', float, 'functional MRI feature 316'),  # importance: 0.0
    # Feature('FNC253', float, 'functional MRI feature 253'),  # importance: 0.0
    # Feature('FNC26', float, 'functional MRI feature 26'),  # importance: 0.0
    # Feature('FNC101', float, 'functional MRI feature 101'),  # importance: 0.0
    # Feature('FNC89', float, 'functional MRI feature 89'),  # importance: 0.0
    # Feature('SBM_map74', float, 'structural ICA map feature 74'),  # importance: 0.0
    # Feature('FNC368', float, 'functional MRI feature 368'),  # importance: 0.0
    # Feature('FNC179', float, 'functional MRI feature 179'),  # importance: 0.0
    # Feature('FNC121', float, 'functional MRI feature 121'),  # importance: 0.0
    # Feature('FNC49', float, 'functional MRI feature 49'),  # importance: 0.0
    # Feature('FNC355', float, 'functional MRI feature 355'),  # importance: 0.0
    # Feature('FNC12', float, 'functional MRI feature 12'),  # importance: 0.0
    # Feature('FNC139', float, 'functional MRI feature 139'),  # importance: 0.0
    # Feature('FNC116', float, 'functional MRI feature 116'),  # importance: 0.0
    # Feature('FNC275', float, 'functional MRI feature 275'),  # importance: 0.0
    # Feature('FNC112', float, 'functional MRI feature 112'),  # importance: 0.0
    # Feature('FNC81', float, 'functional MRI feature 81'),  # importance: 0.0
    # Feature('FNC264', float, 'functional MRI feature 264'),  # importance: 0.0
    # Feature('FNC364', float, 'functional MRI feature 364'),  # importance: 0.0
    # Feature('FNC271', float, 'functional MRI feature 271'),  # importance: 0.0
    # Feature('SBM_map40', float, 'structural ICA map feature 40'),  # importance: 0.0
    # Feature('FNC38', float, 'functional MRI feature 38'),  # importance: 0.0
    # Feature('FNC153', float, 'functional MRI feature 153'),  # importance: 0.0
    # Feature('SBM_map73', float, 'structural ICA map feature 73'),  # importance: 0.0
    # Feature('FNC294', float, 'functional MRI feature 294'),  # importance: 0.0
    # Feature('FNC328', float, 'functional MRI feature 328'),  # importance: 0.0
    # Feature('FNC46', float, 'functional MRI feature 46'),  # importance: 0.0
    # Feature('SBM_map22', float, 'structural ICA map feature 22'),  # importance: 0.0
    # Feature('FNC339', float, 'functional MRI feature 339'),  # importance: 0.0
    # Feature('FNC103', float, 'functional MRI feature 103'),  # importance: 0.0
    # Feature('SBM_map4', float, 'structural ICA map feature 4'),  # importance: 0.0
    # Feature('FNC287', float, 'functional MRI feature 287'),  # importance: 0.0
    # Feature('FNC127', float, 'functional MRI feature 127'),  # importance: 0.0
    # Feature('FNC241', float, 'functional MRI feature 241'),  # importance: 0.0
    # Feature('FNC140', float, 'functional MRI feature 140'),  # importance: 0.0
    # Feature('FNC280', float, 'functional MRI feature 280'),  # importance: 0.0
    # Feature('FNC260', float, 'functional MRI feature 260'),  # importance: 0.0
    # Feature('FNC88', float, 'functional MRI feature 88'),  # importance: 0.0
    # Feature('FNC214', float, 'functional MRI feature 214'),  # importance: 0.0
    # Feature('FNC30', float, 'functional MRI feature 30'),  # importance: 0.0
    # Feature('FNC86', float, 'functional MRI feature 86'),  # importance: 0.0
    # Feature('FNC63', float, 'functional MRI feature 63'),  # importance: 0.0
    # Feature('FNC175', float, 'functional MRI feature 175'),  # importance: 0.0
    # Feature('FNC340', float, 'functional MRI feature 340'),  # importance: 0.0
    # Feature('FNC268', float, 'functional MRI feature 268'),  # importance: 0.0
    # Feature('FNC151', float, 'functional MRI feature 151'),  # importance: 0.0
    # Feature('SBM_map48', float, 'structural ICA map feature 48'),  # importance: 0.0
    # Feature('SBM_map3', float, 'structural ICA map feature 3'),  # importance: 0.0
    # Feature('FNC360', float, 'functional MRI feature 360'),  # importance: 0.0
    # Feature('FNC259', float, 'functional MRI feature 259'),  # importance: 0.0
    # Feature('FNC34', float, 'functional MRI feature 34'),  # importance: 0.0
    # Feature('FNC289', float, 'functional MRI feature 289'),  # importance: 0.0
    # Feature('FNC326', float, 'functional MRI feature 326'),  # importance: 0.0
    # Feature('FNC273', float, 'functional MRI feature 273'),  # importance: 0.0
    # Feature('FNC41', float, 'functional MRI feature 41'),  # importance: 0.0
    # Feature('FNC266', float, 'functional MRI feature 266'),  # importance: 0.0
    # Feature('FNC298', float, 'functional MRI feature 298'),  # importance: 0.0
    # Feature('FNC257', float, 'functional MRI feature 257'),  # importance: 0.0
    # Feature('FNC240', float, 'functional MRI feature 240'),  # importance: 0.0
    # Feature('FNC131', float, 'functional MRI feature 131'),  # importance: 0.0
    # Feature('FNC245', float, 'functional MRI feature 245'),  # importance: 0.0
    # Feature('FNC366', float, 'functional MRI feature 366'),  # importance: 0.0
    # Feature('FNC284', float, 'functional MRI feature 284'),  # importance: 0.0
    # Feature('FNC159', float, 'functional MRI feature 159'),  # importance: 0.0
    # Feature('FNC120', float, 'functional MRI feature 120'),  # importance: 0.0
    # Feature('FNC11', float, 'functional MRI feature 11'),  # importance: 0.0
    # Feature('FNC216', float, 'functional MRI feature 216'),  # importance: 0.0
    # Feature('FNC156', float, 'functional MRI feature 156'),  # importance: 0.0
    # Feature('FNC320', float, 'functional MRI feature 320'),  # importance: 0.0
    # Feature('FNC96', float, 'functional MRI feature 96'),  # importance: 0.0
    # Feature('FNC119', float, 'functional MRI feature 119'),  # importance: 0.0
    # Feature('FNC40', float, 'functional MRI feature 40'),  # importance: 0.0
    # Feature('FNC324', float, 'functional MRI feature 324'),  # importance: 0.0
    # Feature('FNC187', float, 'functional MRI feature 187'),  # importance: 0.0
    # Feature('FNC163', float, 'functional MRI feature 163'),  # importance: 0.0
    # Feature('FNC265', float, 'functional MRI feature 265'),  # importance: 0.0
    # Feature('FNC349', float, 'functional MRI feature 349'),  # importance: 0.0
    # Feature('FNC219', float, 'functional MRI feature 219'),  # importance: 0.0
    # Feature('SBM_map72', float, 'structural ICA map feature 72'),  # importance: 0.0
    # Feature('FNC195', float, 'functional MRI feature 195'),  # importance: 0.0
    # Feature('FNC18', float, 'functional MRI feature 18'),  # importance: 0.0
    # Feature('FNC313', float, 'functional MRI feature 313'),  # importance: 0.0
    # Feature('FNC354', float, 'functional MRI feature 354'),  # importance: 0.0
    # Feature('FNC125', float, 'functional MRI feature 125'),  # importance: 0.0
    # Feature('FNC19', float, 'functional MRI feature 19'),  # importance: 0.0
    # Feature('FNC122', float, 'functional MRI feature 122'),  # importance: 0.0
    # Feature('FNC261', float, 'functional MRI feature 261'),  # importance: 0.0
    # Feature('FNC247', float, 'functional MRI feature 247'),  # importance: 0.0
    # Feature('FNC17', float, 'functional MRI feature 17'),  # importance: 0.0
    # Feature('FNC228', float, 'functional MRI feature 228'),  # importance: 0.0
    # Feature('FNC27', float, 'functional MRI feature 27'),  # importance: 0.0
    # Feature('FNC92', float, 'functional MRI feature 92'),  # importance: 0.0
    # Feature('SBM_map75', float, 'structural ICA map feature 75'),  # importance: 0.0
    # Feature('FNC108', float, 'functional MRI feature 108'),  # importance: 0.0
    # Feature('FNC29', float, 'functional MRI feature 29'),  # importance: 0.0
    # Feature('FNC154', float, 'functional MRI feature 154'),  # importance: 0.0
    # Feature('FNC335', float, 'functional MRI feature 335'),  # importance: 0.0
    # Feature('FNC65', float, 'functional MRI feature 65'),  # importance: 0.0
    # Feature('FNC106', float, 'functional MRI feature 106'),  # importance: 0.0
    # Feature('FNC66', float, 'functional MRI feature 66'),  # importance: 0.0
    # Feature('FNC23', float, 'functional MRI feature 23'),  # importance: 0.0
    # Feature('FNC123', float, 'functional MRI feature 123'),  # importance: 0.0
    # Feature('FNC183', float, 'functional MRI feature 183'),  # importance: 0.0
    # Feature('FNC365', float, 'functional MRI feature 365'),  # importance: 0.0
    # Feature('FNC318', float, 'functional MRI feature 318'),  # importance: 0.0
    # Feature('FNC244', float, 'functional MRI feature 244'),  # importance: 0.0
    # Feature('FNC252', float, 'functional MRI feature 252'),  # importance: 0.0
    # Feature('FNC148', float, 'functional MRI feature 148'),  # importance: 0.0
    # Feature('FNC330', float, 'functional MRI feature 330'),  # importance: 0.0
    # Feature('FNC171', float, 'functional MRI feature 171'),  # importance: 0.0
    # Feature('SBM_map10', float, 'structural ICA map feature 10'),  # importance: 0.0
    # Feature('FNC327', float, 'functional MRI feature 327'),  # importance: 0.0
    # Feature('SBM_map5', float, 'structural ICA map feature 5'),  # importance: 0.0
    # Feature('SBM_map69', float, 'structural ICA map feature 69'),  # importance: 0.0
    # Feature('FNC279', float, 'functional MRI feature 279'),  # importance: 0.0
    # Feature('FNC181', float, 'functional MRI feature 181'),  # importance: 0.0
    # Feature('FNC166', float, 'functional MRI feature 166'),  # importance: 0.0
    # Feature('FNC306', float, 'functional MRI feature 306'),  # importance: 0.0
    # Feature('FNC149', float, 'functional MRI feature 149'),  # importance: 0.0
    # Feature('FNC133', float, 'functional MRI feature 133'),  # importance: 0.0
    # Feature('FNC230', float, 'functional MRI feature 230'),  # importance: 0.0
    # Feature('FNC315', float, 'functional MRI feature 315'),  # importance: 0.0
], documentation="https://www.kaggle.com/competitions/mlsp-2014-mri/data")

TITANIC_FEATURES = FeatureList(features=[
    Feature('PassengerId', int, name_extended="passenger ID"),
    Feature('Survived', int, is_target=True),
    Feature('Pclass', int, name_extended="passenger class"),
    Feature('Name', cat_dtype),
    Feature('Sex', cat_dtype),
    Feature('Age', float, name_extended='age in years'),
    Feature('SibSp', int,
            name_extended='number of siblings / spouses aboard the Titanic'),
    Feature('Parch', int,
            name_extended='number of parents / children aboard the Titanic'),
    Feature('Ticket', cat_dtype, name_extended='ticket number'),
    Feature('Fare', float, name_extended='passenger fare'),
    Feature('Cabin', cat_dtype, name_extended='cabin number'),
    Feature('Embarked', cat_dtype, name_extended='Port of Embarkation',
            value_mapping={"C": "Cherbourg", "Q": "Queenstown",
                           "S": "Southampton"}),
], documentation="https://www.kaggle.com/competitions/titanic/data")

SANTANDER_TRANSACTION_FEATURES = FeatureList(features=[
    Feature('target', int, is_target=True),
    Feature('var_81', float),  # importance: 0.0156
    Feature('var_139', float),  # importance: 0.0127
    Feature('var_110', float),  # importance: 0.0119
    Feature('var_109', float),  # importance: 0.011
    Feature('var_80', float),  # importance: 0.0107
    Feature('var_12', float),  # importance: 0.0101
    Feature('var_26', float),  # importance: 0.0101
    Feature('var_53', float),  # importance: 0.01
    Feature('var_0', float),  # importance: 0.0098
    Feature('var_99', float),  # importance: 0.0097
    Feature('var_198', float),  # importance: 0.0095
    Feature('var_191', float),  # importance: 0.0091
    Feature('var_76', float),  # importance: 0.0091
    Feature('var_78', float),  # importance: 0.009
    Feature('var_174', float),  # importance: 0.009
    Feature('var_166', float),  # importance: 0.0089
    Feature('var_6', float),  # importance: 0.0088
    Feature('var_179', float),  # importance: 0.0088
    Feature('var_22', float),  # importance: 0.0087
    Feature('var_2', float),  # importance: 0.0087
    Feature('var_1', float),  # importance: 0.0085
    Feature('var_33', float),  # importance: 0.0084
    Feature('var_40', float),  # importance: 0.0081
    Feature('var_44', float),  # importance: 0.0081
    Feature('var_13', float),  # importance: 0.008
    Feature('var_148', float),  # importance: 0.008
    Feature('var_133', float),  # importance: 0.008
    Feature('var_146', float),  # importance: 0.0079
    Feature('var_94', float),  # importance: 0.0079
    Feature('var_169', float),  # importance: 0.0079
    Feature('var_190', float),  # importance: 0.0079
    Feature('var_21', float),  # importance: 0.0079
    Feature('var_164', float),  # importance: 0.0079
    ##################################################
    ##################################################
    # Feature('var_108', float),  # importance: 0.0078
    # Feature('var_170', float),  # importance: 0.0078
    # Feature('var_92', float),  # importance: 0.0076
    # Feature('var_34', float),  # importance: 0.0074
    # Feature('var_123', float),  # importance: 0.0074
    # Feature('var_154', float),  # importance: 0.0072
    # Feature('var_9', float),  # importance: 0.0069
    # Feature('var_192', float),  # importance: 0.0069
    # Feature('var_165', float),  # importance: 0.0069
    # Feature('var_184', float),  # importance: 0.0068
    # Feature('var_75', float),  # importance: 0.0067
    # Feature('var_172', float),  # importance: 0.0067
    # Feature('var_180', float),  # importance: 0.0067
    # Feature('var_93', float),  # importance: 0.0067
    # Feature('var_155', float),  # importance: 0.0066
    # Feature('var_149', float),  # importance: 0.0066
    # Feature('var_121', float),  # importance: 0.0065
    # Feature('var_177', float),  # importance: 0.0065
    # Feature('var_188', float),  # importance: 0.0063
    # Feature('var_67', float),  # importance: 0.0061
    # Feature('var_157', float),  # importance: 0.006
    # Feature('var_5', float),  # importance: 0.006
    # Feature('var_173', float),  # importance: 0.006
    # Feature('var_119', float),  # importance: 0.006
    # Feature('var_91', float),  # importance: 0.006
    # Feature('var_127', float),  # importance: 0.0058
    # Feature('var_18', float),  # importance: 0.0058
    # Feature('var_107', float),  # importance: 0.0058
    # Feature('var_115', float),  # importance: 0.0057
    # Feature('var_35', float),  # importance: 0.0057
    # Feature('var_122', float),  # importance: 0.0057
    # Feature('var_95', float),  # importance: 0.0056
    # Feature('var_118', float),  # importance: 0.0056
    # Feature('var_86', float),  # importance: 0.0056
    # Feature('var_147', float),  # importance: 0.0055
    # Feature('var_32', float),  # importance: 0.0054
    # Feature('var_89', float),  # importance: 0.0054
    # Feature('var_141', float),  # importance: 0.0054
    # Feature('var_162', float),  # importance: 0.0052
    # Feature('var_130', float),  # importance: 0.0052
    # Feature('var_106', float),  # importance: 0.0052
    # Feature('var_150', float),  # importance: 0.0052
    # Feature('var_56', float),  # importance: 0.0051
    # Feature('var_111', float),  # importance: 0.0051
    # Feature('var_197', float),  # importance: 0.0051
    # Feature('var_87', float),  # importance: 0.0051
    # Feature('var_125', float),  # importance: 0.005
    # Feature('var_163', float),  # importance: 0.005
    # Feature('var_167', float),  # importance: 0.005
    # Feature('var_48', float),  # importance: 0.0049
    # Feature('var_186', float),  # importance: 0.0049
    # Feature('var_43', float),  # importance: 0.0048
    # Feature('var_70', float),  # importance: 0.0048
    # Feature('var_36', float),  # importance: 0.0048
    # Feature('var_51', float),  # importance: 0.0046
    # Feature('var_135', float),  # importance: 0.0046
    # Feature('var_145', float),  # importance: 0.0046
    # Feature('var_71', float),  # importance: 0.0045
    # Feature('var_85', float),  # importance: 0.0045
    # Feature('var_131', float),  # importance: 0.0045
    # Feature('var_49', float),  # importance: 0.0044
    # Feature('var_175', float),  # importance: 0.0044
    # Feature('var_52', float),  # importance: 0.0043
    # Feature('var_116', float),  # importance: 0.0043
    # Feature('var_128', float),  # importance: 0.0043
    # Feature('var_132', float),  # importance: 0.0042
    # Feature('var_105', float),  # importance: 0.0042
    # Feature('var_24', float),  # importance: 0.0042
    # Feature('var_195', float),  # importance: 0.0041
    # Feature('var_199', float),  # importance: 0.0041
    # Feature('var_151', float),  # importance: 0.0041
    # Feature('var_112', float),  # importance: 0.0041
    # Feature('var_144', float),  # importance: 0.004
    # Feature('var_114', float),  # importance: 0.004
    # Feature('var_82', float),  # importance: 0.004
    # Feature('var_88', float),  # importance: 0.0039
    # Feature('var_23', float),  # importance: 0.0039
    # Feature('var_194', float),  # importance: 0.0039
    # Feature('var_90', float),  # importance: 0.0039
    # Feature('var_102', float),  # importance: 0.0039
    # Feature('var_31', float),  # importance: 0.0039
    # Feature('var_28', float),  # importance: 0.0039
    # Feature('var_11', float),  # importance: 0.0039
    # Feature('var_74', float),  # importance: 0.0037
    # Feature('var_58', float),  # importance: 0.0037
    # Feature('var_183', float),  # importance: 0.0037
    # Feature('var_45', float),  # importance: 0.0037
    # Feature('var_143', float),  # importance: 0.0037
    # Feature('var_50', float),  # importance: 0.0036
    # Feature('var_196', float),  # importance: 0.0036
    # Feature('var_156', float),  # importance: 0.0036
    # Feature('var_77', float),  # importance: 0.0035
    # Feature('var_97', float),  # importance: 0.0035
    # Feature('var_3', float),  # importance: 0.0035
    # Feature('var_54', float),  # importance: 0.0035
    # Feature('var_27', float),  # importance: 0.0035
    # Feature('var_138', float),  # importance: 0.0035
    # Feature('var_168', float),  # importance: 0.0034
    # Feature('var_134', float),  # importance: 0.0034
    # Feature('var_187', float),  # importance: 0.0034
    # Feature('var_37', float),  # importance: 0.0033
    # Feature('var_83', float),  # importance: 0.0033
    # Feature('var_38', float),  # importance: 0.0033
    # Feature('var_64', float),  # importance: 0.0033
    # Feature('var_176', float),  # importance: 0.0033
    # Feature('var_137', float),  # importance: 0.0033
    # Feature('var_66', float),  # importance: 0.0032
    # Feature('var_10', float),  # importance: 0.0032
    # Feature('var_55', float),  # importance: 0.0032
    # Feature('var_113', float),  # importance: 0.0032
    # Feature('var_152', float),  # importance: 0.0032
    # Feature('var_68', float),  # importance: 0.0032
    # Feature('var_63', float),  # importance: 0.0032
    # Feature('var_104', float),  # importance: 0.0032
    # Feature('var_96', float),  # importance: 0.0032
    # Feature('var_14', float),  # importance: 0.0031
    # Feature('var_4', float),  # importance: 0.0031
    # Feature('var_193', float),  # importance: 0.0031
    # Feature('var_20', float),  # importance: 0.0031
    # Feature('var_62', float),  # importance: 0.0031
    # Feature('var_142', float),  # importance: 0.0031
    # Feature('var_182', float),  # importance: 0.0031
    # Feature('var_178', float),  # importance: 0.0031
    # Feature('var_181', float),  # importance: 0.0031
    # Feature('var_171', float),  # importance: 0.003
    # Feature('var_8', float),  # importance: 0.003
    # Feature('var_84', float),  # importance: 0.003
    # Feature('var_19', float),  # importance: 0.003
    # Feature('var_159', float),  # importance: 0.003
    # Feature('var_59', float),  # importance: 0.003
    # Feature('var_101', float),  # importance: 0.003
    # Feature('var_72', float),  # importance: 0.003
    # Feature('var_103', float),  # importance: 0.0029
    # Feature('var_60', float),  # importance: 0.0029
    # Feature('var_29', float),  # importance: 0.0029
    # Feature('var_57', float),  # importance: 0.0029
    # Feature('var_120', float),  # importance: 0.0029
    # Feature('var_158', float),  # importance: 0.0029
    # Feature('var_153', float),  # importance: 0.0028
    # Feature('var_47', float),  # importance: 0.0028
    # Feature('var_65', float),  # importance: 0.0028
    # Feature('var_189', float),  # importance: 0.0028
    # Feature('var_126', float),  # importance: 0.0027
    # Feature('var_73', float),  # importance: 0.0027
    # Feature('var_136', float),  # importance: 0.0027
    # Feature('var_117', float),  # importance: 0.0027
    # Feature('var_46', float),  # importance: 0.0027
    # Feature('var_129', float),  # importance: 0.0026
    # Feature('var_69', float),  # importance: 0.0026
    # Feature('var_42', float),  # importance: 0.0026
    # Feature('var_160', float),  # importance: 0.0026
    # Feature('var_30', float),  # importance: 0.0025
    # Feature('var_16', float),  # importance: 0.0025
    # Feature('var_15', float),  # importance: 0.0025
    # Feature('var_7', float),  # importance: 0.0025
    # Feature('var_161', float),  # importance: 0.0025
    # Feature('var_61', float),  # importance: 0.0025
    # Feature('var_39', float),  # importance: 0.0025
    # Feature('var_100', float),  # importance: 0.0025
    # Feature('var_140', float),  # importance: 0.0024
    # Feature('var_25', float),  # importance: 0.0024
    # Feature('var_79', float),  # importance: 0.0024
    # Feature('var_185', float),  # importance: 0.0023
    # Feature('var_41', float),  # importance: 0.0022
    # Feature('var_98', float),  # importance: 0.0022
    # Feature('var_124', float),  # importance: 0.0021
    # Feature('var_17', float),  # importance: 0.0021
    # Feature('ID_code', cat_dtype),  # importance: 0.0
],
    documentation="https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data")

HOME_CREDIT_DEFAULT_FEATURES = FeatureList(features=[
    Feature('TARGET', int, is_target=True),
    Feature('EXT_SOURCE_3', float, name_extended="Normalized score from external data source 3"),  # importance: 0.0379
    Feature('EXT_SOURCE_2', float, name_extended="Normalized score from external data source 2"),  # importance: 0.0358
    Feature('FLAG_DOCUMENT_3', float, name_extended="Did client provide document 3",
            value_mapping={1.: "Yes", 0.: "No"}),  # importance: 0.0266
    Feature('CODE_GENDER', cat_dtype, name_extended="Gender of the client",
            value_mapping={'F': 'female', 'M': 'male'}),  # importance: 0.0221
    Feature('NAME_EDUCATION_TYPE', cat_dtype, name_extended="Level of highest education the client achieved"),  # importance: 0.0204
    Feature('NAME_CONTRACT_TYPE', cat_dtype, name_extended="Contract status during the month"),  # importance: 0.0195
    Feature('REGION_RATING_CLIENT_W_CITY', float, name_extended="Our rating of the region where client lives with taking city into account"),  # importance: 0.016
    Feature('EXT_SOURCE_1', float, name_extended="Normalized score from external data source 1"),  # importance: 0.0157
    Feature('AMT_CREDIT', float, name_extended="Maximal amount overdue on the Credit Bureau credit at application date"),  # importance: 0.0146
    Feature('AMT_GOODS_PRICE', float, name_extended="Price of good that client asked for (if applicable) on the previous application"),  # importance: 0.0142
    Feature('ORGANIZATION_TYPE', cat_dtype, name_extended="Type of organization where client works"),  # importance: 0.0133
    Feature('NAME_INCOME_TYPE', cat_dtype, name_extended="Client's income type"),  # importance: 0.0131
    Feature('FLOORSMAX_AVG', float, name_extended="Average number of floors in building where client lives"),  # importance: 0.0121
    Feature('FLAG_WORK_PHONE', float, name_extended="Did client provide work phone",
            value_mapping={1.: "Yes", 0.: "No"}),  # importance: 0.012
    Feature('OWN_CAR_AGE', float, name_extended="Age of client's car"),  # importance: 0.012
    Feature('REG_CITY_NOT_LIVE_CITY', float, name_extended="Flag if client's permanent address does not match contact address",
            value_mapping={1.:"different", 0.: "same"}),  # importance: 0.0119
    Feature('AMT_ANNUITY', float, name_extended="Annuity of the Credit Bureau credit"),  # importance: 0.0116
    Feature('FLAG_DOCUMENT_18', float, name_extended="Did client provide document 3",
            value_mapping={1.: "Yes", 0.: "No"}),  # importance: 0.0115
    Feature('LIVINGAREA_AVG', float),  # importance: 0.0112
    Feature('DAYS_BIRTH', float, name_extended="Client's age in days at the time of application"),  # importance: 0.0108
    Feature('FLAG_DOCUMENT_16', float, name_extended="Did client provide document 16",
            value_mapping={1.: "Yes", 0.: "No"}),  # importance: 0.0108
    Feature('OCCUPATION_TYPE', cat_dtype, name_extended="Occupation type of client"),  # importance: 0.0107
    Feature('APARTMENTS_MODE', float, name_extended="Mode of number of apartments in building where client lives"),  # importance: 0.0107
    Feature('DEF_30_CNT_SOCIAL_CIRCLE', float, name_extended="Number of client's social surroundings defaulted on 30 days past due"),  # importance: 0.0104
    Feature('DEF_60_CNT_SOCIAL_CIRCLE', float, name_extended="Number of client's social surroundings defaulted on 60 days past due"),  # importance: 0.0103
    Feature('AMT_REQ_CREDIT_BUREAU_QRT', float, name_extended="Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)"),  # importance: 0.0103
    Feature('NONLIVINGAPARTMENTS_MEDI', float, name_extended="Mode of number of nonliving apartments in building where client lives"),  # importance: 0.0097
    Feature('DAYS_EMPLOYED', float, name_extended="How many days before the application the person started current employment"),  # importance: 0.0097
    Feature('FLAG_DOCUMENT_5', float, name_extended="Did client provide document 5",
            value_mapping={1.: "Yes", 0.: "No"}),  # importance: 0.0096
    Feature('FLOORSMIN_MEDI', float, name_extended="Median of min number of floors in building where client lives"),  # importance: 0.0095
    Feature('OBS_60_CNT_SOCIAL_CIRCLE', float, name_extended="Number of client's social surroundings with observable 60 DPD days past due default"),  # importance: 0.0095
    Feature('AMT_REQ_CREDIT_BUREAU_DAY', float, name_extended="Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application)"),  # importance: 0.0092
    Feature('EMERGENCYSTATE_MODE', cat_dtype, name_extended="Mode of number of emergency state apartments in building where client lives"),  # importance: 0.0091
    ##################################################
    ##################################################
    # Feature('FLAG_DOCUMENT_9', float),  # importance: 0.0089
    # Feature('ELEVATORS_AVG', float),  # importance: 0.0088
    # Feature('APARTMENTS_MEDI', float),  # importance: 0.0088
    # Feature('FLOORSMAX_MEDI', float),  # importance: 0.0087
    # Feature('NAME_TYPE_SUITE', cat_dtype),  # importance: 0.0086
    # Feature('FLOORSMIN_AVG', float),  # importance: 0.0086
    # Feature('LIVINGAPARTMENTS_MEDI', float),  # importance: 0.0086
    # Feature('DAYS_ID_PUBLISH', float),  # importance: 0.0086
    # Feature('FLAG_OWN_CAR', cat_dtype),  # importance: 0.0085
    # Feature('NAME_FAMILY_STATUS', cat_dtype),  # importance: 0.0085
    # Feature('TOTALAREA_MODE', float),  # importance: 0.0085
    # Feature('LIVINGAREA_MEDI', float),  # importance: 0.0084
    # Feature('FLOORSMAX_MODE', float),  # importance: 0.0083
    # Feature('HOUSETYPE_MODE', cat_dtype),  # importance: 0.0082
    # Feature('DAYS_LAST_PHONE_CHANGE', float),  # importance: 0.0081
    # Feature('FLAG_DOCUMENT_14', float),  # importance: 0.0079
    # Feature('WALLSMATERIAL_MODE', cat_dtype),  # importance: 0.0079
    # Feature('DAYS_REGISTRATION', float),  # importance: 0.0079
    # Feature('AMT_REQ_CREDIT_BUREAU_WEEK', float),  # importance: 0.0079
    # Feature('YEARS_BUILD_AVG', float),  # importance: 0.0079
    # Feature('WEEKDAY_APPR_PROCESS_START', cat_dtype),  # importance: 0.0077
    # Feature('LANDAREA_MODE', float),  # importance: 0.0077
    # Feature('FLAG_PHONE', float),  # importance: 0.0077
    # Feature('YEARS_BEGINEXPLUATATION_MODE', float),  # importance: 0.0076
    # Feature('REGION_POPULATION_RELATIVE', float),  # importance: 0.0075
    # Feature('BASEMENTAREA_AVG', float),  # importance: 0.0075
    # Feature('NONLIVINGAPARTMENTS_MODE', float),  # importance: 0.0075
    # Feature('LIVINGAPARTMENTS_AVG', float),  # importance: 0.0074
    # Feature('NAME_HOUSING_TYPE', cat_dtype),  # importance: 0.0074
    # Feature('COMMONAREA_MODE', float),  # importance: 0.0073
    # Feature('AMT_REQ_CREDIT_BUREAU_YEAR', float),  # importance: 0.0073
    # Feature('SK_ID_CURR', float),  # importance: 0.0072
    # Feature('ENTRANCES_MODE', float),  # importance: 0.0072
    # Feature('HOUR_APPR_PROCESS_START', float),  # importance: 0.0071
    # Feature('OBS_30_CNT_SOCIAL_CIRCLE', float),  # importance: 0.0071
    # Feature('AMT_INCOME_TOTAL', float),  # importance: 0.0071
    # Feature('BASEMENTAREA_MODE', float),  # importance: 0.0071
    # Feature('REG_CITY_NOT_WORK_CITY', float),  # importance: 0.007
    # Feature('ENTRANCES_MEDI', float),  # importance: 0.007
    # Feature('APARTMENTS_AVG', float),  # importance: 0.007
    # Feature('NONLIVINGAREA_MEDI', float),  # importance: 0.0069
    # Feature('LIVINGAPARTMENTS_MODE', float),  # importance: 0.0069
    # Feature('ELEVATORS_MODE', float),  # importance: 0.0069
    # Feature('COMMONAREA_AVG', float),  # importance: 0.0069
    # Feature('COMMONAREA_MEDI', float),  # importance: 0.0069
    # Feature('LIVINGAREA_MODE', float),  # importance: 0.0069
    # Feature('FLOORSMIN_MODE', float),  # importance: 0.0069
    # Feature('LANDAREA_AVG', float),  # importance: 0.0068
    # Feature('FONDKAPREMONT_MODE', cat_dtype),  # importance: 0.0067
    # Feature('LANDAREA_MEDI', float),  # importance: 0.0066
    # Feature('YEARS_BEGINEXPLUATATION_AVG', float),  # importance: 0.0065
    # Feature('YEARS_BEGINEXPLUATATION_MEDI', float),  # importance: 0.0065
    # Feature('REGION_RATING_CLIENT', float),  # importance: 0.0065
    # Feature('BASEMENTAREA_MEDI', float),  # importance: 0.0064
    # Feature('FLAG_OWN_REALTY', cat_dtype),  # importance: 0.0064
    # Feature('CNT_CHILDREN', float),  # importance: 0.0064
    # Feature('LIVE_REGION_NOT_WORK_REGION', float),  # importance: 0.0064
    # Feature('NONLIVINGAREA_MODE', float),  # importance: 0.0064
    # Feature('AMT_REQ_CREDIT_BUREAU_MON', float),  # importance: 0.0064
    # Feature('AMT_REQ_CREDIT_BUREAU_HOUR', float),  # importance: 0.0063
    # Feature('ENTRANCES_AVG', float),  # importance: 0.0062
    # Feature('FLAG_EMAIL', float),  # importance: 0.0062
    # Feature('YEARS_BUILD_MEDI', float),  # importance: 0.0061
    # Feature('NONLIVINGAPARTMENTS_AVG', float),  # importance: 0.006
    # Feature('NONLIVINGAREA_AVG', float),  # importance: 0.006
    # Feature('YEARS_BUILD_MODE', float),  # importance: 0.0059
    # Feature('LIVE_CITY_NOT_WORK_CITY', float),  # importance: 0.0058
    # Feature('CNT_FAM_MEMBERS', float),  # importance: 0.0056
    # Feature('FLAG_DOCUMENT_13', float),  # importance: 0.0049
    # Feature('REG_REGION_NOT_LIVE_REGION', float),  # importance: 0.0048
    # Feature('FLAG_DOCUMENT_8', float),  # importance: 0.0045
    # Feature('FLAG_DOCUMENT_6', float),  # importance: 0.0039
    # Feature('ELEVATORS_MEDI', float),  # importance: 0.0035
    # Feature('FLAG_DOCUMENT_11', float),  # importance: 0.0035
    # Feature('FLAG_DOCUMENT_2', float),  # importance: 0.0031
    # Feature('FLAG_DOCUMENT_15', float),  # importance: 0.0025
    # Feature('REG_REGION_NOT_WORK_REGION', float),  # importance: 0.0024
    # Feature('FLAG_EMP_PHONE', float),  # importance: 0.0008
    # Feature('FLAG_DOCUMENT_10', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_21', float),  # importance: 0.0
    # Feature('FLAG_MOBIL', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_20', float),  # importance: 0.0
    # Feature('FLAG_CONT_MOBILE', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_4', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_17', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_7', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_12', float),  # importance: 0.0
    # Feature('FLAG_DOCUMENT_19', float),  # importance: 0.0
],
    documentation="https://www.kaggle.com/competitions/home-credit-default-risk/data")

IEEE_FRAUD_DETECTION_FEATURES = FeatureList(features=[
    Feature('isFraud', int, is_target=True),
    Feature('V258', float),  # importance: 0.1467
    Feature('V201', float),  # importance: 0.1091
    Feature('V244', float),  # importance: 0.0439
    Feature('V295', float),  # importance: 0.0378
    Feature('V70', float),  # importance: 0.0366
    Feature('V225', float),  # importance: 0.0308
    Feature('V189', float),  # importance: 0.0306
    Feature('V91', float),  # importance: 0.0251
    Feature('V294', float),  # importance: 0.0186
    Feature('V209', float),  # importance: 0.0142
    Feature('C14', float),  # importance: 0.0113
    Feature('V274', float),  # importance: 0.0112
    Feature('id_12', cat_dtype),  # importance: 0.0109
    Feature('C1', float),  # importance: 0.0077
    Feature('V172', float),  # importance: 0.0071
    Feature('V210', float),  # importance: 0.0068
    Feature('C8', float),  # importance: 0.0067
    Feature('V48', float),  # importance: 0.0066
    Feature('V133', float),  # importance: 0.0065
    Feature('V308', float),  # importance: 0.0063
    Feature('V45', float),  # importance: 0.0056
    Feature('V62', float),  # importance: 0.0054
    Feature('DeviceInfo', cat_dtype, name_extended="Device info"),  # importance: 0.0054
    Feature('C11', float),  # importance: 0.0053
    Feature('R_emaildomain', cat_dtype, name_extended='recipient email domain'),  # importance: 0.005
    Feature('V82', float),  # importance: 0.0049
    Feature('V315', float),  # importance: 0.0049
    Feature('C13', float),  # importance: 0.0049
    Feature('C12', float),  # importance: 0.0048
    Feature('V253', float),  # importance: 0.0047
    Feature('V219', float),  # importance: 0.0046
    Feature('V175', float),  # importance: 0.0045
    Feature('V296', float),  # importance: 0.0045
    ##################################################
    ##################################################
    # Feature('V53', float),  # importance: 0.0044
    # Feature('card6', cat_dtype),  # importance: 0.0043
    # Feature('V12', float),  # importance: 0.0043
    # Feature('D2', float),  # importance: 0.0042
    # Feature('V74', float),  # importance: 0.0042
    # Feature('id_17', float),  # importance: 0.0041
    # Feature('V46', float),  # importance: 0.004
    # Feature('V217', float),  # importance: 0.0039
    # Feature('V192', float),  # importance: 0.0038
    # Feature('V312', float),  # importance: 0.0038
    # Feature('V279', float),  # importance: 0.0036
    # Feature('D3', float),  # importance: 0.0035
    # Feature('C5', float),  # importance: 0.0034
    # Feature('ProductCD', cat_dtype),  # importance: 0.0033
    # Feature('card3', float),  # importance: 0.0033
    # Feature('V67', float),  # importance: 0.0033
    # Feature('V198', float),  # importance: 0.0032
    # Feature('V60', float),  # importance: 0.0032
    # Feature('V283', float),  # importance: 0.0031
    # Feature('M5', cat_dtype),  # importance: 0.003
    # Feature('V191', float),  # importance: 0.003
    # Feature('V317', float),  # importance: 0.003
    # Feature('V246', float),  # importance: 0.003
    # Feature('V262', float),  # importance: 0.0029
    # Feature('V177', float),  # importance: 0.0029
    # Feature('V254', float),  # importance: 0.0029
    # Feature('V245', float),  # importance: 0.0028
    # Feature('V79', float),  # importance: 0.0028
    # Feature('V281', float),  # importance: 0.0028
    # Feature('C2', float),  # importance: 0.0028
    # Feature('V51', float),  # importance: 0.0027
    # Feature('V29', float),  # importance: 0.0027
    # Feature('C9', float),  # importance: 0.0026
    # Feature('V83', float),  # importance: 0.0025
    # Feature('V126', float),  # importance: 0.0025
    # Feature('V55', float),  # importance: 0.0025
    # Feature('M4', cat_dtype),  # importance: 0.0025
    # Feature('V194', float),  # importance: 0.0025
    # Feature('M6', cat_dtype),  # importance: 0.0024
    # Feature('C7', float),  # importance: 0.0024
    # Feature('C6', float),  # importance: 0.0024
    # Feature('P_emaildomain', cat_dtype),  # importance: 0.0023
    # Feature('V94', float),  # importance: 0.0023
    # Feature('card4', cat_dtype),  # importance: 0.0022
    # Feature('V49', float),  # importance: 0.0022
    # Feature('V320', float),  # importance: 0.0021
    # Feature('D15', float),  # importance: 0.0021
    # Feature('V33', float),  # importance: 0.0021
    # Feature('V76', float),  # importance: 0.0021
    # Feature('id_31', cat_dtype),  # importance: 0.0021
    # Feature('V131', float),  # importance: 0.002
    # Feature('V280', float),  # importance: 0.002
    # Feature('TransactionAmt', float),  # importance: 0.002
    # Feature('D4', float),  # importance: 0.002
    # Feature('V105', float),  # importance: 0.002
    # Feature('V19', float),  # importance: 0.0019
    # Feature('V125', float),  # importance: 0.0018
    # Feature('V124', float),  # importance: 0.0018
    # Feature('V96', float),  # importance: 0.0018
    # Feature('V318', float),  # importance: 0.0018
    # Feature('M3', cat_dtype),  # importance: 0.0018
    # Feature('V293', float),  # importance: 0.0018
    # Feature('dist1', float),  # importance: 0.0018
    # Feature('V130', float),  # importance: 0.0018
    # Feature('V298', float),  # importance: 0.0018
    # Feature('V87', float),  # importance: 0.0017
    # Feature('V277', float),  # importance: 0.0017
    # Feature('C4', float),  # importance: 0.0017
    # Feature('card2', float),  # importance: 0.0017
    # Feature('V313', float),  # importance: 0.0017
    # Feature('V25', float),  # importance: 0.0017
    # Feature('V66', float),  # importance: 0.0017
    # Feature('V271', float),  # importance: 0.0017
    # Feature('V75', float),  # importance: 0.0016
    # Feature('V102', float),  # importance: 0.0016
    # Feature('M2', cat_dtype),  # importance: 0.0016
    # Feature('V314', float),  # importance: 0.0016
    # Feature('V310', float),  # importance: 0.0016
    # Feature('C10', float),  # importance: 0.0016
    # Feature('D1', float),  # importance: 0.0016
    # Feature('V57', float),  # importance: 0.0016
    # Feature('V207', float),  # importance: 0.0015
    # Feature('D10', float),  # importance: 0.0015
    # Feature('V54', float),  # importance: 0.0015
    # Feature('V186', float),  # importance: 0.0015
    # Feature('card1', float),  # importance: 0.0015
    # Feature('V10', float),  # importance: 0.0015
    # Feature('TransactionDT', float),  # importance: 0.0015
    # Feature('V179', float),  # importance: 0.0015
    # Feature('V256', float),  # importance: 0.0015
    # Feature('V178', float),  # importance: 0.0015
    # Feature('V289', float),  # importance: 0.0015
    # Feature('V250', float),  # importance: 0.0014
    # Feature('V78', float),  # importance: 0.0014
    # Feature('V285', float),  # importance: 0.0014
    # Feature('addr1', float),  # importance: 0.0014
    # Feature('V20', float),  # importance: 0.0014
    # Feature('V169', float),  # importance: 0.0014
    # Feature('V257', float),  # importance: 0.0014
    # Feature('V35', float),  # importance: 0.0014
    # Feature('V80', float),  # importance: 0.0014
    # Feature('V282', float),  # importance: 0.0014
    # Feature('V61', float),  # importance: 0.0014
    # Feature('card5', float),  # importance: 0.0014
    # Feature('V136', float),  # importance: 0.0013
    # Feature('V56', float),  # importance: 0.0013
    # Feature('V86', float),  # importance: 0.0013
    # Feature('V307', float),  # importance: 0.0013
    # Feature('M7', cat_dtype),  # importance: 0.0013
    # Feature('V85', float),  # importance: 0.0013
    # Feature('V73', float),  # importance: 0.0013
    # Feature('V248', float),  # importance: 0.0012
    # Feature('V223', float),  # importance: 0.0012
    # Feature('V129', float),  # importance: 0.0012
    # Feature('V227', float),  # importance: 0.0012
    # Feature('V6', float),  # importance: 0.0012
    # Feature('C3', float),  # importance: 0.0012
    # Feature('V311', float),  # importance: 0.0012
    # Feature('V301', float),  # importance: 0.0012
    # Feature('V215', float),  # importance: 0.0012
    # Feature('V5', float),  # importance: 0.0012
    # Feature('V15', float),  # importance: 0.0012
    # Feature('V243', float),  # importance: 0.0012
    # Feature('V224', float),  # importance: 0.0012
    # Feature('D11', float),  # importance: 0.0011
    # Feature('id_15', cat_dtype),  # importance: 0.0011
    # Feature('V98', float),  # importance: 0.0011
    # Feature('V251', float),  # importance: 0.0011
    # Feature('V92', float),  # importance: 0.0011
    # Feature('V300', float),  # importance: 0.0011
    # Feature('V115', float),  # importance: 0.0011
    # Feature('id_06', float),  # importance: 0.0011
    # Feature('V38', float),  # importance: 0.0011
    # Feature('V97', float),  # importance: 0.0011
    # Feature('V266', float),  # importance: 0.0011
    # Feature('V288', float),  # importance: 0.001
    # Feature('V185', float),  # importance: 0.001
    # Feature('V99', float),  # importance: 0.001
    # Feature('V134', float),  # importance: 0.001
    # Feature('V13', float),  # importance: 0.001
    # Feature('V44', float),  # importance: 0.001
    # Feature('V26', float),  # importance: 0.001
    # Feature('V259', float),  # importance: 0.001
    # Feature('V202', float),  # importance: 0.001
    # Feature('M9', cat_dtype),  # importance: 0.001
    # Feature('V63', float),  # importance: 0.001
    # Feature('id_01', float),  # importance: 0.001
    # Feature('V233', float),  # importance: 0.001
    # Feature('V108', float),  # importance: 0.001
    # Feature('V261', float),  # importance: 0.001
    # Feature('id_20', float),  # importance: 0.001
    # Feature('V268', float),  # importance: 0.001
    # Feature('V208', float),  # importance: 0.001
    # Feature('V11', float),  # importance: 0.001
    # Feature('V32', float),  # importance: 0.0009
    # Feature('id_11', float),  # importance: 0.0009
    # Feature('V127', float),  # importance: 0.0009
    # Feature('V206', float),  # importance: 0.0009
    # Feature('V24', float),  # importance: 0.0009
    # Feature('V9', float),  # importance: 0.0009
    # Feature('V128', float),  # importance: 0.0009
    # Feature('id_05', float),  # importance: 0.0009
    # Feature('V263', float),  # importance: 0.0009
    # Feature('V264', float),  # importance: 0.0009
    # Feature('V59', float),  # importance: 0.0009
    # Feature('V47', float),  # importance: 0.0009
    # Feature('V216', float),  # importance: 0.0009
    # Feature('V284', float),  # importance: 0.0009
    # Feature('V234', float),  # importance: 0.0009
    # Feature('V170', float),  # importance: 0.0009
    # Feature('id_13', float),  # importance: 0.0009
    # Feature('id_02', float),  # importance: 0.0009
    # Feature('V137', float),  # importance: 0.0009
    # Feature('V58', float),  # importance: 0.0008
    # Feature('V2', float),  # importance: 0.0008
    # Feature('V36', float),  # importance: 0.0008
    # Feature('V214', float),  # importance: 0.0008
    # Feature('V72', float),  # importance: 0.0008
    # Feature('V187', float),  # importance: 0.0008
    # Feature('V309', float),  # importance: 0.0008
    # Feature('V242', float),  # importance: 0.0008
    # Feature('V64', float),  # importance: 0.0008
    # Feature('V291', float),  # importance: 0.0008
    # Feature('V182', float),  # importance: 0.0008
    # Feature('V316', float),  # importance: 0.0008
    # Feature('V265', float),  # importance: 0.0008
    # Feature('id_19', float),  # importance: 0.0008
    # Feature('V303', float),  # importance: 0.0007
    # Feature('V42', float),  # importance: 0.0007
    # Feature('V199', float),  # importance: 0.0007
    # Feature('V287', float),  # importance: 0.0007
    # Feature('V270', float),  # importance: 0.0007
    # Feature('DeviceType', cat_dtype),  # importance: 0.0007
    # Feature('V204', float),  # importance: 0.0007
    # Feature('M8', cat_dtype),  # importance: 0.0007
    # Feature('V290', float),  # importance: 0.0007
    # Feature('V37', float),  # importance: 0.0007
    # Feature('D5', float),  # importance: 0.0007
    # Feature('V77', float),  # importance: 0.0007
    # Feature('V306', float),  # importance: 0.0007
    # Feature('V211', float),  # importance: 0.0007
    # Feature('V69', float),  # importance: 0.0007
    # Feature('V267', float),  # importance: 0.0007
    # Feature('V81', float),  # importance: 0.0007
    # Feature('V112', float),  # importance: 0.0006
    # Feature('V321', float),  # importance: 0.0006
    # Feature('V8', float),  # importance: 0.0006
    # Feature('id_38', cat_dtype),  # importance: 0.0006
    # Feature('V276', float),  # importance: 0.0006
    # Feature('V23', float),  # importance: 0.0006
    # Feature('V205', float),  # importance: 0.0006
    # Feature('V135', float),  # importance: 0.0006
    # Feature('V93', float),  # importance: 0.0006
    # Feature('id_28', cat_dtype),  # importance: 0.0006
    # Feature('V220', float),  # importance: 0.0006
    # Feature('V273', float),  # importance: 0.0006
    # Feature('V90', float),  # importance: 0.0006
    # Feature('V43', float),  # importance: 0.0006
    # Feature('V260', float),  # importance: 0.0006
    # Feature('V229', float),  # importance: 0.0006
    # Feature('V109', float),  # importance: 0.0006
    # Feature('V84', float),  # importance: 0.0006
    # Feature('V40', float),  # importance: 0.0006
    # Feature('V319', float),  # importance: 0.0005
    # Feature('V95', float),  # importance: 0.0005
    # Feature('V212', float),  # importance: 0.0005
    # Feature('V255', float),  # importance: 0.0005
    # Feature('V247', float),  # importance: 0.0005
    # Feature('V3', float),  # importance: 0.0005
    # Feature('V16', float),  # importance: 0.0005
    # Feature('V221', float),  # importance: 0.0005
    # Feature('V238', float),  # importance: 0.0005
    # Feature('V34', float),  # importance: 0.0005
    # Feature('V39', float),  # importance: 0.0005
    # Feature('V269', float),  # importance: 0.0004
    # Feature('id_16', cat_dtype),  # importance: 0.0004
    # Feature('V183', float),  # importance: 0.0004
    # Feature('V232', float),  # importance: 0.0004
    # Feature('V292', float),  # importance: 0.0004
    # Feature('V30', float),  # importance: 0.0004
    # Feature('id_37', cat_dtype),  # importance: 0.0004
    # Feature('V275', float),  # importance: 0.0004
    # Feature('V278', float),  # importance: 0.0004
    # Feature('V106', float),  # importance: 0.0004
    # Feature('V171', float),  # importance: 0.0004
    # Feature('V237', float),  # importance: 0.0004
    # Feature('V203', float),  # importance: 0.0004
    # Feature('V286', float),  # importance: 0.0004
    # Feature('V188', float),  # importance: 0.0004
    # Feature('V230', float),  # importance: 0.0004
    # Feature('V222', float),  # importance: 0.0004
    # Feature('V272', float),  # importance: 0.0003
    # Feature('V7', float),  # importance: 0.0003
    # Feature('V123', float),  # importance: 0.0003
    # Feature('V213', float),  # importance: 0.0003
    # Feature('V228', float),  # importance: 0.0003
    # Feature('V297', float),  # importance: 0.0003
    # Feature('addr2', float),  # importance: 0.0003
    # Feature('V132', float),  # importance: 0.0003
    # Feature('V302', float),  # importance: 0.0003
    # Feature('V121', float),  # importance: 0.0002
    # Feature('V231', float),  # importance: 0.0002
    # Feature('V168', float),  # importance: 0.0002
    # Feature('V200', float),  # importance: 0.0002
    # Feature('V4', float),  # importance: 0.0002
    # Feature('V190', float),  # importance: 0.0002
    # Feature('V22', float),  # importance: 0.0001
    # Feature('V116', float),  # importance: 0.0
    # Feature('V113', float),  # importance: 0.0
    # Feature('V195', float),  # importance: 0.0
    # Feature('V181', float),  # importance: 0.0
    # Feature('V27', float),  # importance: 0.0
    # Feature('V14', float),  # importance: 0.0
    # Feature('V120', float),  # importance: 0.0
    # Feature('V305', float),  # importance: 0.0
    # Feature('V101', float),  # importance: 0.0
    # Feature('V88', float),  # importance: 0.0
    # Feature('V18', float),  # importance: 0.0
    # Feature('TransactionID', float),  # importance: 0.0
    # Feature('V50', float),  # importance: 0.0
    # Feature('V218', float),  # importance: 0.0
    # Feature('V226', float),  # importance: 0.0
    # Feature('id_36', cat_dtype),  # importance: 0.0
    # Feature('V249', float),  # importance: 0.0
    # Feature('V180', float),  # importance: 0.0
    # Feature('V28', float),  # importance: 0.0
    # Feature('V21', float),  # importance: 0.0
    # Feature('id_29', cat_dtype),  # importance: 0.0
    # Feature('V71', float),  # importance: 0.0
    # Feature('V299', float),  # importance: 0.0
    # Feature('V31', float),  # importance: 0.0
    # Feature('V52', float),  # importance: 0.0
    # Feature('V304', float),  # importance: 0.0
    # Feature('V167', float),  # importance: 0.0
    # Feature('V114', float),  # importance: 0.0
    # Feature('V110', float),  # importance: 0.0
    # Feature('V118', float),  # importance: 0.0
    # Feature('V89', float),  # importance: 0.0
    # Feature('V240', float),  # importance: 0.0
    # Feature('V122', float),  # importance: 0.0
    # Feature('V197', float),  # importance: 0.0
    # Feature('V241', float),  # importance: 0.0
    # Feature('V235', float),  # importance: 0.0
    # Feature('V107', float),  # importance: 0.0
    # Feature('V184', float),  # importance: 0.0
    # Feature('V173', float),  # importance: 0.0
    # Feature('V111', float),  # importance: 0.0
    # Feature('V41', float),  # importance: 0.0
    # Feature('V65', float),  # importance: 0.0
    # Feature('id_35', cat_dtype),  # importance: 0.0
    # Feature('V104', float),  # importance: 0.0
    # Feature('V196', float),  # importance: 0.0
    # Feature('V236', float),  # importance: 0.0
    # Feature('V1', float),  # importance: 0.0
    # Feature('V252', float),  # importance: 0.0
    # Feature('V176', float),  # importance: 0.0
    # Feature('V103', float),  # importance: 0.0
    # Feature('V100', float),  # importance: 0.0
    # Feature('V174', float),  # importance: 0.0
    # Feature('V193', float),  # importance: 0.0
    # Feature('V117', float),  # importance: 0.0
    # Feature('V68', float),  # importance: 0.0
    # Feature('V119', float),  # importance: 0.0
    # Feature('M1', cat_dtype),  # importance: 0.0
    # Feature('V239', float),  # importance: 0.0
    # Feature('V17', float),  # importance: 0.0
],
    documentation='https://www.kaggle.com/competitions/ieee-fraud-detection/data')

SAFE_DRIVER_PREDICTION_FEATURES = FeatureList(features=[
    Feature('id', int),
    Feature('target', int, is_target=True),
    Feature('ps_ind_01', int, name_extended='individual feature 1'),
    Feature('ps_ind_02_cat', int, name_extended='individual categorical feature 2'),
    Feature('ps_ind_03', int, name_extended='individual feature 3'),
    Feature('ps_ind_04_cat', int, name_extended='individual categorical feature 4'),
    Feature('ps_ind_05_cat', int, name_extended='individual categorical feature 5'),
    Feature('ps_ind_06_bin', int, name_extended='individual binary feature 6'),
    Feature('ps_ind_07_bin', int, name_extended='individual binary feature 7'),
    Feature('ps_ind_08_bin', int, name_extended='individual binary feature 8'),
    Feature('ps_ind_09_bin', int, name_extended='individual binary feature 9'),
    Feature('ps_ind_10_bin', int, name_extended='individual binary feature 0'),
    Feature('ps_ind_11_bin', int, name_extended='individual binary feature 1'),
    Feature('ps_ind_12_bin', int, name_extended='individual binary feature 2'),
    Feature('ps_ind_13_bin', int, name_extended='individual binary feature 3'),
    Feature('ps_ind_14', int, name_extended='individual feature 4'),
    Feature('ps_ind_15', int, name_extended='individual feature 5'),
    Feature('ps_ind_16_bin', int, name_extended='individual binary feature 6'),
    Feature('ps_ind_17_bin', int, name_extended='individual binary feature 7'),
    Feature('ps_ind_18_bin', int, name_extended='individual binary feature 8'),
    Feature('ps_reg_01', float, name_extended='registration feature 1'),
    Feature('ps_reg_02', float, name_extended='registration feature 2'),
    Feature('ps_reg_03', float, name_extended='registration feature 3'),
    Feature('ps_car_01_cat', int, name_extended='car categorical feature 1'),
    Feature('ps_car_02_cat', int, name_extended='car categorical feature 2'),
    Feature('ps_car_03_cat', int, name_extended='car categorical feature 3'),
    Feature('ps_car_04_cat', int, name_extended='car categorical feature 4'),
    Feature('ps_car_05_cat', int, name_extended='car categorical feature 5'),
    Feature('ps_car_06_cat', int, name_extended='car categorical feature 6'),
    Feature('ps_car_07_cat', int, name_extended='car categorical feature 7'),
    Feature('ps_car_08_cat', int, name_extended='car categorical feature 8'),
    Feature('ps_car_09_cat', int, name_extended='car categorical feature 9'),
    Feature('ps_car_10_cat', int, name_extended='car categorical feature 0'),
    Feature('ps_car_11_cat', int, name_extended='car categorical feature 1'),
    Feature('ps_car_11', int, name_extended='car feature 1'),
    Feature('ps_car_12', float, name_extended='car feature 2'),
    Feature('ps_car_13', float, name_extended='car feature 3'),
    Feature('ps_car_14', float, name_extended='car feature 4'),
    Feature('ps_car_15', float, name_extended='car feature 5'),
    Feature('ps_calc_01', float, name_extended='calculated feature 01'),
    Feature('ps_calc_02', float, name_extended='calculated feature 02'),
    Feature('ps_calc_03', float, name_extended='calculated feature 03'),
    Feature('ps_calc_04', int, name_extended='calculated feature 04'),
    Feature('ps_calc_05', int, name_extended='calculated feature 05'),
    Feature('ps_calc_06', int, name_extended='calculated feature 06'),
    Feature('ps_calc_07', int, name_extended='calculated feature 07'),
    Feature('ps_calc_08', int, name_extended='calculated feature 08'),
    Feature('ps_calc_09', int, name_extended='calculated feature 09'),
    Feature('ps_calc_10', int, name_extended='calculated feature 10'),
    Feature('ps_calc_11', int, name_extended='calculated feature 11'),
    Feature('ps_calc_12', int, name_extended='calculated feature 12'),
    Feature('ps_calc_13', int, name_extended='calculated feature 13'),
    Feature('ps_calc_14', int, name_extended='calculated feature 14'),
    Feature('ps_calc_15_bin', int, name_extended='calculated binary feature 15'),
    Feature('ps_calc_16_bin', int, name_extended='calculated binary feature 16'),
    Feature('ps_calc_17_bin', int, name_extended='calculated binary feature 17'),
    Feature('ps_calc_18_bin', int, name_extended='calculated binary feature 18'),
    Feature('ps_calc_19_bin', int, name_extended='calculated binary feature 19'),
    Feature('ps_calc_20_bin', int, name_extended='calculated binary feature 20'),

],
    documentation='https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data')

SANTANDER_CUSTOMER_SATISFACTION_FEATURES = FeatureList(features=[
    Feature('TARGET', int, is_target=True),
    Feature('saldo_var30', float),  # importance: 0.0634
    Feature('var15', float),  # importance: 0.0323
    Feature('num_var30_0', float),  # importance: 0.0238
    Feature('ind_var25_cte', float),  # importance: 0.0213
    Feature('saldo_medio_var8_ult1', float),  # importance: 0.0191
    Feature('var21', float),  # importance: 0.0182
    Feature('imp_op_var41_efect_ult1', float),  # importance: 0.0162
    Feature('num_var41_0', float),  # importance: 0.0162
    Feature('ind_var26_cte', float),  # importance: 0.0151
    Feature('ind_var39_0', float),  # importance: 0.0144
    Feature('imp_var7_recib_ult1', float),  # importance: 0.0136
    Feature('num_var26_0', float),  # importance: 0.0132
    Feature('ind_var10cte_ult1', float),  # importance: 0.0131
    Feature('var3', float),  # importance: 0.0128
    Feature('imp_op_var41_efect_ult3', float),  # importance: 0.0126
    Feature('num_meses_var12_ult3', float),  # importance: 0.0124
    Feature('ind_var9_ult1', float),  # importance: 0.012
    Feature('saldo_medio_var5_ult1', float),  # importance: 0.0117
    Feature('saldo_medio_var5_hace2', float),  # importance: 0.0117
    Feature('ind_var12_0', float),  # importance: 0.0115
    Feature('ind_var8_0', float),  # importance: 0.0113
    Feature('num_aport_var13_hace3', float),  # importance: 0.0109
    Feature('saldo_medio_var5_ult3', float),  # importance: 0.0106
    Feature('num_meses_var5_ult3', float),  # importance: 0.0106
    Feature('num_var37_0', float),  # importance: 0.0105
    Feature('ind_var41_0', float),  # importance: 0.0103
    Feature('ind_var5', float),  # importance: 0.0102
    Feature('var38', float),  # importance: 0.0101
    Feature('num_var4', float),  # importance: 0.01
    Feature('num_meses_var39_vig_ult3', float),  # importance: 0.0098
    Feature('num_var22_ult1', float),  # importance: 0.0098
    Feature('saldo_medio_var8_ult3', float),  # importance: 0.0096
    Feature('var36', float),  # importance: 0.0094
    ##################################################
    ##################################################
    # Feature('imp_op_var39_efect_ult1', float),  # importance: 0.0094
    # Feature('imp_op_var39_ult1', float),  # importance: 0.0093
    # Feature('saldo_medio_var5_hace3', float),  # importance: 0.0092
    # Feature('num_op_var39_comer_ult1', float),  # importance: 0.0092
    # Feature('num_var22_hace2', float),  # importance: 0.0091
    # Feature('imp_ent_var16_ult1', float),  # importance: 0.0091
    # Feature('num_var45_hace2', float),  # importance: 0.009
    # Feature('num_op_var41_ult1', float),  # importance: 0.0089
    # Feature('imp_op_var41_ult1', float),  # importance: 0.0089
    # Feature('imp_var43_emit_ult1', float),  # importance: 0.0089
    # Feature('saldo_var5', float),  # importance: 0.0085
    # Feature('imp_op_var41_comer_ult1', float),  # importance: 0.0085
    # Feature('num_op_var41_ult3', float),  # importance: 0.0083
    # Feature('num_var45_hace3', float),  # importance: 0.0082
    # Feature('saldo_var42', float),  # importance: 0.0082
    # Feature('saldo_medio_var13_corto_hace2', float),  # importance: 0.008
    # Feature('saldo_var26', float),  # importance: 0.0079
    # Feature('num_op_var41_comer_ult3', float),  # importance: 0.0078
    # Feature('ID', float),  # importance: 0.0078
    # Feature('num_op_var39_efect_ult1', float),  # importance: 0.0078
    # Feature('ind_var14_0', float),  # importance: 0.0077
    # Feature('ind_var43_emit_ult1', float),  # importance: 0.0077
    # Feature('num_var22_ult3', float),  # importance: 0.0077
    # Feature('imp_op_var41_comer_ult3', float),  # importance: 0.0076
    # Feature('num_op_var39_hace2', float),  # importance: 0.0076
    # Feature('num_var45_ult1', float),  # importance: 0.0075
    # Feature('num_var22_hace3', float),  # importance: 0.0073
    # Feature('num_op_var39_efect_ult3', float),  # importance: 0.0071
    # Feature('imp_op_var39_comer_ult1', float),  # importance: 0.0071
    # Feature('num_var42_0', float),  # importance: 0.0071
    # Feature('ind_var30', float),  # importance: 0.0071
    # Feature('num_var35', float),  # importance: 0.007
    # Feature('num_var45_ult3', float),  # importance: 0.007
    # Feature('num_var43_recib_ult1', float),  # importance: 0.0069
    # Feature('num_op_var39_ult1', float),  # importance: 0.0067
    # Feature('saldo_var37', float),  # importance: 0.0067
    # Feature('ind_var10_ult1', float),  # importance: 0.0066
    # Feature('imp_trans_var37_ult1', float),  # importance: 0.0066
    # Feature('imp_op_var39_comer_ult3', float),  # importance: 0.0065
    # Feature('ind_var9_cte_ult1', float),  # importance: 0.0065
    # Feature('saldo_medio_var12_hace2', float),  # importance: 0.0064
    # Feature('num_var43_emit_ult1', float),  # importance: 0.0062
    # Feature('saldo_medio_var12_ult3', float),  # importance: 0.006
    # Feature('saldo_var13_corto', float),  # importance: 0.0059
    # Feature('saldo_var13_largo', float),  # importance: 0.0057
    # Feature('imp_sal_var16_ult1', float),  # importance: 0.0057
    # Feature('num_op_var39_ult3', float),  # importance: 0.0056
    # Feature('saldo_medio_var8_hace2', float),  # importance: 0.0056
    # Feature('num_var37_med_ult2', float),  # importance: 0.0056
    # Feature('num_op_var41_hace2', float),  # importance: 0.0056
    # Feature('saldo_medio_var8_hace3', float),  # importance: 0.0055
    # Feature('num_op_var39_comer_ult3', float),  # importance: 0.0055
    # Feature('num_op_var39_hace3', float),  # importance: 0.0054
    # Feature('saldo_medio_var13_corto_ult3', float),  # importance: 0.0053
    # Feature('saldo_var13', float),  # importance: 0.0052
    # Feature('saldo_medio_var12_ult1', float),  # importance: 0.0047
    # Feature('imp_aport_var13_ult1', float),  # importance: 0.0047
    # Feature('num_op_var41_comer_ult1', float),  # importance: 0.0047
    # Feature('saldo_var24', float),  # importance: 0.0046
    # Feature('ind_var37_cte', float),  # importance: 0.0045
    # Feature('num_var5_0', float),  # importance: 0.0045
    # Feature('ind_var24_0', float),  # importance: 0.0044
    # Feature('imp_compra_var44_ult1', float),  # importance: 0.0044
    # Feature('num_op_var41_efect_ult3', float),  # importance: 0.0043
    # Feature('num_ent_var16_ult1', float),  # importance: 0.0042
    # Feature('num_op_var41_efect_ult1', float),  # importance: 0.0041
    # Feature('num_var39_0', float),  # importance: 0.004
    # Feature('num_var13_0', float),  # importance: 0.004
    # Feature('num_meses_var8_ult3', float),  # importance: 0.0039
    # Feature('saldo_var25', float),  # importance: 0.0039
    # Feature('num_var31_0', float),  # importance: 0.0038
    # Feature('num_meses_var13_corto_ult3', float),  # importance: 0.0038
    # Feature('num_var5', float),  # importance: 0.0036
    # Feature('num_var25', float),  # importance: 0.0035
    # Feature('num_var32_0', float),  # importance: 0.0033
    # Feature('saldo_var40', float),  # importance: 0.0032
    # Feature('num_var30', float),  # importance: 0.0032
    # Feature('saldo_var12', float),  # importance: 0.003
    # Feature('num_sal_var16_ult1', float),  # importance: 0.0027
    # Feature('saldo_medio_var12_hace3', float),  # importance: 0.0024
    # Feature('imp_op_var39_efect_ult3', float),  # importance: 0.0022
    # Feature('imp_aport_var13_hace3', float),  # importance: 0.0016
    # Feature('saldo_var8', float),  # importance: 0.0013
    # Feature('imp_op_var40_efect_ult3', float),  # importance: 0.0011
    # Feature('saldo_var2_ult1', float),  # importance: 0.0
    # Feature('num_aport_var33_hace3', float),  # importance: 0.0
    # Feature('ind_var34_0', float),  # importance: 0.0
    # Feature('ind_var13_corto_0', float),  # importance: 0.0
    # Feature('saldo_medio_var33_hace2', float),  # importance: 0.0
    # Feature('num_var46_0', float),  # importance: 0.0
    # Feature('ind_var44_0', float),  # importance: 0.0
    # Feature('num_var33_0', float),  # importance: 0.0
    # Feature('saldo_medio_var29_hace3', float),  # importance: 0.0
    # Feature('ind_var5_0', float),  # importance: 0.0
    # Feature('ind_var34', float),  # importance: 0.0
    # Feature('saldo_var13_medio', float),  # importance: 0.0
    # Feature('ind_var40', float),  # importance: 0.0
    # Feature('num_trasp_var33_in_ult1', float),  # importance: 0.0
    # Feature('num_var24_0', float),  # importance: 0.0
    # Feature('num_var7_recib_ult1', float),  # importance: 0.0
    # Feature('num_var7_emit_ult1', float),  # importance: 0.0
    # Feature('ind_var31', float),  # importance: 0.0
    # Feature('num_reemb_var13_ult1', float),  # importance: 0.0
    # Feature('num_var46', float),  # importance: 0.0
    # Feature('num_var39', float),  # importance: 0.0
    # Feature('imp_venta_var44_ult1', float),  # importance: 0.0
    # Feature('ind_var32_cte', float),  # importance: 0.0
    # Feature('imp_aport_var33_hace3', float),  # importance: 0.0
    # Feature('num_var13_medio', float),  # importance: 0.0
    # Feature('imp_reemb_var17_hace3', float),  # importance: 0.0
    # Feature('num_var13_medio_0', float),  # importance: 0.0
    # Feature('num_var8_0', float),  # importance: 0.0
    # Feature('saldo_var46', float),  # importance: 0.0
    # Feature('saldo_var32', float),  # importance: 0.0
    # Feature('ind_var32_0', float),  # importance: 0.0
    # Feature('num_op_var40_efect_ult1', float),  # importance: 0.0
    # Feature('ind_var26', float),  # importance: 0.0
    # Feature('ind_var13_medio_0', float),  # importance: 0.0
    # Feature('ind_var39', float),  # importance: 0.0
    # Feature('num_op_var40_comer_ult1', float),  # importance: 0.0
    # Feature('ind_var28', float),  # importance: 0.0
    # Feature('delta_num_venta_var44_1y3', float),  # importance: 0.0
    # Feature('num_var29', float),  # importance: 0.0
    # Feature('num_var37', float),  # importance: 0.0
    # Feature('num_venta_var44_ult1', float),  # importance: 0.0
    # Feature('ind_var33', float),  # importance: 0.0
    # Feature('ind_var43_recib_ult1', float),  # importance: 0.0
    # Feature('saldo_var18', float),  # importance: 0.0
    # Feature('imp_amort_var18_hace3', float),  # importance: 0.0
    # Feature('saldo_var27', float),  # importance: 0.0
    # Feature('saldo_medio_var17_ult1', float),  # importance: 0.0
    # Feature('num_med_var22_ult3', float),  # importance: 0.0
    # Feature('ind_var12', float),  # importance: 0.0
    # Feature('ind_var25', float),  # importance: 0.0
    # Feature('imp_trasp_var17_in_ult1', float),  # importance: 0.0
    # Feature('ind_var27_0', float),  # importance: 0.0
    # Feature('ind_var19', float),  # importance: 0.0
    # Feature('saldo_medio_var13_medio_ult1', float),  # importance: 0.0
    # Feature('delta_imp_trasp_var33_out_1y3', float),  # importance: 0.0
    # Feature('num_var12', float),  # importance: 0.0
    # Feature('num_var6_0', float),  # importance: 0.0
    # Feature('saldo_medio_var44_hace2', float),  # importance: 0.0
    # Feature('saldo_var44', float),  # importance: 0.0
    # Feature('num_var34', float),  # importance: 0.0
    # Feature('delta_num_aport_var33_1y3', float),  # importance: 0.0
    # Feature('ind_var6', float),  # importance: 0.0
    # Feature('ind_var6_0', float),  # importance: 0.0
    # Feature('num_var13_largo_0', float),  # importance: 0.0
    # Feature('num_var41', float),  # importance: 0.0
    # Feature('ind_var37_0', float),  # importance: 0.0
    # Feature('num_var25_0', float),  # importance: 0.0
    # Feature('imp_trasp_var33_in_ult1', float),  # importance: 0.0
    # Feature('ind_var13_corto', float),  # importance: 0.0
    # Feature('ind_var30_0', float),  # importance: 0.0
    # Feature('delta_num_trasp_var33_in_1y3', float),  # importance: 0.0
    # Feature('saldo_medio_var33_hace3', float),  # importance: 0.0
    # Feature('num_var33', float),  # importance: 0.0
    # Feature('ind_var26_0', float),  # importance: 0.0
    # Feature('saldo_medio_var13_largo_hace3', float),  # importance: 0.0
    # Feature('ind_var13_0', float),  # importance: 0.0
    # Feature('delta_imp_trasp_var33_in_1y3', float),  # importance: 0.0
    # Feature('saldo_medio_var13_medio_ult3', float),  # importance: 0.0
    # Feature('delta_imp_amort_var34_1y3', float),  # importance: 0.0
    # Feature('saldo_medio_var17_hace3', float),  # importance: 0.0
    # Feature('num_meses_var33_ult3', float),  # importance: 0.0
    # Feature('saldo_medio_var17_ult3', float),  # importance: 0.0
    # Feature('imp_var7_emit_ult1', float),  # importance: 0.0
    # Feature('num_var12_0', float),  # importance: 0.0
    # Feature('num_reemb_var33_ult1', float),  # importance: 0.0
    # Feature('saldo_medio_var13_largo_ult1', float),  # importance: 0.0
    # Feature('imp_reemb_var13_ult1', float),  # importance: 0.0
    # Feature('num_trasp_var17_out_hace3', float),  # importance: 0.0
    # Feature('imp_op_var40_comer_ult1', float),  # importance: 0.0
    # Feature('num_compra_var44_ult1', float),  # importance: 0.0
    # Feature('num_var31', float),  # importance: 0.0
    # Feature('imp_aport_var17_hace3', float),  # importance: 0.0
    # Feature('num_aport_var17_ult1', float),  # importance: 0.0
    # Feature('num_var26', float),  # importance: 0.0
    # Feature('num_var44_0', float),  # importance: 0.0
    # Feature('ind_var17', float),  # importance: 0.0
    # Feature('num_op_var40_hace2', float),  # importance: 0.0
    # Feature('imp_trasp_var33_out_ult1', float),  # importance: 0.0
    # Feature('num_med_var45_ult3', float),  # importance: 0.0
    # Feature('ind_var28_0', float),  # importance: 0.0
    # Feature('num_var13', float),  # importance: 0.0
    # Feature('imp_reemb_var33_hace3', float),  # importance: 0.0
    # Feature('delta_num_trasp_var33_out_1y3', float),  # importance: 0.0
    # Feature('imp_trasp_var17_in_hace3', float),  # importance: 0.0
    # Feature('delta_imp_amort_var18_1y3', float),  # importance: 0.0
    # Feature('num_var20_0', float),  # importance: 0.0
    # Feature('num_var18', float),  # importance: 0.0
    # Feature('delta_imp_aport_var13_1y3', float),  # importance: 0.0
    # Feature('delta_imp_venta_var44_1y3', float),  # importance: 0.0
    # Feature('num_venta_var44_hace3', float),  # importance: 0.0
    # Feature('num_var20', float),  # importance: 0.0
    # Feature('saldo_medio_var13_medio_hace2', float),  # importance: 0.0
    # Feature('num_var40_0', float),  # importance: 0.0
    # Feature('saldo_var41', float),  # importance: 0.0
    # Feature('num_var2_0_ult1', float),  # importance: 0.0
    # Feature('saldo_medio_var13_corto_hace3', float),  # importance: 0.0
    # Feature('imp_op_var40_efect_ult1', float),  # importance: 0.0
    # Feature('num_var28_0', float),  # importance: 0.0
    # Feature('ind_var14', float),  # importance: 0.0
    # Feature('ind_var37', float),  # importance: 0.0
    # Feature('delta_imp_reemb_var13_1y3', float),  # importance: 0.0
    # Feature('imp_reemb_var17_ult1', float),  # importance: 0.0
    # Feature('delta_imp_trasp_var17_out_1y3', float),  # importance: 0.0
    # Feature('saldo_medio_var13_largo_ult3', float),  # importance: 0.0
    # Feature('imp_trasp_var33_in_hace3', float),  # importance: 0.0
    # Feature('num_trasp_var33_out_hace3', float),  # importance: 0.0
    # Feature('num_var44', float),  # importance: 0.0
    # Feature('num_var28', float),  # importance: 0.0
    # Feature('num_var34_0', float),  # importance: 0.0
    # Feature('ind_var46_0', float),  # importance: 0.0
    # Feature('saldo_medio_var13_corto_ult1', float),  # importance: 0.0
    # Feature('saldo_var33', float),  # importance: 0.0
    # Feature('ind_var8', float),  # importance: 0.0
    # Feature('num_trasp_var33_out_ult1', float),  # importance: 0.0
    # Feature('num_trasp_var33_in_hace3', float),  # importance: 0.0
    # Feature('imp_op_var40_ult1', float),  # importance: 0.0
    # Feature('saldo_var14', float),  # importance: 0.0
    # Feature('num_reemb_var17_ult1', float),  # importance: 0.0
    # Feature('num_aport_var17_hace3', float),  # importance: 0.0
    # Feature('ind_var13', float),  # importance: 0.0
    # Feature('delta_imp_compra_var44_1y3', float),  # importance: 0.0
    # Feature('ind_var7_emit_ult1', float),  # importance: 0.0
    # Feature('num_reemb_var13_hace3', float),  # importance: 0.0
    # Feature('ind_var29_0', float),  # importance: 0.0
    # Feature('delta_imp_aport_var33_1y3', float),  # importance: 0.0
    # Feature('saldo_medio_var44_ult1', float),  # importance: 0.0
    # Feature('num_var40', float),  # importance: 0.0
    # Feature('num_var17_0', float),  # importance: 0.0
    # Feature('num_op_var41_hace3', float),  # importance: 0.0
    # Feature('ind_var13_medio', float),  # importance: 0.0
    # Feature('num_aport_var33_ult1', float),  # importance: 0.0
    # Feature('num_trasp_var11_ult1', float),  # importance: 0.0
    # Feature('num_op_var40_ult3', float),  # importance: 0.0
    # Feature('num_aport_var13_ult1', float),  # importance: 0.0
    # Feature('num_reemb_var33_hace3', float),  # importance: 0.0
    # Feature('num_var29_0', float),  # importance: 0.0
    # Feature('saldo_medio_var44_ult3', float),  # importance: 0.0
    # Feature('delta_num_reemb_var33_1y3', float),  # importance: 0.0
    # Feature('saldo_medio_var33_ult1', float),  # importance: 0.0
    # Feature('ind_var41', float),  # importance: 0.0
    # Feature('ind_var33_0', float),  # importance: 0.0
    # Feature('num_op_var40_hace3', float),  # importance: 0.0
    # Feature('ind_var17_0', float),  # importance: 0.0
    # Feature('ind_var20', float),  # importance: 0.0
    # Feature('ind_var24', float),  # importance: 0.0
    # Feature('imp_aport_var33_ult1', float),  # importance: 0.0
    # Feature('ind_var18', float),  # importance: 0.0
    # Feature('saldo_var20', float),  # importance: 0.0
    # Feature('num_var27_0', float),  # importance: 0.0
    # Feature('delta_num_compra_var44_1y3', float),  # importance: 0.0
    # Feature('delta_num_trasp_var17_in_1y3', float),  # importance: 0.0
    # Feature('imp_amort_var34_ult1', float),  # importance: 0.0
    # Feature('ind_var46', float),  # importance: 0.0
    # Feature('num_var18_0', float),  # importance: 0.0
    # Feature('ind_var27', float),  # importance: 0.0
    # Feature('num_var13_largo', float),  # importance: 0.0
    # Feature('num_op_var40_comer_ult3', float),  # importance: 0.0
    # Feature('num_var1', float),  # importance: 0.0
    # Feature('imp_venta_var44_hace3', float),  # importance: 0.0
    # Feature('saldo_var28', float),  # importance: 0.0
    # Feature('num_var2_ult1', float),  # importance: 0.0
    # Feature('num_var24', float),  # importance: 0.0
    # Feature('num_compra_var44_hace3', float),  # importance: 0.0
    # Feature('num_var17', float),  # importance: 0.0
    # Feature('delta_num_reemb_var13_1y3', float),  # importance: 0.0
    # Feature('imp_trasp_var33_out_hace3', float),  # importance: 0.0
    # Feature('saldo_medio_var29_hace2', float),  # importance: 0.0
    # Feature('ind_var31_0', float),  # importance: 0.0
    # Feature('ind_var44', float),  # importance: 0.0
    # Feature('num_meses_var13_largo_ult3', float),  # importance: 0.0
    # Feature('delta_imp_trasp_var17_in_1y3', float),  # importance: 0.0
    # Feature('num_trasp_var17_in_ult1', float),  # importance: 0.0
    # Feature('num_var14_0', float),  # importance: 0.0
    # Feature('saldo_var31', float),  # importance: 0.0
    # Feature('saldo_medio_var17_hace2', float),  # importance: 0.0
    # Feature('num_trasp_var17_out_ult1', float),  # importance: 0.0
    # Feature('ind_var13_largo_0', float),  # importance: 0.0
    # Feature('saldo_medio_var13_medio_hace3', float),  # importance: 0.0
    # Feature('num_meses_var44_ult3', float),  # importance: 0.0
    # Feature('ind_var20_0', float),  # importance: 0.0
    # Feature('ind_var1_0', float),  # importance: 0.0
    # Feature('saldo_var34', float),  # importance: 0.0
    # Feature('ind_var13_largo', float),  # importance: 0.0
    # Feature('ind_var29', float),  # importance: 0.0
    # Feature('saldo_medio_var29_ult1', float),  # importance: 0.0
    # Feature('delta_num_trasp_var17_out_1y3', float),  # importance: 0.0
    # Feature('num_meses_var17_ult3', float),  # importance: 0.0
    # Feature('num_var13_corto_0', float),  # importance: 0.0
    # Feature('saldo_medio_var29_ult3', float),  # importance: 0.0
    # Feature('num_var32', float),  # importance: 0.0
    # Feature('imp_amort_var18_ult1', float),  # importance: 0.0
    # Feature('delta_imp_aport_var17_1y3', float),  # importance: 0.0
    # Feature('num_var6', float),  # importance: 0.0
    # Feature('ind_var1', float),  # importance: 0.0
    # Feature('num_var42', float),  # importance: 0.0
    # Feature('imp_amort_var34_hace3', float),  # importance: 0.0
    # Feature('num_var27', float),  # importance: 0.0
    # Feature('num_trasp_var17_in_hace3', float),  # importance: 0.0
    # Feature('num_var14', float),  # importance: 0.0
    # Feature('delta_num_reemb_var17_1y3', float),  # importance: 0.0
    # Feature('num_meses_var29_ult3', float),  # importance: 0.0
    # Feature('num_var8', float),  # importance: 0.0
    # Feature('num_reemb_var17_hace3', float),  # importance: 0.0
    # Feature('delta_num_aport_var13_1y3', float),  # importance: 0.0
    # Feature('imp_trasp_var17_out_ult1', float),  # importance: 0.0
    # Feature('imp_reemb_var33_ult1', float),  # importance: 0.0
    # Feature('saldo_var17', float),  # importance: 0.0
    # Feature('delta_imp_reemb_var17_1y3', float),  # importance: 0.0
    # Feature('imp_op_var40_comer_ult3', float),  # importance: 0.0
    # Feature('ind_var7_recib_ult1', float),  # importance: 0.0
    # Feature('imp_trasp_var17_out_hace3', float),  # importance: 0.0
    # Feature('num_var1_0', float),  # importance: 0.0
    # Feature('saldo_medio_var33_ult3', float),  # importance: 0.0
    # Feature('ind_var40_0', float),  # importance: 0.0
    # Feature('ind_var2', float),  # importance: 0.0
    # Feature('ind_var2_0', float),  # importance: 0.0
    # Feature('saldo_medio_var13_largo_hace2', float),  # importance: 0.0
    # Feature('ind_var32', float),  # importance: 0.0
    # Feature('saldo_var29', float),  # importance: 0.0
    # Feature('delta_imp_reemb_var33_1y3', float),  # importance: 0.0
    # Feature('imp_compra_var44_hace3', float),  # importance: 0.0
    # Feature('saldo_var1', float),  # importance: 0.0
    # Feature('saldo_var6', float),  # importance: 0.0
    # Feature('num_op_var40_efect_ult3', float),  # importance: 0.0
    # Feature('ind_var25_0', float),  # importance: 0.0
    # Feature('ind_var18_0', float),  # importance: 0.0
    # Feature('num_meses_var13_medio_ult3', float),  # importance: 0.0
    # Feature('num_op_var40_ult1', float),  # importance: 0.0
    # Feature('imp_reemb_var13_hace3', float),  # importance: 0.0
    # Feature('imp_aport_var17_ult1', float),  # importance: 0.0
    # Feature('saldo_medio_var44_hace3', float),  # importance: 0.0
    # Feature('num_var13_corto', float),  # importance: 0.0
    # Feature('delta_num_aport_var17_1y3', float),  # importance: 0.0
],
    documentation='https://www.kaggle.com/competitions/santander-customer-satisfaction/data'
)

AMEX_DEFAULT_FEATURES = FeatureList(features=[

    Feature('target', int, is_target=True, name_extended='default event within 120 days'),
    Feature('P_2', float, name_extended='Payment 2'),  # importance: 0.0782
    Feature('B_9', float, name_extended='Balance 9'),  # importance: 0.0652
    Feature('D_51', float, name_extended='Delinquency 51'),  # importance: 0.0564
    Feature('B_1', float, name_extended='Balance 1'),  # importance: 0.0448
    Feature('D_45', float, name_extended='Delinquency 45'),  # importance: 0.0355
    Feature('D_61', float, name_extended='Delinquency 61'),  # importance: 0.0335
    Feature('D_75', float, name_extended='Delinquency 75'),  # importance: 0.0321
    Feature('D_62', float, name_extended='Delinquency 62'),  # importance: 0.0307
    Feature('R_27', float, name_extended='Risk 27'),  # importance: 0.0303
    Feature('D_50', float, name_extended='Delinquency 50'),  # importance: 0.0255
    Feature('S_3', float, name_extended='Spend 3'),  # importance: 0.0241
    Feature('D_44', float, name_extended='Delinquency 44'),  # importance: 0.0236
    Feature('B_3', float, name_extended='Balance 3'),  # importance: 0.0221
    Feature('B_22', float, name_extended='Balance 22'),  # importance: 0.0209
    Feature('D_56', float, name_extended='Delinquency 56'),  # importance: 0.02
    Feature('D_79', float, name_extended='Delinquency 79'),  # importance: 0.0198
    Feature('D_48', float, name_extended='Delinquency 48'),  # importance: 0.0188
    Feature('B_38', float, name_extended='Balance 38'),  # importance: 0.0185
    Feature('D_64', cat_dtype, name_extended='Delinquency 64'),  # importance: 0.0178
    Feature('D_133', float, name_extended='Delinquency 133'),  # importance: 0.0158
    Feature('D_43', float, name_extended='Delinquency 43'),  # importance: 0.0158
    Feature('B_8', float, name_extended='Balance 8'),  # importance: 0.0157
    Feature('B_10', float, name_extended='Balance 10'),  # importance: 0.0151
    Feature('S_5', float, name_extended='Spend 5'),  # importance: 0.0148
    Feature('B_7', float, name_extended='Balance 7'),  # importance: 0.0147
    Feature('B_2', float, name_extended='Balance 2'),  # importance: 0.0138
    Feature('S_15', float, name_extended='Spend 15'),  # importance: 0.0126
    Feature('D_112', float, name_extended='Delinquency 112'),  # importance: 0.0123
    Feature('D_52', float, name_extended='Delinquency 52'),  # importance: 0.0122
    Feature('D_117', float, name_extended='Delinquency 117'),  # importance: 0.0121
    Feature('R_3', float, name_extended='Risk 3'),  # importance: 0.0118
    Feature('D_77', float, name_extended='Delinquency 77'),  # importance: 0.0111
    Feature('D_63', cat_dtype, name_extended='Delinquency 63'),  # importance: 0.0111
    ##################################################
    ##################################################
    # Feature('B_4', float, name_extended='Balance 4'),  # importance: 0.0109
    # Feature('customer_ID', cat_dtype),  # importance: 0.0101
    # Feature('D_121', float, name_extended='Delinquency 121'),  # importance: 0.0098
    # Feature('D_47', float, name_extended='Delinquency 47'),  # importance: 0.0097
    # Feature('B_5', float, name_extended='Balance 5'),  # importance: 0.009
    # Feature('S_7', float, name_extended='Spend 7'),  # importance: 0.0085
    # Feature('B_6', float, name_extended='Balance 6'),  # importance: 0.0083
    # Feature('D_41', float, name_extended='Delinquency 41'),  # importance: 0.0082
    # Feature('S_11', float, name_extended='Spend 11'),  # importance: 0.0078
    # Feature('D_131', float, name_extended='Delinquency 131'),  # importance: 0.0073
    # Feature('S_26', float, name_extended='Spend 26'),  # importance: 0.007
    # Feature('D_128', float, name_extended='Delinquency 128'),  # importance: 0.0069
    # Feature('D_46', float, name_extended='Delinquency 46'),  # importance: 0.0065
    # Feature('S_9', float, name_extended='Spend 9'),  # importance: 0.0064
    # Feature('S_2', cat_dtype, name_extended='Spend 2'),  # importance: 0.0063
    # Feature('D_122', float, name_extended='Delinquency 122'),  # importance: 0.0061
    # Feature('S_23', float, name_extended='Spend 23'),  # importance: 0.0059
    # Feature('S_24', float, name_extended='Spend 24'),  # importance: 0.0058
    # Feature('D_39', float, name_extended='Delinquency 39'),  # importance: 0.0055
    # Feature('D_119', float, name_extended='Delinquency 119'),  # importance: 0.0054
    # Feature('D_54', float, name_extended='Delinquency 54'),  # importance: 0.0047
    # Feature('B_24', float, name_extended='Balance 24'),  # importance: 0.0046
    # Feature('D_70', float, name_extended='Delinquency 70'),  # importance: 0.0043
    # Feature('B_36', float, name_extended='Balance 36'),  # importance: 0.004
    # Feature('R_1', float, name_extended='Risk 1'),  # importance: 0.0036
    # Feature('B_19', float, name_extended='Balance 19'),  # importance: 0.0036
    # Feature('D_144', float, name_extended='Delinquency 144'),  # importance: 0.0031
    # Feature('D_53', float, name_extended='Delinquency 53'),  # importance: 0.0029
    # Feature('B_21', float, name_extended='Balance 21'),  # importance: 0.0027
    # Feature('S_22', float, name_extended='Spend 22'),  # importance: 0.0023
    # Feature('B_18', float, name_extended='Balance 18'),  # importance: 0.0016
    # Feature('B_11', float, name_extended='Balance 11'),  # importance: 0.0016
    # Feature('P_3', float, name_extended='Payment 3'),  # importance: 0.0009
    # Feature('D_59', float, name_extended='Delinquency 59'),  # importance: 0.0003
    # Feature('R_5', float, name_extended='Risk 5'),  # importance: 0.0003
    # Feature('B_37', float, name_extended='Balance 37'),  # importance: 0.0003
    # Feature('B_13', float, name_extended='Balance 13'),  # importance: 0.0002
    # Feature('B_25', float, name_extended='Balance 25'),  # importance: 0.0001
    # Feature('B_20', float, name_extended='Balance 20'),  # importance: 0.0001
    # Feature('B_15', float, name_extended='Balance 15'),  # importance: 0.0001
    # Feature('D_124', float, name_extended='Delinquency 124'),  # importance: 0.0001
    # Feature('B_17', float, name_extended='Balance 17'),  # importance: 0.0001
    # Feature('B_23', float, name_extended='Balance 23'),  # importance: 0.0001
    # Feature('S_19', float, name_extended='Spend 19'),  # importance: 0.0
    # Feature('D_58', float, name_extended='Delinquency 58'),  # importance: 0.0
    # Feature('S_17', float, name_extended='Spend 17'),  # importance: 0.0
    # Feature('D_105', float, name_extended='Delinquency 105'),  # importance: 0.0
    # Feature('B_33', float, name_extended='Balance 33'),  # importance: 0.0
    # Feature('D_107', float, name_extended='Delinquency 107'),  # importance: 0.0
    # Feature('D_82', float, name_extended='Delinquency 82'),  # importance: 0.0
    # Feature('D_89', float, name_extended='Delinquency 89'),  # importance: 0.0
    # Feature('D_123', float, name_extended='Delinquency 123'),  # importance: 0.0
    # Feature('D_103', float, name_extended='Delinquency 103'),  # importance: 0.0
    # Feature('R_16', float, name_extended='Risk 16'),  # importance: 0.0
    # Feature('S_16', float, name_extended='Spend 16'),  # importance: 0.0
    # Feature('D_55', float, name_extended='Delinquency 55'),  # importance: 0.0
    # Feature('D_84', float, name_extended='Delinquency 84'),  # importance: 0.0
    # Feature('D_113', float, name_extended='Delinquency 113'),  # importance: 0.0
    # Feature('D_78', float, name_extended='Delinquency 78'),  # importance: 0.0
    # Feature('B_26', float, name_extended='Balance 26'),  # importance: 0.0
    # Feature('R_28', float, name_extended='Risk 28'),  # importance: 0.0
    # Feature('D_127', float, name_extended='Delinquency 127'),  # importance: 0.0
    # Feature('D_102', float, name_extended='Delinquency 102'),  # importance: 0.0
    # Feature('D_104', float, name_extended='Delinquency 104'),  # importance: 0.0
    # Feature('B_40', float, name_extended='Balance 40'),  # importance: 0.0
    # Feature('R_15', float, name_extended='Risk 15'),  # importance: 0.0
    # Feature('R_10', float, name_extended='Risk 10'),  # importance: 0.0
    # Feature('D_143', float, name_extended='Delinquency 143'),  # importance: 0.0
    # Feature('R_7', float, name_extended='Risk 7'),  # importance: 0.0
    # Feature('R_4', float, name_extended='Risk 4'),  # importance: 0.0
    # Feature('D_92', float, name_extended='Delinquency 92'),  # importance: 0.0
    # Feature('B_28', float, name_extended='Balance 28'),  # importance: 0.0
    # Feature('D_65', float, name_extended='Delinquency 65'),  # importance: 0.0
    # Feature('R_2', float, name_extended='Risk 2'),  # importance: 0.0
    # Feature('D_140', float, name_extended='Delinquency 140'),  # importance: 0.0
    # Feature('S_25', float, name_extended='Spend 25'),  # importance: 0.0
    # Feature('R_6', float, name_extended='Risk 6'),  # importance: 0.0
    # Feature('P_4', float, name_extended='Payment 4'),  # importance: 0.0
    # Feature('B_14', float, name_extended='Balance 14'),  # importance: 0.0
    # Feature('D_60', float, name_extended='Delinquency 60'),  # importance: 0.0
    # Feature('R_20', float, name_extended='Risk 20'),  # importance: 0.0
    # Feature('S_6', float, name_extended='Spend 6'),  # importance: 0.0
    # Feature('S_27', float, name_extended='Spend 27'),  # importance: 0.0
    # Feature('S_8', float, name_extended='Spend 8'),  # importance: 0.0
    # Feature('D_74', float, name_extended='Delinquency 74'),  # importance: 0.0
    # Feature('D_83', float, name_extended='Delinquency 83'),  # importance: 0.0
    # Feature('S_12', float, name_extended='Spend 12'),  # importance: 0.0
    # Feature('R_11', float, name_extended='Risk 11'),  # importance: 0.0
    # Feature('R_17', float, name_extended='Risk 17'),  # importance: 0.0
    # Feature('D_81', float, name_extended='Delinquency 81'),  # importance: 0.0
    # Feature('S_13', float, name_extended='Spend 13'),  # importance: 0.0
    # Feature('B_41', float, name_extended='Balance 41'),  # importance: 0.0
    # Feature('D_72', float, name_extended='Delinquency 72'),  # importance: 0.0
    # Feature('B_16', float, name_extended='Balance 16'),  # importance: 0.0
    # Feature('B_32', float, name_extended='Balance 32'),  # importance: 0.0
    # Feature('D_141', float, name_extended='Delinquency 141'),  # importance: 0.0
    # Feature('B_12', float, name_extended='Balance 12'),  # importance: 0.0
    # Feature('D_80', float, name_extended='Delinquency 80'),  # importance: 0.0
    # Feature('R_19', float, name_extended='Risk 19'),  # importance: 0.0
    # Feature('D_68', float, name_extended='Delinquency 68'),  # importance: 0.0
    # Feature('B_30', float, name_extended='Balance 30'),  # importance: 0.0
    # Feature('D_69', float, name_extended='Delinquency 69'),  # importance: 0.0
    # Feature('D_94', float, name_extended='Delinquency 94'),  # importance: 0.0
    # Feature('D_130', float, name_extended='Delinquency 130'),  # importance: 0.0
    # Feature('B_27', float, name_extended='Balance 27'),  # importance: 0.0
    # Feature('D_96', float, name_extended='Delinquency 96'),  # importance: 0.0
    # Feature('D_115', float, name_extended='Delinquency 115'),  # importance: 0.0
    # Feature('B_31', float, name_extended='Balance 31'),  # importance: 0.0
    # Feature('D_125', float, name_extended='Delinquency 125'),  # importance: 0.0
    # Feature('D_139', float, name_extended='Delinquency 139'),  # importance: 0.0
    # Feature('D_145', float, name_extended='Delinquency 145'),  # importance: 0.0
    # Feature('D_116', float, name_extended='Delinquency 116'),  # importance: 0.0
    # Feature('D_120', float, name_extended='Delinquency 120'),  # importance: 0.0
    # Feature('S_20', float, name_extended='Spend 20'),  # importance: 0.0
    # Feature('D_126', float, name_extended='Delinquency 126'),  # importance: 0.0
    # Feature('R_18', float, name_extended='Risk 18'),  # importance: 0.0
    # Feature('D_109', float, name_extended='Delinquency 109'),  # importance: 0.0
    # Feature('D_129', float, name_extended='Delinquency 129'),  # importance: 0.0
    # Feature('R_13', float, name_extended='Risk 13'),  # importance: 0.0
    # Feature('D_86', float, name_extended='Delinquency 86'),  # importance: 0.0
    # Feature('R_22', float, name_extended='Risk 22'),  # importance: 0.0
    # Feature('R_21', float, name_extended='Risk 21'),  # importance: 0.0
    # Feature('R_24', float, name_extended='Risk 24'),  # importance: 0.0
    # Feature('R_8', float, name_extended='Risk 8'),  # importance: 0.0
    # Feature('R_14', float, name_extended='Risk 14'),  # importance: 0.0
    # Feature('D_91', float, name_extended='Delinquency 91'),  # importance: 0.0
    # Feature('R_25', float, name_extended='Risk 25'),  # importance: 0.0
    # Feature('D_114', float, name_extended='Delinquency 114'),  # importance: 0.0
    # Feature('D_118', float, name_extended='Delinquency 118'),  # importance: 0.0
    # Feature('D_93', float, name_extended='Delinquency 93'),  # importance: 0.0
    # Feature('R_12', float, name_extended='Risk 12'),  # importance: 0.0
    # Feature('D_71', float, name_extended='Delinquency 71'),  # importance: 0.0
    # Feature('R_23', float, name_extended='Risk 23'),  # importance: 0.0
    # Feature('S_18', float),  # importance: 0.0
],
    documentation='https://www.kaggle.com/competitions/amex-default-prediction/data')

AD_FRAUD_FEATURES = FeatureList(features=[
    Feature('ip', int, name_extended='ip address'),
    Feature('app', int, name_extended='app id'),
    Feature('device', int, name_extended='device type id of user mobile phone'),
    Feature('os', int),
    Feature('channel', int, name_extended='channel id of mobile ad publisher'),
    Feature('click_time', cat_dtype, name_extended='UTC timestamp of click'),
    # Feature('attributed_time', cat_dtype),
    Feature('is_attributed', int, is_target=True,
            name_extended='indicator for whether app was downloaded'),
],
    documentation='https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/data')


def preprocess_otto(df: pd.DataFrame) -> pd.DataFrame:
    df['target'] = df['target'].apply(lambda x: x.replace('Class_', '')).astype(
        int)
    return df


def preprocess_walmart(df: pd.DataFrame) -> pd.DataFrame:
    df[WALMART_FEATURES.target] = df[WALMART_FEATURES.target].astype(int)
    return df
