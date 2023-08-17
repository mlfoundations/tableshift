"""
CatBoost quality benchmarks; adapted from
https://github.com/catboost/benchmarks/tree/master/quality_benchmarks

For more information on datasets and access TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift
"""
import re

import pandas as pd
from pandas import DataFrame

from tableshift.core.features import Feature, FeatureList, cat_dtype

AMAZON_FEATURES = FeatureList(features=[
    Feature('ACTION', float,
            "ACTION is 1 if the resource was approved, 0 if the resource was not",
            name_extended="access to resource was approved",
            is_target=True),
    Feature('RESOURCE', int, "An ID for each resource",
            name_extended="resource ID"),
    Feature('MGR_ID', int,
            "The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time",
            name_extended="manager ID"),
    Feature('ROLE_ROLLUP_1', int,
            "Company role grouping category id 1 (e.g. US Engineering)",
            name_extended="company role grouping category 1"),
    Feature('ROLE_ROLLUP_2', int,
            "Company role grouping category id 2 (e.g. US Retail)",
            name_extended="company role grouping category 2"),
    Feature('ROLE_DEPTNAME', int,
            'Company role department description (e.g. Retail)',
            name_extended='company role department description'),
    Feature('ROLE_TITLE', int,
            'Company role business title description (e.g. Senior Engineering Retail Manager)',
            name_extended='company role business title description'),
    Feature('ROLE_FAMILY_DESC', int,
            'Company role family extended description (e.g. Retail Manager, Software Engineering)',
            name_extended='company role family extended description'),
    Feature('ROLE_FAMILY', int,
            'Company role family description (e.g. Retail Manager)',
            name_extended='company role family description'),
    Feature('ROLE_CODE', int,
            'Company role code; this code is unique to each role (e.g. Manager)',
            name_extended='company role code'),
], documentation="https://www.kaggle.com/c/amazon-employee-access-challenge")

APPETENCY_FEATURES = FeatureList(features=[
    Feature('label', float, name_extended='class label', is_target=True),
    Feature('Var202', cat_dtype),  # importance: 0.0881
    Feature('Var220', cat_dtype),  # importance: 0.0622
    Feature('Var218', cat_dtype),  # importance: 0.0532
    Feature('Var198', cat_dtype),  # importance: 0.0449
    Feature('Var214', cat_dtype),  # importance: 0.04
    Feature('Var192', cat_dtype),  # importance: 0.0365
    Feature('Var199', cat_dtype),  # importance: 0.0359
    Feature('Var217', cat_dtype),  # importance: 0.0319
    Feature('Var126', float),  # importance: 0.027
    Feature('Var222', cat_dtype),  # importance: 0.0234
    Feature('Var216', cat_dtype),  # importance: 0.0174
    Feature('Var78', float),  # importance: 0.0168
    Feature('Var126_imputed', float),  # importance: 0.014
    Feature('Var212', cat_dtype),  # importance: 0.0138
    Feature('Var197', cat_dtype),  # importance: 0.0137
    Feature('Var204', cat_dtype),  # importance: 0.0128
    Feature('Var7', float),  # importance: 0.0121
    Feature('Var191', cat_dtype),  # importance: 0.0119
    Feature('Var211', cat_dtype),  # importance: 0.0118
    Feature('Var144_imputed', float),  # importance: 0.0111
    Feature('Var189', float),  # importance: 0.0109
    Feature('Var194', cat_dtype),  # importance: 0.0108
    Feature('Var83', float),  # importance: 0.0105
    Feature('Var228', cat_dtype),  # importance: 0.01
    Feature('Var229', cat_dtype),  # importance: 0.0098
    Feature('Var205', cat_dtype),  # importance: 0.0097
    Feature('Var206', cat_dtype),  # importance: 0.0097
    Feature('Var38', float),  # importance: 0.0096
    Feature('Var207', cat_dtype),  # importance: 0.0092
    Feature('Var223', cat_dtype),  # importance: 0.009
    Feature('Var24', float),  # importance: 0.009
    Feature('Var225', cat_dtype),  # importance: 0.0089
    Feature('Var125', float),  # importance: 0.0088
    ##################################################
    ##################################################
    # Feature('Var173', float),  # importance: 0.0088
    # Feature('Var132', float),  # importance: 0.0087
    # Feature('Var144', float),  # importance: 0.0085
    # Feature('Var72', float),  # importance: 0.0084
    # Feature('Var65', float),  # importance: 0.0084
    # Feature('Var109', float),  # importance: 0.0083
    # Feature('Var149', float),  # importance: 0.0082
    # Feature('Var134', float),  # importance: 0.0081
    # Feature('Var81', float),  # importance: 0.0081
    # Feature('Var133', float),  # importance: 0.0078
    # Feature('Var227', cat_dtype),  # importance: 0.0077
    # Feature('Var123', float),  # importance: 0.0076
    # Feature('Var153', float),  # importance: 0.0076
    # Feature('Var73', float),  # importance: 0.0075
    # Feature('Var112', float),  # importance: 0.0074
    # Feature('Var160', float),  # importance: 0.0074
    # Feature('Var21', float),  # importance: 0.0073
    # Feature('Var57', float),  # importance: 0.0072
    # Feature('Var6', float),  # importance: 0.0072
    # Feature('Var35', float),  # importance: 0.0071
    # Feature('Var94', float),  # importance: 0.0071
    # Feature('Var226', cat_dtype),  # importance: 0.007
    # Feature('Var113', float),  # importance: 0.0068
    # Feature('Var140', float),  # importance: 0.0068
    # Feature('Var28', float),  # importance: 0.0066
    # Feature('Var13', float),  # importance: 0.0066
    # Feature('Var76', float),  # importance: 0.0066
    # Feature('Var219', cat_dtype),  # importance: 0.0063
    # Feature('Var200', cat_dtype),  # importance: 0.0061
    # Feature('Var85', float),  # importance: 0.006
    # Feature('Var196', cat_dtype),  # importance: 0.006
    # Feature('Var74', float),  # importance: 0.0057
    # Feature('Var25', float),  # importance: 0.0056
    # Feature('Var163', float),  # importance: 0.0055
    # Feature('Var221', cat_dtype),  # importance: 0.0054
    # Feature('Var193', cat_dtype),  # importance: 0.0052
    # Feature('Var119', float),  # importance: 0.0051
    # Feature('Var203', cat_dtype),  # importance: 0.0039
    # Feature('Var22', float),  # importance: 0.0039
    # Feature('Var51', float),  # importance: 0.0037
    # Feature('Var163_imputed', float),  # importance: 0.0036
    # Feature('Var224', cat_dtype),  # importance: 0.0036
    # Feature('Var213', cat_dtype),  # importance: 0.0036
    # Feature('Var189_imputed', float),  # importance: 0.0035
    # Feature('Var208', cat_dtype),  # importance: 0.0034
    # Feature('Var143', float),  # importance: 0.0019
    # Feature('Var210', cat_dtype),  # importance: 0.0017
    # Feature('Var201', cat_dtype),  # importance: 0.0009
    # Feature('Var181', float),  # importance: 0.0
    # Feature('Var143_imputed', float),  # importance: 0.0
    # Feature('Var119_imputed', float),  # importance: 0.0
    # Feature('Var76_imputed', float),  # importance: 0.0
    # Feature('Var149_imputed', float),  # importance: 0.0
    # Feature('Var85_imputed', float),  # importance: 0.0
    # Feature('Var7_imputed', float),  # importance: 0.0
    # Feature('Var160_imputed', float),  # importance: 0.0
    # Feature('Var72_imputed', float),  # importance: 0.0
    # Feature('Var78_imputed', float),  # importance: 0.0
    # Feature('Var153_imputed', float),  # importance: 0.0
    # Feature('Var173_imputed', float),  # importance: 0.0
    # Feature('Var112_imputed', float),  # importance: 0.0
    # Feature('Var44', float),  # importance: 0.0
    # Feature('Var25_imputed', float),  # importance: 0.0
    # Feature('Var133_imputed', float),  # importance: 0.0
    # Feature('Var132_imputed', float),  # importance: 0.0
    # Feature('Var215', cat_dtype),  # importance: 0.0
    # Feature('Var35_imputed', float),  # importance: 0.0
    # Feature('Var6_imputed', float),  # importance: 0.0
    # Feature('Var140_imputed', float),  # importance: 0.0
    # Feature('Var51_imputed', float),  # importance: 0.0
    # Feature('Var13_imputed', float),  # importance: 0.0
    # Feature('Var94_imputed', float),  # importance: 0.0
    # Feature('Var125_imputed', float),  # importance: 0.0
    # Feature('Var181_imputed', float),  # importance: 0.0
    # Feature('Var28_imputed', float),  # importance: 0.0
    # Feature('Var22_imputed', float),  # importance: 0.0
    # Feature('Var83_imputed', float),  # importance: 0.0
    # Feature('Var195', cat_dtype),  # importance: 0.0
    # Feature('Var134_imputed', float),  # importance: 0.0
    # Feature('Var65_imputed', float),  # importance: 0.0
    # Feature('Var24_imputed', float),  # importance: 0.0
    # Feature('Var21_imputed', float),  # importance: 0.0
    # Feature('Var38_imputed', float),  # importance: 0.0
    # Feature('Var74_imputed', float),  # importance: 0.0
    # Feature('Var44_imputed', float),  # importance: 0.0
    # Feature('Var109_imputed', float),  # importance: 0.0
    # Feature('Var81_imputed', float),  # importance: 0.0
    # Feature('Var123_imputed', float),  # importance: 0.0
], documentation='https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data ,'
                 'https://medium.com/@kushaldps1996/customer-relationship-prediction-kdd-cup-2009-6b57d08ffb0')

CHURN_FEATURES = FeatureList(features=[
    Feature('label', float, name_extended='class label', is_target=True),
    Feature('Var202', cat_dtype),  # importance: 0.1061
    Feature('Var222', cat_dtype),  # importance: 0.0707
    Feature('Var220', cat_dtype),  # importance: 0.0699
    Feature('Var217', cat_dtype),  # importance: 0.0606
    Feature('Var199', cat_dtype),  # importance: 0.0393
    Feature('Var126', float),  # importance: 0.0371
    Feature('Var192', cat_dtype),  # importance: 0.0319
    Feature('Var51_imputed', float),  # importance: 0.0208
    Feature('Var216', cat_dtype),  # importance: 0.0207
    Feature('Var126_imputed', float),  # importance: 0.0194
    Feature('Var198', cat_dtype),  # importance: 0.0181
    Feature('Var197', cat_dtype),  # importance: 0.0166
    Feature('Var204', cat_dtype),  # importance: 0.0163
    Feature('Var74', float),  # importance: 0.0132
    Feature('Var78', float),  # importance: 0.012
    Feature('Var7_imputed', float),  # importance: 0.0115
    Feature('Var207', cat_dtype),  # importance: 0.0107
    Feature('Var206', cat_dtype),  # importance: 0.0107
    Feature('Var210', cat_dtype),  # importance: 0.0107
    Feature('Var194', cat_dtype),  # importance: 0.0105
    Feature('Var228', cat_dtype),  # importance: 0.0096
    Feature('Var226', cat_dtype),  # importance: 0.0095
    Feature('Var218', cat_dtype),  # importance: 0.0093
    Feature('Var181', float),  # importance: 0.009
    Feature('Var212', cat_dtype),  # importance: 0.0089
    Feature('Var73', float),  # importance: 0.0089
    Feature('Var44', float),  # importance: 0.0088
    Feature('Var109', float),  # importance: 0.0084
    Feature('Var13', float),  # importance: 0.0083
    Feature('Var113', float),  # importance: 0.008
    Feature('Var229', cat_dtype),  # importance: 0.0079
    Feature('Var227', cat_dtype),  # importance: 0.0079
    Feature('Var140', float),  # importance: 0.0078
    ##################################################
    ##################################################
    # Feature('Var225', cat_dtype),  # importance: 0.0078
    # Feature('Var28', float),  # importance: 0.0076
    # Feature('Var189', float),  # importance: 0.0076
    # Feature('Var195', cat_dtype),  # importance: 0.0076
    # Feature('Var72', float),  # importance: 0.0075
    # Feature('Var38', float),  # importance: 0.0073
    # Feature('Var65', float),  # importance: 0.0073
    # Feature('Var81', float),  # importance: 0.0073
    # Feature('Var25', float),  # importance: 0.0072
    # Feature('Var153', float),  # importance: 0.0072
    # Feature('Var125', float),  # importance: 0.0071
    # Feature('Var221', cat_dtype),  # importance: 0.0071
    # Feature('Var21', float),  # importance: 0.007
    # Feature('Var219', cat_dtype),  # importance: 0.007
    # Feature('Var134', float),  # importance: 0.0069
    # Feature('Var6', float),  # importance: 0.0069
    # Feature('Var205', cat_dtype),  # importance: 0.0068
    # Feature('Var57', float),  # importance: 0.0068
    # Feature('Var160', float),  # importance: 0.0067
    # Feature('Var123', float),  # importance: 0.0066
    # Feature('Var83', float),  # importance: 0.0066
    # Feature('Var76', float),  # importance: 0.0065
    # Feature('Var149', float),  # importance: 0.0065
    # Feature('Var133', float),  # importance: 0.0064
    # Feature('Var144', float),  # importance: 0.0064
    # Feature('Var112', float),  # importance: 0.0061
    # Feature('Var94', float),  # importance: 0.006
    # Feature('Var208', cat_dtype),  # importance: 0.006
    # Feature('Var132', float),  # importance: 0.0058
    # Feature('Var193', cat_dtype),  # importance: 0.0058
    # Feature('Var163', float),  # importance: 0.0058
    # Feature('Var223', cat_dtype),  # importance: 0.0057
    # Feature('Var24', float),  # importance: 0.0057
    # Feature('Var203', cat_dtype),  # importance: 0.0055
    # Feature('Var119', float),  # importance: 0.0055
    # Feature('Var7', float),  # importance: 0.0053
    # Feature('Var215', cat_dtype),  # importance: 0.0052
    # Feature('Var189_imputed', float),  # importance: 0.0049
    # Feature('Var85', float),  # importance: 0.0049
    # Feature('Var22', float),  # importance: 0.0039
    # Feature('Var224', cat_dtype),  # importance: 0.0036
    # Feature('Var211', cat_dtype),  # importance: 0.0032
    # Feature('Var51', float),  # importance: 0.0031
    # Feature('Var35', float),  # importance: 0.0028
    # Feature('Var213', cat_dtype),  # importance: 0.0027
    # Feature('Var200', cat_dtype),  # importance: 0.0024
    # Feature('Var163_imputed', float),  # importance: 0.0023
    # Feature('Var196', cat_dtype),  # importance: 0.002
    # Feature('Var191', cat_dtype),  # importance: 0.0009
    # Feature('Var78_imputed', float),  # importance: 0.0
    # Feature('Var35_imputed', float),  # importance: 0.0
    # Feature('Var149_imputed', float),  # importance: 0.0
    # Feature('Var173', float),  # importance: 0.0
    # Feature('Var153_imputed', float),  # importance: 0.0
    # Feature('Var160_imputed', float),  # importance: 0.0
    # Feature('Var201', cat_dtype),  # importance: 0.0
    # Feature('Var173_imputed', float),  # importance: 0.0
    # Feature('Var25_imputed', float),  # importance: 0.0
    # Feature('Var112_imputed', float),  # importance: 0.0
    # Feature('Var214', cat_dtype),  # importance: 0.0
    # Feature('Var38_imputed', float),  # importance: 0.0
    # Feature('Var83_imputed', float),  # importance: 0.0
    # Feature('Var144_imputed', float),  # importance: 0.0
    # Feature('Var143_imputed', float),  # importance: 0.0
    # Feature('Var72_imputed', float),  # importance: 0.0
    # Feature('Var119_imputed', float),  # importance: 0.0
    # Feature('Var125_imputed', float),  # importance: 0.0
    # Feature('Var133_imputed', float),  # importance: 0.0
    # Feature('Var22_imputed', float),  # importance: 0.0
    # Feature('Var123_imputed', float),  # importance: 0.0
    # Feature('Var109_imputed', float),  # importance: 0.0
    # Feature('Var140_imputed', float),  # importance: 0.0
    # Feature('Var13_imputed', float),  # importance: 0.0
    # Feature('Var94_imputed', float),  # importance: 0.0
    # Feature('Var21_imputed', float),  # importance: 0.0
    # Feature('Var181_imputed', float),  # importance: 0.0
    # Feature('Var28_imputed', float),  # importance: 0.0
    # Feature('Var81_imputed', float),  # importance: 0.0
    # Feature('Var132_imputed', float),  # importance: 0.0
    # Feature('Var6_imputed', float),  # importance: 0.0
    # Feature('Var65_imputed', float),  # importance: 0.0
    # Feature('Var24_imputed', float),  # importance: 0.0
    # Feature('Var143', float),  # importance: 0.0
    # Feature('Var74_imputed', float),  # importance: 0.0
    # Feature('Var44_imputed', float),  # importance: 0.0
    # Feature('Var85_imputed', float),  # importance: 0.0
    # Feature('Var134_imputed', float),  # importance: 0.0
    # Feature('Var76_imputed', float),  # importance: 0.0
], documentation='https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data ,'
                 'https://medium.com/@kushaldps1996/customer-relationship-prediction-kdd-cup-2009-6b57d08ffb0')

UPSELLING_FEATURES = FeatureList(features=[
    Feature('label', float, name_extended='class label', is_target=True),
    Feature('Var126', float),  # importance: 0.1205
    Feature('Var202', cat_dtype),  # importance: 0.0812
    Feature('Var198', cat_dtype),  # importance: 0.0687
    Feature('Var211', cat_dtype),  # importance: 0.0635
    Feature('Var222', cat_dtype),  # importance: 0.0623
    Feature('Var28', float),  # importance: 0.0424
    Feature('Var217', cat_dtype),  # importance: 0.0402
    Feature('Var192', cat_dtype),  # importance: 0.0293
    Feature('Var199', cat_dtype),  # importance: 0.0273
    Feature('Var216', cat_dtype),  # importance: 0.0176
    Feature('Var204', cat_dtype),  # importance: 0.0169
    Feature('Var197', cat_dtype),  # importance: 0.0157
    Feature('Var126_imputed', float),  # importance: 0.0141
    Feature('Var220', cat_dtype),  # importance: 0.0119
    Feature('Var22', float),  # importance: 0.0116
    Feature('Var227', cat_dtype),  # importance: 0.0102
    Feature('Var225', cat_dtype),  # importance: 0.0102
    Feature('Var65', float),  # importance: 0.0102
    Feature('Var206', cat_dtype),  # importance: 0.0095
    Feature('Var153', float),  # importance: 0.009
    Feature('Var212', cat_dtype),  # importance: 0.0084
    Feature('Var228', cat_dtype),  # importance: 0.0082
    Feature('Var207', cat_dtype),  # importance: 0.0079
    Feature('Var81', float),  # importance: 0.0078
    Feature('Var223', cat_dtype),  # importance: 0.0076
    Feature('Var160', float),  # importance: 0.0075
    Feature('Var218', cat_dtype),  # importance: 0.0074
    Feature('Var213', cat_dtype),  # importance: 0.0073
    Feature('Var226', cat_dtype),  # importance: 0.0071
    Feature('Var140', float),  # importance: 0.0071
    Feature('Var193', cat_dtype),  # importance: 0.0071
    Feature('Var78', float),  # importance: 0.007
    Feature('Var133', float),  # importance: 0.0068
    ##################################################
    ##################################################
    # Feature('Var85', float),  # importance: 0.0066
    # Feature('Var194', cat_dtype),  # importance: 0.0066
    # Feature('Var221', cat_dtype),  # importance: 0.0065
    # Feature('Var113', float),  # importance: 0.0065
    # Feature('Var119', float),  # importance: 0.0065
    # Feature('Var219', cat_dtype),  # importance: 0.0062
    # Feature('Var200', cat_dtype),  # importance: 0.0062
    # Feature('Var13', float),  # importance: 0.0061
    # Feature('Var6', float),  # importance: 0.0061
    # Feature('Var21', float),  # importance: 0.0061
    # Feature('Var163', float),  # importance: 0.0061
    # Feature('Var57', float),  # importance: 0.006
    # Feature('Var125', float),  # importance: 0.0059
    # Feature('Var205', cat_dtype),  # importance: 0.0058
    # Feature('Var38', float),  # importance: 0.0058
    # Feature('Var94', float),  # importance: 0.0058
    # Feature('Var24', float),  # importance: 0.0057
    # Feature('Var210', cat_dtype),  # importance: 0.0056
    # Feature('Var144', float),  # importance: 0.0056
    # Feature('Var134', float),  # importance: 0.0055
    # Feature('Var149', float),  # importance: 0.0054
    # Feature('Var83', float),  # importance: 0.0054
    # Feature('Var229', cat_dtype),  # importance: 0.0054
    # Feature('Var25', float),  # importance: 0.0053
    # Feature('Var123', float),  # importance: 0.0053
    # Feature('Var76', float),  # importance: 0.005
    # Feature('Var74', float),  # importance: 0.0049
    # Feature('Var109', float),  # importance: 0.0048
    # Feature('Var112', float),  # importance: 0.0048
    # Feature('Var189', float),  # importance: 0.0047
    # Feature('Var203', cat_dtype),  # importance: 0.0047
    # Feature('Var7', float),  # importance: 0.0046
    # Feature('Var208', cat_dtype),  # importance: 0.0045
    # Feature('Var191', cat_dtype),  # importance: 0.0045
    # Feature('Var73', float),  # importance: 0.0045
    # Feature('Var72', float),  # importance: 0.0045
    # Feature('Var132', float),  # importance: 0.0043
    # Feature('Var7_imputed', float),  # importance: 0.004
    # Feature('Var196', cat_dtype),  # importance: 0.0037
    # Feature('Var35', float),  # importance: 0.0034
    # Feature('Var195', cat_dtype),  # importance: 0.0027
    # Feature('Var173', float),  # importance: 0.0026
    # Feature('Var51', float),  # importance: 0.0026
    # Feature('Var189_imputed', float),  # importance: 0.0025
    # Feature('Var224', cat_dtype),  # importance: 0.002
    # Feature('Var163_imputed', float),  # importance: 0.0018
    # Feature('Var181', float),  # importance: 0.0014
    # Feature('Var83_imputed', float),  # importance: 0.0
    # Feature('Var119_imputed', float),  # importance: 0.0
    # Feature('Var215', cat_dtype),  # importance: 0.0
    # Feature('Var76_imputed', float),  # importance: 0.0
    # Feature('Var85_imputed', float),  # importance: 0.0
    # Feature('Var173_imputed', float),  # importance: 0.0
    # Feature('Var160_imputed', float),  # importance: 0.0
    # Feature('Var78_imputed', float),  # importance: 0.0
    # Feature('Var134_imputed', float),  # importance: 0.0
    # Feature('Var25_imputed', float),  # importance: 0.0
    # Feature('Var201', cat_dtype),  # importance: 0.0
    # Feature('Var214', cat_dtype),  # importance: 0.0
    # Feature('Var144_imputed', float),  # importance: 0.0
    # Feature('Var143_imputed', float),  # importance: 0.0
    # Feature('Var44', float),  # importance: 0.0
    # Feature('Var72_imputed', float),  # importance: 0.0
    # Feature('Var6_imputed', float),  # importance: 0.0
    # Feature('Var74_imputed', float),  # importance: 0.0
    # Feature('Var81_imputed', float),  # importance: 0.0
    # Feature('Var28_imputed', float),  # importance: 0.0
    # Feature('Var125_imputed', float),  # importance: 0.0
    # Feature('Var123_imputed', float),  # importance: 0.0
    # Feature('Var109_imputed', float),  # importance: 0.0
    # Feature('Var13_imputed', float),  # importance: 0.0
    # Feature('Var94_imputed', float),  # importance: 0.0
    # Feature('Var140_imputed', float),  # importance: 0.0
    # Feature('Var133_imputed', float),  # importance: 0.0
    # Feature('Var149_imputed', float),  # importance: 0.0
    # Feature('Var181_imputed', float),  # importance: 0.0
    # Feature('Var22_imputed', float),  # importance: 0.0
    # Feature('Var44_imputed', float),  # importance: 0.0
    # Feature('Var35_imputed', float),  # importance: 0.0
    # Feature('Var153_imputed', float),  # importance: 0.0
    # Feature('Var65_imputed', float),  # importance: 0.0
    # Feature('Var38_imputed', float),  # importance: 0.0
    # Feature('Var24_imputed', float),  # importance: 0.0
    # Feature('Var143', float),  # importance: 0.0
    # Feature('Var21_imputed', float),  # importance: 0.0
    # Feature('Var112_imputed', float),  # importance: 0.0
    # Feature('Var51_imputed', float),  # importance: 0.0
    # Feature('Var132_imputed', float),  # importance: 0.0
], documentation='https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data ,'
                 'https://medium.com/@kushaldps1996/customer-relationship-prediction-kdd-cup-2009-6b57d08ffb0')

CLICK_FEATURES = FeatureList(features=[
    Feature('click', float, is_target=True,
            name_extended='user clicked ad at least once'),
    Feature('impression', cat_dtype),
    Feature('url_hash', cat_dtype, name_extended='URL hash'),
    Feature('ad_id', cat_dtype, name_extended='ad ID'),
    Feature('advertiser_id', float, name_extended='advertiser ID'),
    Feature('depth', float,
            name_extended='number of ads impressed in this session'),
    Feature('position', cat_dtype,
            name_extended='order of this ad in the impression list'),
    Feature('query_id', cat_dtype, name_extended='query ID'),
    Feature('keyword_id', cat_dtype, name_extended='keyword ID'),
    Feature('title_id', cat_dtype, name_extended='title ID'),
    Feature('description_id', cat_dtype, name_extended='description ID'),
    Feature('user_id', float, name_extended='user ID'),
], documentation='https://www.kaggle.com/competitions/kddcup2012-track2/data ,'
                 'http://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-2')

KICK_FEATURES = FeatureList(features=[
    Feature('RefId', cat_dtype,
            name_extended='Unique (sequential) number assigned to vehicles'),
    Feature('IsBadBuy', int, is_target=True,
            name_extended='indicator for whether the kicked vehicle was an avoidable purchase'),
    Feature('PurchDate', cat_dtype,
            name_extended='date the vehicle was purchased at auction'),
    Feature('Auction', cat_dtype,
            name_extended="Auction provider at which the  vehicle was purchased"),
    Feature('VehYear', int, name_extended="manufacture year of the vehicle"),
    Feature('VehicleAge', int,
            name_extended="Years elapsed since the manufacture year"),
    Feature('Make', cat_dtype),
    Feature('Model', cat_dtype),
    Feature('Trim', cat_dtype, name_extended='Trim Level'),
    Feature('SubModel', cat_dtype),
    Feature('Color', cat_dtype),
    Feature('Transmission', cat_dtype,
            name_extended='Vehicle transmission type'),
    Feature('WheelTypeID', cat_dtype,
            name_extended='type id of the vehicle wheel'),
    Feature('WheelType', cat_dtype, name_extended='wheel type description'),
    Feature('VehOdo', int, name_extended='odometer reading'),
    Feature('Nationality', cat_dtype, name_extended="manufacturer's country"),
    Feature('Size', cat_dtype, name_extended='size category of the vehicle'),
    Feature('TopThreeAmericanName', cat_dtype,
            name_extended='manufacturer is one of the top three American manufacturers'),
    Feature('MMRAcquisitionAuctionAveragePrice', float,
            name_extended='average acquisition price at auction for this vehicle in average condition at time of purchase'),
    Feature('MMRAcquisitionAuctionCleanPrice', float,
            name_extended='average acquisition price at auction for this vehicle in the above average condition at time of purchase'),
    Feature('MMRAcquisitionRetailAveragePrice', float,
            name_extended='average retail price for this vehicle in average condition at time of purchase'),
    Feature('MMRAcquisitonRetailCleanPrice', float,
            name_extended='average retail price for this vehicle in above average condition at time of purchase'),
    Feature('MMRCurrentAuctionAveragePrice', float,
            name_extended='average acquisition price at auction for this vehicle in average condition as of current day'),
    Feature('MMRCurrentAuctionCleanPrice', float,
            name_extended='average acquisition price at auction for this vehicle in above average condition as of current day'),
    Feature('MMRCurrentRetailAveragePrice', float,
            name_extended='average retail price for this vehicle in average condition as of current day'),
    Feature('MMRCurrentRetailCleanPrice', float,
            name_extended='average retail price for this vehicle in above average condition as of current day'),
    Feature('PRIMEUNIT', cat_dtype,
            name_extended='vehicle would have a higher demand than a standard purchase'),
    Feature('AUCGUART', cat_dtype,
            name_extended='acquisition method of vehicle'),
    Feature('BYRNO', int,
            name_extended='level guarantee provided by auction for the vehicle'),
    Feature('VNZIP1', cat_dtype, 'ZIP code where the car was purchased'),
    Feature('VNST', cat_dtype,
            name_extended='State where the the car was purchased'),
    Feature('VehBCost', float,
            name_extended='acquisition cost paid for the vehicle at time of purchase'),
    Feature('IsOnlineSale', int,
            name_extended='vehicle was originally purchased online'),
    Feature('WarrantyCost', int,
            name_extended='Warranty price (with term=36 month and mileage=36K)'),
], documentation="https://www.kaggle.com/competitions/DontGetKicked/data")


def preprocess_kick(df: DataFrame) -> DataFrame:
    return df


def preprocess_click(data: DataFrame) -> DataFrame:
    categorical_features = {1, 2, 3, 6, 7, 8, 9, 10}

    def clean_string(s):
        return "v_" + re.sub('[^A-Za-z0-9]+', "_", str(s))

    for i in categorical_features:
        data[data.columns[i]] = data[data.columns[i]].apply(clean_string)

    data["click"] = data["click"].apply(lambda x: 1 if x != 0 else -1)

    return data


def preprocess_appetency(data: DataFrame) -> DataFrame:
    """Adapted from https://github.com/catboost/benchmarks/blob/master
    /quality_benchmarks/prepare_appetency_churn_upselling
    /prepare_appetency_churn_upselling.ipynb """

    # preparing categorical features

    categorical_features = {190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                            200, 201, 202, 203, 204, 205, 206, 207, 209, 210,
                            211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
                            221, 222, 223, 224, 225, 226, 227, 228}

    numeric_colnames = {column for i, column in enumerate(data.columns) if
                        i not in categorical_features}

    # Note: we do not need to explicitly cast categorical features to string;
    # these are already of dtype object.

    for i in categorical_features:
        data[data.columns[i]] = data[data.columns[i]].fillna("MISSING").apply(
            str).astype("category")

    # prepare numerical features

    # drop any numeric column that is >= 95% missing
    all_missing = data.columns[(pd.isnull(data).sum() >= 0.95 * len(data))]
    data.drop(columns=all_missing, inplace=True)
    numeric_colnames -= set(all_missing)

    columns_to_impute = []
    for column in numeric_colnames:
        if pd.isnull(data[column]).any():
            columns_to_impute.append(column)

    for column_name in columns_to_impute:
        data[column_name + "_imputed"] = pd.isnull(data[column_name]).astype(
            float)
        data[column_name].fillna(0, inplace=True)

    for column in numeric_colnames:
        data[column] = data[column].astype(float)

    return data
