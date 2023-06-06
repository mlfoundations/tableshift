import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

EXTERNAL_RISK_THRESHOLD = 63

HELOC_FEATURES = FeatureList(features=[
    Feature('RiskPerformance', int, "Paid as negotiated flag (12-36 Months). "
                                    "String of Good and Bad", is_target=True),
    Feature('ExternalRiskEstimateLow', int,
            "Consolidated version of risk markers",
            name_extended=f"Indicator for whether external risk <= {EXTERNAL_RISK_THRESHOLD}"),
    Feature('MSinceOldestTradeOpen', int,
            name_extended='Months Since Oldest Trade Open'),
    Feature('MSinceMostRecentTradeOpen', int,
            name_extended='Months Since Most Recent Trade Open'),
    Feature('AverageMInFile', int,
            name_extended='Average Months in File'),
    Feature('NumSatisfactoryTrades', int,
            name_extended='Number of Satisfactory Trades'),
    Feature('NumTrades60Ever2DerogPubRec', int,
            name_extended='Number of Trades 60+ Ever'),
    Feature('NumTrades90Ever2DerogPubRec', int,
            name_extended='Number of Trades 90+ Ever'),
    Feature('PercentTradesNeverDelq', int,
            name_extended='Percent of Trades Never Delinquent'),
    Feature('MSinceMostRecentDelq', int,
            name_extended='Months Since Most Recent Delinquency'),
    Feature('MaxDelq2PublicRecLast12M', cat_dtype,
            name_extended="Max Delinquent/Public Records Last 12 Months",
            value_mapping={
                -9: "unknown",
                0: "derogatory comment",
                1: "120+ days delinquent",
                2: "90 days delinquent",
                3: "60 days delinquent",
                4: "30 days delinquent",
                5: "unknown delinquency",
                6: "unknown delinquency",
                7: "current and never delinquent",
                9: "all other",
            }),
    Feature('MaxDelqEver', cat_dtype,
            "Max delinquency ever",
            value_mapping={
                -9: "all other",
                9: "all other",
                1: "No such value",
                2: "derogatory comment",
                3: "120+ days delinquent",
                4: "90 days delinquent",
                5: "60 days delinquent",
                6: "30 days delinquent",
                7: "unknown delinquency",
                8: "current and never delinquent"}),
    Feature('NumTotalTrades', int,
            name_extended='Number of Total Trades (total number of credit accounts)'),
    Feature('NumTradesOpeninLast12M', int,
            name_extended='Number of Trades Open in Last 12 Months'),
    Feature('PercentInstallTrades', int,
            name_extended='Percent Installment Trades'),
    Feature('MSinceMostRecentInqexcl7days', int,
            name_extended='Months Since Most Recent Inq excl 7 days'),
    Feature('NumInqLast6M', int,
            name_extended='Number of inquiries last 6 Months'),
    Feature('NumInqLast6Mexcl7days', int, """Number of Inq Last 6 Months excl 
    7days. Excluding the last 7 days removes inquiries that are likely due to 
    price comparision shopping.""",
            name_extended="Number of inquiries in last 6 months excluding last 7 days"),
    Feature('NetFractionRevolvingBurden', int,
            name_extended="Net Fraction Revolving Burden "
                          "(revolving balance divided by credit limit)"),
    Feature('NetFractionInstallBurden', int,
            name_extended="Net Fraction Installment Burden "
                          "(installment balance divided by original loan amount)"),
    Feature('NumRevolvingTradesWBalance', int,
            name_extended='Number of revolving trades with balance'),
    Feature('NumInstallTradesWBalance', int,
           name_extended= 'Number of installment trades with balance'),
    Feature('NumBank2NatlTradesWHighUtilization', int,
            name_extended='Number of bank/national trades with high utilization ratio'),
    Feature('PercentTradesWBalance', int,
            name_extended='Percent of trades with balance'),
], documentation="""Data dictionary .xslx file can be accessed after filling 
out the data agreement at https://community.fico.com/s/explainable-machine
-learning-challenge """)


def preprocess_heloc(df: pd.DataFrame) -> pd.DataFrame:
    # Transform target to integer
    target = HELOC_FEATURES.target
    df[target] = (df[target] == "Good").astype(int)

    df['ExternalRiskEstimateLow'] = (
            df['ExternalRiskEstimate'] <= EXTERNAL_RISK_THRESHOLD)
    df.drop(columns=['ExternalRiskEstimate'], inplace=True)
    return df
