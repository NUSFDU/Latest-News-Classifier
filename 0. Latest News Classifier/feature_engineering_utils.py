from collections import defaultdict
# fields_2_codes = defaultdict(dict)
# fields_decodes = defaultdict(dict)
# field_ = 'product_type'


def create_indicators_from_categorical_field(data_input, field_, fields_2_codes, fields_decodes):
    """
    data_input: pandas.core.frame.DataFrame
    field_: str
    fields_2_codes: defaultdict(dict)
    fields_decodes: defaultdict(dict)
    return: pandas.core.frame.DataFrame
    """
    for idx, raw_value in enumerate(dict(data_input[field_].value_counts()).keys()):
        fields_2_codes[field_][raw_value] = idx
        fields_decodes[field_][idx] = raw_value

    indicators = pd.DataFrame()
    indicator_list = []
    indicator_column_name = 'is_' + field_.lower() + '__nan'
    indicators[indicator_column_name] = (data_input[field_].isna()).astype('int')
    indicator_list.append(indicator_column_name)

    for indicator_value_ in fields_2_codes[field_].keys():
        indicator_column_name = 'is_' + field_ + '__' + indicator_value_
        indicator_column_name = indicator_column_name.lower()
        indicators[indicator_column_name] = (data_input[field_] == indicator_value_).astype('int')
        indicator_list.append(indicator_column_name)
    return indicators


def categorical_chi2_feature_test(features_, target_column_, pvalue_threshold_=0.05, print_length=20):
    """
    features_: pandas.core.frame.DataFrame
    target_column_: pandas.core.series.Series
    print_length: int
    return: pandas.core.frame.DataFrame
    """
    from sklearn.feature_selection import chi2
    import numpy as np
    significant_feature_ = set()
    for label_ in sorted(target_column_.unique()):
        features_chi2 = chi2(features_, target_column_ == label_)
        indices = np.argsort(features_chi2[0])
        feature_names = features_.columns[indices]
        significances = features_chi2[1][indices]
        for pv, fn in zip(significances, feature_names):
            if pv < pvalue_threshold_:
                significant_feature_.add(fn)
        print("# '{}' LABEL:".format(label_))
        print(". Most correlated features:\n. ")
        for pv, fn in zip(significances[-print_length:], feature_names[-print_length:]):
            print("p-value {:5.4f}  {}".format(pv, fn))
        print("")
    return significant_feature_
