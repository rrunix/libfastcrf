import pandas as pd
import numpy as np
from fastcrf import datasets
import os
import pickle
from sklearn.preprocessing import StandardScaler

DATASET_CACHE_DIR = '../datasets_cache/'

dataset_list = {}


def dataset(f):
    dataset_name = f.__name__.replace('load_', '')

    def call(**kwargs):
        if 'cache' in kwargs:
            cache = kwargs.pop('cache')
        else:
            cache = False

        dataset_file = dataset_name + str(hash(tuple(kwargs.items())))
        dataset_cache_filename = os.path.join(DATASET_CACHE_DIR, dataset_file + '.pickle')

        if os.path.exists(dataset_cache_filename) and cache:
            with open(dataset_cache_filename, 'rb') as fin:
                return pickle.load(fin)
        else:
            dataset_info = f(**kwargs)

            if cache:
                with open(dataset_cache_filename, 'wb') as fout:
                    pickle.dump(dataset_info, fout)

            return dataset_info

    dataset_list[dataset_name] = call
    return call


def get_dataset_args(initial_args, kwargs):
    kwargs = kwargs.copy()

    for key in initial_args:
        if key not in kwargs:
            kwargs[key] = initial_args[key]

    return kwargs


@dataset
def load_credit(**kwargs):
    raw_df = pd.read_csv('research_paper/datasets/credit.data', index_col=0)
    processed_df = pd.DataFrame()

    # convert NTD to USD using spot rate in 09-2005
    NTD_to_USD = 32.75  # see https://www.poundsterlinglive.com/bank-of-england-spot/historical-spot-exchange-rates/usd/USD-to-TWD-2005
    monetary_features = list(
        filter(lambda x: ('BILL_AMT' in x) or ('PAY_AMT' in x) or ('LIMIT_BAL' in x), raw_df.columns))
    raw_df[monetary_features] = raw_df[monetary_features].applymap(lambda x: x / NTD_to_USD).round(-1).astype(int)

    # outcome variable in first column
    processed_df['y'] = 1 - raw_df['default payment next month (label)']

    # Gender (old; male = 1, female = 2) --> (new; male = 0, female = 1)
    # <Removed by Berk> processed_df['Female'] = raw_df['SEX'] == 2
    processed_df.loc[raw_df['SEX'] == 1, 'isMale'] = True
    processed_df.loc[raw_df['SEX'] == 2, 'isMale'] = False

    # Married (old; married = 1; single = 2; other = 3) --> (new; married = 1; single = 2; other = 3)
    # <Removed by Amir> processed_df['Married'] = raw_df['MARRIAGE'] == 1
    # <Removed by Amir> processed_df['Single'] = raw_df['MARRIAGE'] == 2
    processed_df.loc[
        raw_df.MARRIAGE == 1, 'isMarried'] = True  # married (use T/F, but not 1/0, so that some values become NAN and can be dropped later!)
    processed_df.loc[
        raw_df.MARRIAGE == 2, 'isMarried'] = False  # single (use T/F, but not 1/0, so that some values become NAN and can be dropped later!)
    # <Set to NAN by Amir> processed_df.loc[raw_df.MARRIAGE == 0, 'isMarried'] = 3 # other
    # <Set to NAN by Amir> processed_df.loc[raw_df.MARRIAGE == 3, 'isMarried'] = 3 # other

    # Age
    # <Removed by Amir> processed_df['Age_lt_25'] = raw_df['AGE'] < 25
    # <Removed by Amir> processed_df['Age_in_25_to_40'] = raw_df['AGE'].between(25, 40, inclusive = True)
    # <Removed by Amir> processed_df['Age_in_40_to_59'] = raw_df['AGE'].between(40, 59, inclusive = True)
    # <Removed by Amir> processed_df['Age_geq_60'] = raw_df['AGE'] >= 60
    processed_df.loc[raw_df['AGE'] < 25, 'AgeGroup'] = 1
    processed_df.loc[raw_df['AGE'].between(25, 40, inclusive=True), 'AgeGroup'] = 2
    processed_df.loc[raw_df['AGE'].between(40, 59, inclusive=True), 'AgeGroup'] = 3
    processed_df.loc[raw_df['AGE'] >= 60, 'AgeGroup'] = 4

    # EducationLevel (currently, 1 = graduate school; 2 = university; 3 = high school; 4 = others)
    processed_df['EducationLevel'] = 1
    processed_df.loc[raw_df['EDUCATION'] == 3, 'EducationLevel'] = 2  # HS
    processed_df.loc[raw_df['EDUCATION'] == 2, 'EducationLevel'] = 3  # University
    processed_df.loc[raw_df['EDUCATION'] == 1, 'EducationLevel'] = 4  # Graduate

    # Process Bill Related Variables
    pay_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

    # processed_df['LastBillAmount'] = np.maximum(raw_df['BILL_AMT1'], 0)
    processed_df['MaxBillAmountOverLast6Months'] = np.maximum(raw_df[bill_columns].max(axis=1), 0)
    processed_df['MaxPaymentAmountOverLast6Months'] = np.maximum(raw_df[pay_columns].max(axis=1), 0)
    processed_df['MonthsWithZeroBalanceOverLast6Months'] = np.sum(
        np.greater(raw_df[pay_columns].values, raw_df[bill_columns].values), axis=1)
    processed_df['MonthsWithLowSpendingOverLast6Months'] = np.sum(
        raw_df[bill_columns].div(raw_df['LIMIT_BAL'], axis=0) < 0.20, axis=1)
    processed_df['MonthsWithHighSpendingOverLast6Months'] = np.sum(
        raw_df[bill_columns].div(raw_df['LIMIT_BAL'], axis=0) > 0.80, axis=1)
    processed_df['MostRecentBillAmount'] = np.maximum(raw_df[bill_columns[0]], 0)
    processed_df['MostRecentPaymentAmount'] = np.maximum(raw_df[pay_columns[0]], 0)

    # Credit History
    # PAY_M' = months since last payment (as recorded last month)
    # PAY_6 =  months since last payment (as recorded 6 months ago)
    # PAY_M = -1 if paid duly in month M
    # PAY_M = -2 if customer was issued refund M
    raw_df = raw_df.rename(columns={'PAY_0': 'MonthsOverdue_1',
                                    'PAY_2': 'MonthsOverdue_2',
                                    'PAY_3': 'MonthsOverdue_3',
                                    'PAY_4': 'MonthsOverdue_4',
                                    'PAY_5': 'MonthsOverdue_5',
                                    'PAY_6': 'MonthsOverdue_6'})

    overdue = ['MonthsOverdue_%d' % j for j in range(1, 7)]
    raw_df[overdue] = raw_df[overdue].replace(to_replace=[-2, -1], value=[0, 0])
    overdue_history = raw_df[overdue].to_numpy() > 0
    payment_history = np.logical_not(overdue_history)

    def count_zero_streaks(a):
        # adapted from zero_runs function of https://stackoverflow.com/a/24892274/568249
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        runs = np.where(absdiff == 1)[0].reshape(-1, 2)
        n_streaks = runs.shape[0]
        # streak_lengths = np.sum(runs[:,1] - runs[:,0])
        return n_streaks

    overdue_counts = np.repeat(np.nan, len(raw_df))
    n_overdue_months = np.sum(overdue_history > 0, axis=1)
    overdue_counts[n_overdue_months == 0] = 0  # count_zero_streaks doesn't work for edge cases
    overdue_counts[n_overdue_months == 6] = 1
    for k in range(1, len(overdue)):
        idx = n_overdue_months == k
        overdue_counts[idx] = [count_zero_streaks(a) for a in payment_history[idx, :]]

    overdue_counts = overdue_counts.astype(np.int_)
    processed_df['TotalOverdueCounts'] = overdue_counts
    processed_df['TotalMonthsOverdue'] = raw_df[overdue].sum(axis=1)
    processed_df['HasHistoryOfOverduePayments'] = raw_df[overdue].sum(axis=1) > 0

    override_dtypes = {
        'isMale': object, 'isMarried': object, 'AgeGroup': int, 'EducationLevel': int,
        'MaxBillAmountOverLast6Months': float, 'MaxPaymentAmountOverLast6Months': float,
        'MonthsWithZeroBalanceOverLast6Months': int, 'MonthsWithLowSpendingOverLast6Months': int,
        'MonthsWithHighSpendingOverLast6Months': int, 'MostRecentBillAmount': float,
        'MostRecentPaymentAmount': float, 'TotalOverdueCounts': int, 'TotalMonthsOverdue': int,
        'HasHistoryOfOverduePayments': object
    }

    return process_pandas_dataset(processed_df[list(override_dtypes.keys()) + ['y']].copy(), 'y',
                                  **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_pima(**kwargs):
    df = pd.read_csv('research_paper/datasets/pima.csv')
    override_dtypes = {
        'pregnancies': int,
        'glucose': int, 'blood_pressure': int, 'skin_thickness': int, 'insulin': int, 'bmi': float,
        'diabetes_pedigree_function': float, 'age': int
    }

    return process_pandas_dataset(df[list(override_dtypes.keys()) + ['y']], 'y',
                                  **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_seismic(**kwargs):
    df = pd.read_csv('research_paper/datasets/seismic.csv')
    override_dtypes = {
        'seismic': object,
        'seismoacoustic': object,
        'shift': object,
        'genergy': int,
        'gpuls': int,
        'gdenergy': int,
        'gdpuls': int,
        'ghazard': object,
        'nbumps': int,
        'nbumps2': int,
        'nbumps3': int,
        'nbumps4': int,
        'nbumps5': int,
        'nbumps6': int,
        'nbumps7': int,
        'nbumps89': int,
        'energy': int,
        'maxenergy': int
    }

    return process_pandas_dataset(df, 'y', **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_occupancy(**kwargs):
    df = pd.read_csv('research_paper/datasets/occupancy.csv', index_col=0)
    override_dtypes = {
        "Temperature": float,
        "Humidity": float,
        "Light": float,
        "CO2": float,
        "HumidityRatio": float,
    }

    return process_pandas_dataset(df[list(override_dtypes.keys()) + ['y']], 'y',
                                  **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_compas(**kwargs):
    # https://github.com/amirhk/mace/blob/master/_data_main/process_credit_data.py
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count",
                               "c_charge_degree"]  # features to be used for classification
    CLASS_FEATURE = "two_year_recid"  # the decision variable

    # load the data and get some stats
    df = pd.read_csv('research_paper/datasets/compas-scores-two-years.csv')
    df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals

    tmp = \
        ((df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)) & \
        (df["is_recid"] != -1) & \
        (df["c_charge_degree"] != "O") & \
        (df["score_text"] != "NA") & \
        ((df["race"] == "African-American") | (df["race"] == "Caucasian"))

    df = df[tmp == True]
    df = pd.concat([
        df[FEATURES_CLASSIFICATION],
        df[CLASS_FEATURE],
    ], axis=1)

    processed_df = pd.DataFrame()

    processed_df['y'] = df['two_year_recid']
    processed_df.loc[df['age_cat'] == 'Less than 25', 'AgeGroup'] = 1
    processed_df.loc[df['age_cat'] == '25 - 45', 'AgeGroup'] = 2
    processed_df.loc[df['age_cat'] == 'Greater than 45', 'AgeGroup'] = 3
    processed_df.loc[df['race'] == 'African-American', 'Race'] = 1
    processed_df.loc[df['race'] == 'Caucasian', 'Race'] = 2
    processed_df.loc[df['sex'] == 'Male', 'Sex'] = 1
    processed_df.loc[df['sex'] == 'Female', 'Sex'] = 2
    processed_df['PriorsCount'] = df['priors_count']
    processed_df.loc[df['c_charge_degree'] == 'M', 'ChargeDegree'] = 1
    processed_df.loc[df['c_charge_degree'] == 'F', 'ChargeDegree'] = 2

    override_dtypes = {
        'PriorsCount': int,
        'AgeGroup': int,
        'Sex': int,
        'ChargeDegree': int,
        'Race': object
    }

    return process_pandas_dataset(processed_df, 'y',
                                  **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_adult(**kwargs):
    # Preprocessing source: https://github.com/amirhk/mace/blob/master/_data_main/fair_adult_data.py
    df = pd.read_csv("research_paper/datasets/adult.data")
    df['native-country'] = df['native-country'].str.strip()
    attrs_to_ignore = ['sex', 'race', 'fnlwgt']

    df.drop(columns=attrs_to_ignore, inplace=True)
    df.loc[df['native-country'] != 'United-States', 'native-country'] = 'Non-United-Stated'

    education_mapping = {"Preschool": "prim-middle-school", "1st-4th": "prim-middle-school",
                         "5th-6th": "prim-middle-school", "7th-8th": "prim-middle-school",
                         "9th": "high-school", "10th": "high-school", "11th": "high-school", "12th": "high-school"}

    df['education'] = df['education'].str.strip()
    df['education'] = df['education'].apply(lambda x: education_mapping.get(x, x))
    df['workclass'] = df['workclass'].str.strip()
    df['marital-status'] = df['marital-status'].str.strip()
    df['occupation'] = df['occupation'].str.strip()
    df['relationship'] = df['relationship'].str.strip()

    processed_df = pd.DataFrame()
    processed_df['y'] = df['y']
    processed_df['age'] = df['age']
    processed_df.loc[df['native-country'] == 'United-States', 'NativeCountry'] = 1
    processed_df.loc[df['native-country'] == 'Non-United-Stated', 'NativeCountry'] = 2
    processed_df.loc[df['workclass'] == 'Federal-gov', 'WorkClass'] = 1
    processed_df.loc[df['workclass'] == 'Local-gov', 'WorkClass'] = 2
    processed_df.loc[df['workclass'] == 'Private', 'WorkClass'] = 3
    processed_df.loc[df['workclass'] == 'Self-emp-inc', 'WorkClass'] = 4
    processed_df.loc[df['workclass'] == 'Self-emp-not-inc', 'WorkClass'] = 5
    processed_df.loc[df['workclass'] == 'State-gov', 'WorkClass'] = 6
    processed_df.loc[df['workclass'] == 'Without-pay', 'WorkClass'] = 7
    processed_df['EducationNumber'] = df['education-num']
    processed_df.loc[df['education'] == 'prim-middle-school', 'EducationLevel'] = 1
    processed_df.loc[df['education'] == 'high-school', 'EducationLevel'] = 2
    processed_df.loc[df['education'] == 'HS-grad', 'EducationLevel'] = 3
    processed_df.loc[df['education'] == 'Some-college', 'EducationLevel'] = 4
    processed_df.loc[df['education'] == 'Bachelors', 'EducationLevel'] = 5
    processed_df.loc[df['education'] == 'Masters', 'EducationLevel'] = 6
    processed_df.loc[df['education'] == 'Doctorate', 'EducationLevel'] = 7
    processed_df.loc[df['education'] == 'Assoc-voc', 'EducationLevel'] = 8
    processed_df.loc[df['education'] == 'Assoc-acdm', 'EducationLevel'] = 9
    processed_df.loc[df['education'] == 'Prof-school', 'EducationLevel'] = 10
    processed_df.loc[df['marital-status'] == 'Divorced', 'MaritalStatus'] = 1
    processed_df.loc[df['marital-status'] == 'Married-AF-spouse', 'MaritalStatus'] = 2
    processed_df.loc[df['marital-status'] == 'Married-civ-spouse', 'MaritalStatus'] = 3
    processed_df.loc[df['marital-status'] == 'Married-spouse-absent', 'MaritalStatus'] = 4
    processed_df.loc[df['marital-status'] == 'Never-married', 'MaritalStatus'] = 5
    processed_df.loc[df['marital-status'] == 'Separated', 'MaritalStatus'] = 6
    processed_df.loc[df['marital-status'] == 'Widowed', 'MaritalStatus'] = 7
    processed_df.loc[df['occupation'] == 'Adm-clerical', 'Occupation'] = 1
    processed_df.loc[df['occupation'] == 'Armed-Forces', 'Occupation'] = 2
    processed_df.loc[df['occupation'] == 'Craft-repair', 'Occupation'] = 3
    processed_df.loc[df['occupation'] == 'Exec-managerial', 'Occupation'] = 4
    processed_df.loc[df['occupation'] == 'Farming-fishing', 'Occupation'] = 5
    processed_df.loc[df['occupation'] == 'Handlers-cleaners', 'Occupation'] = 6
    processed_df.loc[df['occupation'] == 'Machine-op-inspct', 'Occupation'] = 7
    processed_df.loc[df['occupation'] == 'Other-service', 'Occupation'] = 8
    processed_df.loc[df['occupation'] == 'Priv-house-serv', 'Occupation'] = 9
    processed_df.loc[df['occupation'] == 'Prof-specialty', 'Occupation'] = 10
    processed_df.loc[df['occupation'] == 'Protective-serv', 'Occupation'] = 11
    processed_df.loc[df['occupation'] == 'Sales', 'Occupation'] = 12
    processed_df.loc[df['occupation'] == 'Tech-support', 'Occupation'] = 13
    processed_df.loc[df['occupation'] == 'Transport-moving', 'Occupation'] = 14
    processed_df.loc[df['relationship'] == 'Husband', 'Relationship'] = 1
    processed_df.loc[df['relationship'] == 'Not-in-family', 'Relationship'] = 2
    processed_df.loc[df['relationship'] == 'Other-relative', 'Relationship'] = 3
    processed_df.loc[df['relationship'] == 'Own-child', 'Relationship'] = 4
    processed_df.loc[df['relationship'] == 'Unmarried', 'Relationship'] = 5
    processed_df.loc[df['relationship'] == 'Wife', 'Relationship'] = 6
    processed_df['CapitalGain'] = df['capital-gain']
    processed_df['CapitalLoss'] = df['capital-loss']
    processed_df['HoursPerWeek'] = df['hours-per-week']

    override_dtypes = {
        'age': int,
        'NativeCountry': object,
        'WorkClass': object,
        'EducationNumber': int,
        'EducationLevel': int,
        'MaritalStatus': object,
        'Occupation': object,
        'Relationship': object,
        'CapitalGain': float,
        'CapitalLoss': float,
        'HoursPerWeek': int
    }

    processed_df = processed_df[list(override_dtypes.keys()) + ['y']]
    return process_pandas_dataset(processed_df, 'y',
                                  **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_mammographic_mases(**kwargs):
    df = pd.read_csv('research_paper/datasets/mammographic_masses.csv')
    df = df[(df == '?').sum(axis=1) == 0].copy()

    override_dtypes = {
        'bi-rads': int,
        'age': int,
        'shape': int,
        'margin': int,
        'density': int
    }

    return process_pandas_dataset(df, 'y', **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_ionosphere(**kwargs):
    df = pd.read_csv('research_paper/datasets/ionosphere.csv')
    return process_pandas_dataset(df, 'y', kwargs)


@dataset
def load_postoperative(**kwargs):
    df = pd.read_csv('research_paper/datasets/post-operative.csv')
    df = df[(df == '?').sum(axis=1) == 0].copy()

    for column in df.columns:
        df[column] = df[column].str.strip()

    override_dtypes = {
        'l_core': object, 'l_surf': object, 'l_02': object, 'l_bp': object, 'surf_stbl': object, 'core_stbl': object,
        'bp_stbl': object, 'comfort': int
    }

    return process_pandas_dataset(df, 'y', **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_banknote(**kwargs):
    df = pd.read_csv('research_paper/datasets/banknote.csv')
    return process_pandas_dataset(df, 'y', kwargs)


@dataset
def load_abalone(**kwargs):
    df = pd.read_csv('research_paper/datasets/abalone.csv')
    df['y'] = (df['y'] <= df['y'].median()).astype(int)

    override_dtypes = {
        'sex': object,
        'length': float,
        'diameter': float,
        'height': float,
        'whole_weight': float,
        'shucked_weight': float,
        'viscera_weight': float,
        'shell_weight': float
    }

    return process_pandas_dataset(df, 'y', **get_dataset_args(dict(override_feature_types=override_dtypes), kwargs))


@dataset
def load_wine(**kwargs):
    df = pd.read_csv('research_paper/datasets/wine.csv', index_col=None)
    df = df[df['y'].isin([1, 2])].copy()
    return process_pandas_dataset(df, 'y', kwargs)


def load_dataset(name, **kwargs):
    if name in dataset_list:
        return dataset_list[name](**kwargs)
    else:
        raise KeyError('Dataset {name} not found'.format(name=name))


def process_pandas_dataset(df, y_name, dataset_name='dataset', use_one_hot=False, override_feature_types=None,
                           select_dtypes=None, max_size=None):
    df = df.copy()
    df.dropna(axis=0, inplace=True)
    if override_feature_types is not None:
        for feature, feature_type in override_feature_types.items():
            df[feature] = df[feature].astype(feature_type)

    y = df[y_name]
    X = df.drop(columns=y_name)

    if y.dtype == float:
        raise ValueError("Y class should be an integer or object")
    else:
        unique_values = y.unique()
        values_mapping = dict(zip(unique_values, range(len(unique_values))))
        y = y.apply(lambda x: values_mapping[x])

    if select_dtypes:
        X = X.select_dtypes(select_dtypes)

    X_prep = None

    for feature in X.columns:
        if len(X[feature].unique()) <= 1:
            print("Column {feature} has been removed because it is constant".format(feature=feature))
            X.drop(columns=feature, inplace=True)

    if max_size is not None and len(X) > max_size:
        X = X.sample(max_size, random_state=0).copy()
        y = y.loc[X.index].copy()

    if float in X.dtypes.values:
        float_transformer = StandardScaler()
        float_data = X.select_dtypes(float)
        transformed_float_data = float_transformer.fit_transform(float_data)
        transformed_float_data = pd.DataFrame(data=transformed_float_data, columns=float_data.columns,
                                              index=float_data.index)
        X.loc[transformed_float_data.index, transformed_float_data.columns] = transformed_float_data
    else:
        float_transformer = None

    dataset_builder = datasets.DatasetInfoBuilder(dataset_name, float_transformer)
    columns = []

    for column, dtype in X.dtypes.to_dict().items():
        current_feature = len(columns)
        column_data = X[column]

        if dtype == int:
            dataset_builder.add_ordinal_variable(current_feature, column_data.min(), column_data.max(),
                                                 name=column)
            columns.append(column)
            new_column_data = column_data
        elif dtype == float:
            dataset_builder.add_numerical_variable(current_feature, column_data.min(), column_data.max(),
                                                   name=column)
            columns.append(column)
            new_column_data = column_data
        elif dtype == object:
            if use_one_hot:
                column_dummies = pd.get_dummies(column_data, prefix='{}'.format(column))

                if len(column_dummies.columns) == 2:
                    new_column_data = column_dummies[column_dummies.columns[0]]
                    dataset_builder.add_binary_variable(current_feature, name=column,
                                                        category_names=column_dummies.columns)
                    columns.append(column_dummies.columns[0])
                else:
                    dataset_builder.add_one_hot_varible(current_feature, len(column_dummies.columns),
                                                        name=column, category_names=column_dummies.columns)
                    new_column_data = column_dummies
                    columns.extend(column_dummies.columns)
            else:
                unique_values = column_data.unique()
                unique_values_mapping = dict(zip(sorted(unique_values), range(len(unique_values))))
                new_column_data = column_data.apply(lambda x: unique_values_mapping[x])
                columns.append(column)
                dataset_builder.add_categorical_numerical(current_feature, len(unique_values), name=column,
                                                          category_names=unique_values_mapping)
        else:
            raise ValueError("Type {} in column {} not supported".format(dtype, column))

        if X_prep is None:
            X_prep = pd.DataFrame(new_column_data)
        else:
            X_prep = pd.concat((X_prep, new_column_data), axis=1)

    return dataset_builder.create_dataset_info(), X_prep[columns], y
