# RAW Column Names
RAW_EMPLOYMENT_LESS_ONE_Y = "emp_length_< 1 year"
RAW_EMPLOYMENT_MORE_TEN_Y = "emp_length_10+ years"

# Preprocessed Column names
PREPROC_EMPLOYMENT_LESS_ONE_Y = "emp_length_lt_1_year"
PREPROC_EMPLOYMENT_MORE_TEN_Y = "emp_length_more_10_year"

# Renaming dict
RENAMING_DICT = {
    RAW_EMPLOYMENT_LESS_ONE_Y: PREPROC_EMPLOYMENT_LESS_ONE_Y,
    RAW_EMPLOYMENT_MORE_TEN_Y: PREPROC_EMPLOYMENT_MORE_TEN_Y,
}

# Binary columns
LOAN_DURATION = "loan duration"
BINARY_COLUMNS = [LOAN_DURATION]

# Categorical Columns
PURPOSE = "purpose"
HOME_OWNERSHIP = "home_ownership"
EMP_TITLE = "emp_title"
EMP_LENGTH = "emp_length"
CATEGORICAL_COLUMNS = [PURPOSE, HOME_OWNERSHIP, EMP_TITLE, EMP_LENGTH]

# Ordinal Columns
GRADE = "grade"
ORDINAL_COLUMNS = [GRADE]

# Columns to drop
DROP_COLUMNS = ["Unnamed: 0", "sub_grade", "revol_util", "num_rev_accts", "zip_code"]
