# importing required modules
import pandas as pd
import sys
import os

EXPORTED_PATH = '.\\updated-files'


def formatCheck(file_name):
    # split the name and extension apart and return
    name, ext = file_name.split('.')
    return name, ext


def readDataSet(path, ext):
    # reading dataset function
    # check the extension and load accordingly into dataframe
    try:
        if ext == 'csv':
            df = pd.read_csv(path)
        elif ext == 'xlsx':
            df = pd.read_excel(path)
        elif ext == 'html':
            df = pd.read_html(path)
        elif ext == 'json':
            df = pd.read_json(path)
        else:
            print('  ERROR: Not supported format')
            print('  It supports following format\n\t1| .csv\n\t2| .xlsx'
                  '\n\t  3| .html\n\t4| .json')
            exit(1)
    except:
        print(f'  ERROR: dataset name {path} not found')
        exit(1)

    return df


def Import():
    # dataset importing functions
    length = len(sys.argv)
    if length == 2:
        # get the path of the file
        path = sys.argv[1]
        file_name = path.split('\\')[-1]
    else:
        print('  ERROR: dataset not specified\n  Command: python dataClean.py <DatasetName | path_to_file>')
        exit(1)

    # get the name and extension apart
    file_name, ext = formatCheck(file_name)

    # load the dataset into dataframe
    df = readDataSet(path, ext)

    return df, file_name, ext


def duplicateRows(df):
    # function for removing duplicates rows if any
    return df.drop_duplicates()


def datatypeChanged(changing_col, df):
    # for each column in changing_col change datatype to 'object'
    for col in changing_col:
        df = df.astype({f'{col}': 'object'})
    return df


def changeDataType(df):
    # function for changing the datatype to object
    change_col = []
    change_type = []
    change_count = []
    rows, col = df.shape
    # finding unique values and datatype of each row
    for column in df.columns:
        unique = df[column].nunique()
        dtype = df[column].dtype
        if dtype != 'object' and (unique / rows * 100) <= 1.5:
            change_col.append(column)
            change_type.append(dtype)
            change_count.append(unique)

    changing_col = []
    # asking from the user to change the datatype
    for i in range(len(change_col)):
        print(f'    {change_col[i]} has only {change_count[i]} unique value out of {rows} rows')
        ch = input(f"    You should change it's datatype from {change_type[i]} to object (y/n) : ")
        if ch.lower() == 'y':
            changing_col.append(change_col[i])

    # if changing_col has any element then call the function
    if len(changing_col) > 0:
        df = datatypeChanged(changing_col, df)
    return df


def ImputingMissing(final_cols, df):
    # Imputing missing value function
    # for each column in final_cols
    # check the data-type and impute missing value accordingly
    for col in final_cols:
        if df[col].dtype == 'float64':
            df.fillna(value=df[col].mean(), inplace=True)

        if df[col].dtype == 'int64':
            df.fillna(value=int(df[col].mean()), inplace=True)

        if df[col].dtype == 'object':
            df.fillna(value=df[col].mode(), inplace=True)

    return df


def missingValue(df):
    # checking missing value columns
    missing_cols = []
    missing_counts = []
    rows, cols = df.shape
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            missing_cols.append(col)
            missing_counts.append(missing)

    final_cols = []
    for i in range(len(missing_cols)):
        per = missing_counts[i] / rows * 100
        # drop if percentage is greater than 30
        if per >= 30:
            print(f"    You must delete column-{missing_cols[i]} as it has "
                  f"    {missing_counts[i]} / {rows} missing-entities which is {per} %")
            ch = input("    You choice (Y/N) : ")
            if ch.lower() == 'y':
                df.drop(missing_cols[i], axis='columns', inplace=True)
            else:
                final_cols.append(missing_cols[i])
        else:
            # print(f'appending {missing_cols[i]}')
            final_cols.append(missing_cols[i])

    # if final_cols has entries than impute missing value in those columns
    if len(final_cols) > 0:
        df = ImputingMissing(final_cols, df)

    return df


def makedummy(df):
    # making dummy ( i.e converting objects data-types into int
    # so can easily fed into models)
    rows, cols = df.shape
    dropping_list = []
    for column in df.columns:
        if df[column].dtype == 'object' and (df[column].nunique()/rows * 100) > 1:
            dropping_list.append(column)
            print(f'Dropped {column} ...')
    return pd.get_dummies(df.drop(dropping_list, axis=1))


def Export(df, file_name, ext):
    # Exporting function
    print(f'Exporting {file_name}....')
    if ext == 'csv':
        df.to_csv(os.path.join(EXPORTED_PATH, str(file_name) + '-cleaned.' + str(ext)), index=False)
    elif ext == 'xlsx':
        df.to_excel(os.path.join(EXPORTED_PATH, str(file_name) + '-cleaned.' + str(ext)), index=False)
    elif ext == 'html':
        df.to_html(os.path.join(EXPORTED_PATH, str(file_name) + '-cleaned.' + str(ext)), index=False)
    elif ext == 'json':
        df.to_json(os.path.join(EXPORTED_PATH, str(file_name) + '-cleaned.' + str(ext)), index=False)


def Check(df):
    # checking function
    temp = pd.DataFrame()
    temp['unique'] = df.nunique()
    temp['dtype'] = df.dtypes
    temp['null'] = df.isnull().sum()
    print(temp)


def main():   # main function
    # importing dataset
    df, file_name, ext = Import()
    # print(f'After Importing: {df.dtypes}')

    # a function to check unique value, datatype and
    # null value in each column of a dataframe
    # NOTE: used for debugging
    # Check(df)

    # removing duplicates rows from the dataset
    df = duplicateRows(df)
    print(f'Duplicates rows checked...')

    # changing datatype to object, if any category is there
    df = changeDataType(df)
    print(f'DataTypes checked...')

    # Imputing missing values in the dataset
    df = missingValue(df)
    print(f'Missing Values checked...')

    # making the dummy
    df = makedummy(df)

    # NOTE: used for debugging
    # Check(df)

    # Exporting dataset
    Export(df, file_name, ext)
    print(f'Exported')


if __name__ == '__main__':
    main()
