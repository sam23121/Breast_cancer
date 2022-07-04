import os
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

def DBConnect(dbName=None):
    """
    Parameters
    ----------
    dbName :
        Default value = None)
    Returns
    -------
    """
    conn = mysql.connect(host='localhost', user='root', password='sam12345',
                         database=dbName, buffered=True)
    cur = conn.cursor()
    return conn, cur

def emojiDB(dbName: str) -> None:
    conn, cur = DBConnect(dbName)
    dbQuery = f"ALTER DATABASE {dbName} CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;"
    cur.execute(dbQuery)
    conn.commit()

def createDB(dbName: str) -> None:
    """
    Parameters
    ----------
    dbName :
        str:
    dbName :
        str:
    dbName:str :
    Returns
    -------
    """
    conn, cur = DBConnect()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {dbName};")
    conn.commit()
    cur.close()

def createTables(dbName: str) -> None:
    """
    Parameters
    ----------
    dbName :
        str:
    dbName :
        str:
    dbName:str :
    Returns
    -------
    """
    conn, cur = DBConnect(dbName)
    sqlFile = 'bc_schema.sql'
    fd = open(sqlFile, 'r')
    readSqlFile = fd.read()
    fd.close()

    sqlCommands = readSqlFile.split(';')

    for command in sqlCommands:
        try:
            res = cur.execute(command)
        except Exception as ex:
            print("Command skipped: ", command)
            print(ex)
    conn.commit()
    cur.close()

    return
# def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
#     """

#     Parameters
#     ----------
#     df :
#         pd.DataFrame:
#     df :
#         pd.DataFrame:
#     df:pd.DataFrame :


#     Returns
#     -------

#     """
#     cols_2_drop = ['Unnamed: 0']
#     try:
#         df = df.drop(columns=cols_2_drop, axis=1)
#     except KeyError as e:
#         print("Error:", e)

#     return df

def insert_to_table(dbName: str, df: pd.DataFrame, table_name: str) -> None:
    """
    Parameters
    ----------
    dbName :
        str:
    df :
        pd.DataFrame:
    table_name :
        str:
    dbName :
        str:
    df :
        pd.DataFrame:
    table_name :
        str:
    dbName:str :
    df:pd.DataFrame :
    table_name:str :
    Returns
    -------
    """
    conn, cur = DBConnect(dbName)

    # df = preprocess_df(df)
    df = df.astype(object).where(pd.notnull(df), None)

    for _, row in df.iterrows():
        sqlQuery = f"""INSERT INTO {table_name} (`diagnosis`  `radius_mean` ,
    `texture_mean` ,
    `perimeter_mean` ,
    `area_mean` ,
    `concavity_mean` ,
    `concave points_mean` ,
    `area_se` ,
    `radius_worst` ,
    `texture_worst` ,
    `perimeter_worst` ,
    `area_worst` ,
    `smoothness_worst` ,
    `compactness_worst` ,
    `concavity_worst` ,
    `concave points_worst` ,
    `symmetry_worst`)
             VALUES(%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s);"""
        data = (row[0], row[1], row[2])

        try:
            # Execute the SQL command
            cur.execute(sqlQuery, data)
            # Commit your changes in the database
            conn.commit()
            print("Data Inserted Successfully")
        except Exception as e:
            conn.rollback()
            print("Error: ", e)
    return

def db_execute_fetch(*args, many=False, tablename='', rdf=True, **kwargs) -> pd.DataFrame:
    """
    Parameters
    ----------
    *args :
    many :
         (Default value = False)
    tablename :
         (Default value = '')
    rdf :
         (Default value = True)
    **kwargs :
    Returns
    -------
    """
    connection, cursor1 = DBConnect(**kwargs)
    if many:
        cursor1.executemany(*args)
    else:
        cursor1.execute(*args)

    # get column names
    field_names = [i[0] for i in cursor1.description]

    # get column values
    res = cursor1.fetchall()

    # get row count and show info
    nrow = cursor1.rowcount
    if tablename:
        print(f"{nrow} records fetched from {tablename} table")

    cursor1.close()
    connection.close()

    # return result
    if rdf:
        return pd.DataFrame(res, columns=field_names)
    else:
        return res


if __name__ == "__main__":
    createDB(dbName='breast_cancer')
    emojiDB(dbName='breast_cancer')
    createTables(dbName='breast_cancer')

    df = pd.read_csv('../data/16_features.csv', index_col=[0])

    insert_to_table(dbName='breast_cancer', df=df, table_name='bc')