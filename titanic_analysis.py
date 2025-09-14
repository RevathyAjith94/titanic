import pandas as pd

print(pd.__version__)

df = pd.read_csv(r'C:\Users\revat\Downloads\titanic_project\titanic.csv')

print("First 5 rows of the dataset:")

print(df.head())

print(df.info())

print(df.describe())

print("\n=== Missing Values Before ===")
print(df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)


print("\n=== Missing Values After ===")
print(df.isnull().sum())


df.rename(columns={'Survived':'Target', 'Pclass':'PassengerClass'}, inplace=True)


print("\n=== Target Counts ===")
print(df['Target'].value_counts())

print("\n=== Mean Age ===")
print(df['Age'].mean())

print("\n=== Survival by Class ===")
print(df.groupby('PassengerClass')['Target'].mean())

df.to_csv("titanic_cleaned.csv", index=False)

print("\n=== Titanic Data Cleaning & Analysis Completed ===")


