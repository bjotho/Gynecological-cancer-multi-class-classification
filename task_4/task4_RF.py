import warnings
import pyreadstat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


warnings.simplefilter(action="ignore", category=FutureWarning)

# Read sav file and create a pandas dataframe and extract metadata
df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", dates_as_pandas_datetime=True)

date_fields = ["Født", "Oppdaget_når", "Død", "Avsluttet_prim_beh"]
string_fields = ["Ktr_andre", "Oppdaget_annen", "Sympt_annet", "Verif_annet", "Lok_annet", "Hvorfor_ikke"]

for i in date_fields:
    df[i] = (df[i].astype(np.int64) // 10 ** 9) / 10 ** 9

for i in string_fields:
    df = df.drop(i, axis=1)

imp = SimpleImputer(strategy="mean", fill_value=0)
clean_dataset = imp.fit_transform(df)

col_names = [i for i in meta.column_names if i not in string_fields]

df = pd.DataFrame(clean_dataset, columns=col_names)

rounds = 100
average = 0

for i in range(rounds):

    if i % 10 == 0:
        print("%.0f" % ((i*100)/rounds), "%")

    # Set random seed for reproducible results
    seed = i
    np.random.seed(seed)

    # Randomly shuffle rows
    round_df = df.sample(frac=1)
    # print(df.head())

    X_train, X_test, Y_train, Y_test = train_test_split(round_df.drop("kreftform", axis=1), round_df["kreftform"])

    classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    classifier.fit(X_train, Y_train)

    predictions = classifier.predict(X_test)

    average += accuracy_score(Y_test, predictions)*100 / rounds

print("Average accuracy:", "%.2f" % average)