import warnings
import pyreadstat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


warnings.simplefilter(action="ignore", category=FutureWarning)

rounds = 100
average = 0

for i in range(rounds):
    # Set random seed for reproducible results
    np.random.seed(i)

    # Read sav file and create a pandas dataframe and extract metadata
    df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", usecols=["Sympt_blødning", "Sympt_smerter", "Sympt_ascites", "Sympt_fatigue", "Lengde_sympt_dager", "Lengde_sympt_uker", "Lengde_sympt_mnd", "kreftform"])

    # Randomly shuffle rows
    df = df.sample(frac=1)
    # print(df.head())

    X_train, X_test, Y_train, Y_test = train_test_split(df.drop("kreftform", axis=1), df["kreftform"])

    classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    classifier.fit(X_train, Y_train)

    predictions = classifier.predict(X_test)

    print("Round", i, "\n", pd.crosstab(Y_test, predictions, rownames=["Actual class"], colnames=["Predicted class"]))

    average += accuracy_score(Y_test, predictions)*100 / rounds

print("\nAverage accuracy:", "%.2f" % average + "%")
