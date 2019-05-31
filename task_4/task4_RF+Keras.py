import warnings
import pyreadstat
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


warnings.simplefilter(action="ignore", category=FutureWarning)

# Read sav file into a pandas dataframe and extract metadata
df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", dates_as_pandas_datetime=True)

date_columns = ["Født", "Oppdaget_når", "Død", "Avsluttet_prim_beh"]
string_columns = ["Ktr_andre", "Oppdaget_annen", "Sympt_annet", "Verif_annet", "Lok_annet", "Hvorfor_ikke"]
remove_columns = ["Pas_ID", "hist_nummer", "Histo_nummer", "Cytol_nummer"]

# Convert date columns to float
for i in date_columns:
    df[i] = (df[i].astype(np.int64) // 10 ** 9) / 10 ** 9

# Remove string columns from the dataframe
for i in string_columns:
    df = df.drop(i, axis=1)

# Remove specified columns from the dataframe
for i in remove_columns:
    df = df.drop(i, axis=1)

imp = SimpleImputer(strategy="constant", fill_value=-1)
clean_dataset = imp.fit_transform(df)

col_names = [i for i in meta.column_names if (i not in string_columns) and (i not in remove_columns)]

df = pd.DataFrame(clean_dataset, columns=col_names)

col_names.remove("kreftform")

best_accuracy = 0
average_accuracy = 0
best_accuracy_round = 0
rounds = 5
output = []

for round in range(rounds):
    print("\n/////////////////////////    ROUND", (round+1), "    /////////////////////////\n")

    # Set random seed for reproducible results
    np.random.seed(round)

    # Randomly shuffle rows
    df = df.sample(frac=1)

    output.append([])

    Y_df = df["kreftform"]
    Y = Y_df.values
    round_df = df.drop("kreftform", axis=1)

    # # Normalize certain columns in the dataframe
    # for i in irrelevant_fields:
    #     df[i] = (df[i]-df[i].mean())/df[i].std()

    X_train, X_test, Y_train, Y_test = train_test_split(round_df, Y_df)

    classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    classifier.fit(X_train, Y_train)

    output[round].append(accuracy_score(Y_test, classifier.predict(X_test)))

    RF_predictions = classifier.predict(X_test.values)

    usecols = ["Sympt_blødning", "Sympt_smerter", "Sympt_ascites", "Sympt_fatigue"]

    df2 = pd.DataFrame()

    features = []
    for n, i in enumerate(col_names):
        features.append([])
        for j in X_test[i]:
            features[n].append(j)

    for n, i in enumerate(usecols):
        df2[i] = features[n]

    df2["RF_predictions"] = RF_predictions

    Y = []
    for i in Y_test:
        Y.append(i)

    df2["kreftform"] = Y

    df2 = df2.drop("kreftform", axis=1)

    X = df2.values

    print(df2)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to lists of binary variables (i.e. one-hot encoded)
    onehot_Y = np_utils.to_categorical(encoded_Y)

    model = Sequential()
    model.add(Dense(len(X[0]), input_dim=(len(X[0])), activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(len(onehot_Y[0]), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X, onehot_Y, validation_split=0.33, epochs=1000)
    accuracy = model.evaluate(X, onehot_Y)[1]

    output[round].append(accuracy)

    average_accuracy += accuracy / rounds

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_accuracy_round = (round+1)

    # prediction_prob = model.predict(X)
    #
    # output = []
    # lines = 5
    #
    # for n, i in enumerate(prediction_prob):
    #     output.append([])
    #     for j in i:
    #         output[n].append("%.2f" % j)
    #
    #     if n < lines:
    #         print(output[n])

print("---------------------------------------------------------------")
print("\nBest accuracy:", "%.2f" % (best_accuracy*100), "%", "\nAt round:", best_accuracy_round, "\n\nAverage accuracy:", "%.2f" % (average_accuracy*100))

print("Comparison acuracies ([0] = RF acc, [1] = MLP acc):", output)