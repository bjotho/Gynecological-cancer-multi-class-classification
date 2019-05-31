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
df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", usecols=["Sympt_blødning", "Sympt_smerter", "Sympt_ascites", "Sympt_fatigue", "kreftform"])

col_names = [i for i in meta.column_names]

col_names.remove("kreftform")

best_accuracy = 0
average_accuracy = 0
best_accuracy_round = 0
rounds = 1

for round in range(rounds):
    print("\n/////////////////////////    ROUND", (round+1), "    /////////////////////////\n")

    # Set random seed for reproducible results
    np.random.seed(round)

    # Randomly shuffle rows
    df = df.sample(frac=1)

    Y_df = df["kreftform"]
    Y = Y_df.values
    round_df = df.drop("kreftform", axis=1)

    # # Normalize certain columns in the dataframe
    # for i in irrelevant_fields:
    #     df[i] = (df[i]-df[i].mean())/df[i].std()

    X_train, X_test, Y_train, Y_test = train_test_split(round_df, Y_df, test_size=0.5)

    classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    classifier.fit(X_train, Y_train)

    # print(accuracy_score(Y_test, classifier.predict(X_test)))

    RF_predictions = classifier.predict(X_test.values)

    # usecols = ["Sympt_blødning", "Sympt_smerter", "Sympt_ascites", "Sympt_fatigue", "Sympt_tilstede", "Sympt_vekttap", "Sympt_utflod", "Sympt_avføring", "Sympt_ileus", "Sympt_vannlating", "Sympt_blodfeces", "Sympt_blodurin", "Sympt_anorexi", "Sympt_kakeksi", "Sympt_hudtumor", "sympt_luftveier", "FIGO_stadium", "Substadium", "gradering", "Primærbehandling", "Lengde_sympt_dager", "Lengde_sympt_uker", "Lengde_sympt_mnd", "Oppdaget_pas", "Oppdaget_FL", "Oppdaget_Priv", "Oppdaget_Univ", "Oppdaget_Kir", "Tidligere_kontakt", "Verif_cytol", "Verif_CT", "Verif_RTH", "Verif_ABD_UL", "Verif_VAG_UL", "ULverif", "Verif_Colon", "Lok_vagina", "Lok_bekken", "Lok_ØvreAbd", "Lok_lunger", "Lok_lever", "Lok_lok_LK", "Lok_fjern_LK", "Lok_skjelett", "Lok_hjernen", "Henvist_Uni", "Behandling", "symptomer_kliniske", "KlinSympt_2dic", "symptomer", "Symtomer_tilstede", "Sympt_tilstede2dic", "Distal_recurrence", "Local_recurrence"]
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

    print(df2)

    df2 = df2.drop("kreftform", axis=1)

    X = df2.values

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to lists of binary variables (i.e. one-hot encoded)
    onehot_Y = np_utils.to_categorical(encoded_Y)

    model = Sequential()
    model.add(Dense(5, input_dim=(len(X[0])), activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(onehot_Y[0]), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X, onehot_Y, validation_split=0.33, epochs=1000)
    accuracy = model.evaluate(X, onehot_Y)[1]

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
print("\nBest accuracy:", best_accuracy, "\nAt round:", best_accuracy_round, "\n\nAverage accuracy:", average_accuracy)