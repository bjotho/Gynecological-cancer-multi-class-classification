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

# Set random seed for reproducible results
np.random.seed(1)

# Read sav file and create a pandas dataframe and extract metadata
df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", usecols=["Sympt_blødning", "Sympt_smerter", "Sympt_ascites", "Sympt_fatigue", "Lengde_sympt_dager", "Lengde_sympt_uker", "Lengde_sympt_mnd", "kreftform"], dates_as_pandas_datetime=True)

# Randomly shuffle rows
df = df.sample(frac=1)

Y_df = df["kreftform"]
Y = Y_df.values
df = df.drop("kreftform", axis=1)

X = df.values

X_train, X_test, Y_train, Y_test = train_test_split(df, Y_df)

classifier = RandomForestClassifier(n_jobs=2, random_state=0)
classifier.fit(X_train, Y_train)

RF_predictions = classifier.predict(df.values)

df["RF_predictions"] = RF_predictions


X = df.values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to lists of binary variables (i.e. one-hot encoded)
onehot_Y = np_utils.to_categorical(encoded_Y)

model = Sequential()

model.add(Dense(5, input_dim=(len(X[0])), activation="sigmoid"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(onehot_Y[0]), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, onehot_Y, validation_split=0.33, epochs=1000)
accuracy = "%.2f" % (model.evaluate(X, onehot_Y)[1]*100)

print("Accuracy:", accuracy, "%")
