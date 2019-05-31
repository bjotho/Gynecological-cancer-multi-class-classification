import pyreadstat
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducible results
np.random.seed(1)

# Read sav file and create a pandas dataframe and extract metadata
df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", dates_as_pandas_datetime=True)

date_columns = ["Født", "Oppdaget_når", "Død", "Avsluttet_prim_beh"]
string_columns = ["Ktr_andre", "Oppdaget_annen", "Sympt_annet", "Verif_annet", "Lok_annet", "Hvorfor_ikke"]
remove_columns = ["Pas_ID", "hist_nummer", "Histo_nummer", "Cytol_nummer"]

for i in date_columns:
    df[i] = (df[i].astype(np.int64) // 10 ** 9) / 10 ** 9

for i in string_columns:
    df = df.drop(i, axis=1)

for i in remove_columns:
    df = df.drop(i, axis=1)

imp = SimpleImputer(strategy="constant", fill_value=-1)
clean_dataset = imp.fit_transform(df)

col_names = [i for i in meta.column_names if (i not in string_columns) and (i not in remove_columns)]

df = pd.DataFrame(clean_dataset, columns=col_names)

Y = df["kreftform"].values
X = df.drop("kreftform", axis=1).values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one-hot encoded)
onehot_Y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(len(X[0]), input_dim=(len(X[0])), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(onehot_Y[0]), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, onehot_Y, validation_split=0.33, epochs=1000)
accuracy = "%.2f" % (model.evaluate(X, onehot_Y)[1]*100)
print("Accuracy:", accuracy, "%")
