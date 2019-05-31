import pyreadstat
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducible results
np.random.seed(1)

# Read sav file and create a pandas dataframe and extract metadata
df, meta = pyreadstat.read_sav("RESIDIV_Vimala.sav", usecols=["Sympt_blødning", "Sympt_smerter", "Sympt_ascites", "Sympt_fatigue", "kreftform"])

Y = df["kreftform"].values
X = df.drop("kreftform", axis=1).values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one-hot encoded)
onehot_Y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(5, input_dim=(len(X[0])), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(onehot_Y[0]), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, onehot_Y, validation_split=0.33, epochs=200)
accuracy = "%.2f" % (model.evaluate(X, onehot_Y)[1]*100)
print("Accuracy:", accuracy, "%")
