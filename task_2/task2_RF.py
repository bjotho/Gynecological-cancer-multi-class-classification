import warnings
import pyreadstat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


warnings.simplefilter(action="ignore", category=FutureWarning)

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

# Set random seed for reproducible results
seed = 2
np.random.seed(seed)

# Randomly shuffle rows
df = df.sample(frac=1)

X = df.drop("kreftform", axis=1)
Y = df["kreftform"]

feature_list = []
for feature in X.columns:
    feature_list.append([feature, 0])

rounds = 100

for i in range(rounds):

    if i % 10 == 0:
        print("%.0f" % ((i*100)/rounds), "%")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    classifier.fit(X_train, Y_train)

    # print("Accuracy:", accuracy_score(Y_test, classifier.predict(X_test)))

    for n, feature in enumerate(zip(X.columns, classifier.feature_importances_)):
        feature_list[n][1] += feature[1] / rounds

feature_df = pd.DataFrame(feature_list)
sorted_features = feature_df.sort_values(by=[1], ascending=False)
print(sorted_features)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(sorted_features)

sns.set(rc={"figure.figsize":(11.7,8.27)})
sns.set(style="darkgrid")
ax = sns.barplot(x=0, y=1, data=sorted_features[:10].reset_index())
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.xlabel("Feature name")
plt.ylabel("Importance")
plt.show()
# ax.figure.savefig("task2_RF.png")
