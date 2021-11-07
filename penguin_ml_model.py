import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

penguin_df = pd.read_csv("penguins.csv")
#print(penguin_df.head())
#print(penguin_df.tail())

#remove nan value
penguin_df.dropna(inplace=True)

output = penguin_df['species'] #target
features = penguin_df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']]

#original after cleaning nan value
#print(output.tail())
#print(features.head())

#feature after
features = pd.get_dummies(features)
#print(features.tail())

output, uniques = pd.factorize(output)
#print(uniques)
#print(output)

#train test split
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.2)

rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)

#predict parameter using test set data
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print ("Our accuracy score for this model is {}".format(score))

#save the penguin RF RandomForestClassifierr
rf_pickle=open("random_forest_penguin.pickle",'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle = open ('output_penguin.pickle','wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

print('Success')
