
import pandas as pd
import numpy as np

from sklearn import preprocessing


def load():
	le = preprocessing.LabelEncoder()
	# lb = preprocessing.LabelBinarizer()

	train_df = pd.read_csv('data/datasets/train.csv')
	# test_df = pd.read_csv('datasets/test.csv')

	# tmp = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


	# train_df.Cabin = train_df.Cabin.fillna('N')
	# train_df.Embarked = train_df.Embarked.fillna('Nan')
	train_df.Age = train_df.Age.fillna(-1)
	train_df.Age = train_df.Fare.fillna(-1)
	train_df.Cabin = train_df.Cabin.fillna("Nan")
	train_df.Cabin = train_df.Cabin.apply(lambda x: x[0])
	# train_df.Cabin = train_df.Cabin.map( {'N': -1, 'C': 0, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8,} ).astype(float)
	# train_df.Embarked = train_df.Embarked.map( {'N' : -1,'C': 0, 'Q': 1, 'S': 2} ).astype(float)
	# train_df.Sex = train_df.Sex.map( {'female':-1, 'male': 1} ).astype(float)

	for dataset in [train_df]:

		# dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
		# dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
		# dataset['Cabin'].fillna(dataset['Cabin'].mode()[0], inplace=True)


		# dataset['Sex'] = le.fit_transform(dataset['Sex'])
		# dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
		dataset['Cabin'] = le.fit_transform(dataset['Cabin'])
		dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
		# dataset['Title'] = le.fit_transform(dataset['Title'])
		# dataset['CabinNum'] = dataset['Cabin'][1:]
		# dataset['CabinNum'].fillna(dataset['CabinNum'].mode(), inplace=True)



		# dataset['TrAge6'] = 0
		# dataset['TrAge18'] = 0
		# dataset['TrAge40'] = 0
		# dataset['TrAge60'] = 0
		# dataset['TrAgeOld'] = 0
		# dataset.loc[dataset['Age'] <= 6, 'TrAge6'] = 1
		# dataset.loc[(dataset['Age'] > 6) & (dataset['Age'] <= 18), 'TrAge18'] = 1
		# dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 40), 'TrAge40'] = 1
		# dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'TrAge40'] = 1
		# dataset.loc[ dataset['Age'] > 60, 'TrAgeOld'] = 1


		# dataset['TrFare50'] = 0
		# dataset['TrFare100'] = 0
		# dataset['TrFare200'] = 0
		# dataset['TrFare300'] = 0
		# dataset['TrFareOld'] = 0
		# dataset.loc[dataset['Fare'] <= 50, 'TrFare50'] = 1
		# dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 100), 'TrFare100'] = 1
		# dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 200), 'TrFare200'] = 1
		# dataset.loc[(dataset['Fare'] > 200) & (dataset['Fare'] <= 300), 'TrFare300'] = 1
		# dataset.loc[ dataset['Fare'] > 300, 'TrFareOld'] = 1

		# print(pd.get_dummies(dataset['Age']))
		# exit(0)

		dataset['Family_size'] = (dataset['Parch'] + dataset['SibSp'])

		dataset['IsAlone'] = 0
		# dataset['IsTwo'] = 0
		dataset['IsNotAlone'] = 0
		dataset.loc[dataset['Family_size'] == 1, 'IsAlone'] = 1
		# dataset.loc[dataset['Family_size'] == 2, 'IsTwo'] = 1
		dataset.loc[dataset['Family_size'] > 1, 'IsNotAlone'] = 1

	train_df = pd.DataFrame(data=train_df)

	forDummies = ['Ticket','Cabin', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch', 'Family_size', 'Title']
	for label in forDummies:
		train_df = pd.concat([train_df, pd.get_dummies(train_df[label], prefix=label)], axis=1)

	# train_df = train_df.dropna(axis=0, how='any')

	remove = ['Age', 'Fare', 'PassengerId', 'Name'] + forDummies
	# remove = []
	train_df = train_df.drop(remove, axis=1)

	# Check where is null
	# null_columns=train_df.columns[train_df.isnull().any()]
	# print(train_df[null_columns].isnull().sum())
	# exit(0)



	X_train = train_df.drop("Survived", axis=1)
	Y_train = train_df["Survived"]

	print(X_train[:10].to_string())
	# exit(0)

	############################## Scale ##############################
	X_train = preprocessing.MinMaxScaler().fit_transform(X_train.values)

	############################## Split ##############################

	split = int(len(X_train) * 0.2)

	X_test = X_train[:split]
	X_train = X_train[split:]

	Y_test = Y_train[:split]
	Y_train = Y_train[split:]

	# print(X_train.shape)
	# print(X_test.shape)
	# exit(0)

	return (X_train, Y_train), (X_test, Y_test)
