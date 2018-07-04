import pandas as pd 
import matplotlib.pyplot as plt

!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# 2. Load a file by ID and create local file.
downloaded = drive.CreateFile({'id':'15m9PfW4rgUTlN3PH_XAVohxUuPe_-80G'}) # replace fileid with Id of file you want to access
downloaded.GetContentFile('export.csv') # now you can use export.csv

dataset=pd.read_csv('export.csv')
print(dataset.shape)
print(dataset)

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(dataset['y'])

import numpy as np

dataset.shape

target=dataset['y']
X=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107', 'X108', 'X109', 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120', 'X121', 'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134', 'X135', 'X136', 'X137', 'X138', 'X139', 'X140', 'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147', 'X148', 'X149', 'X150', 'X151', 'X152', 'X153', 'X154', 'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161', 'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178' ]

#variable = dataset[X]
#lda = LDA(n_components=2)
#sne = TSNE(n_components=2)

from sklearn.preprocessing import StandardScaler
#features = ['X53','X54','X59','X60','X76','X77','X83','X84','X85','X86','X88','X114','X115','X116','X117','X123','X143','X144']
# Separating out the features
x = dataset.loc[:, X].values
# Separating out the target
y = dataset.loc[:,['y']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#print(x)

for i in range(len(y)):
    if y[i] == 1:
        y[i] = 1
    else:
        y[i] = 0

print(y)

#Data Preprocessing
np.random.seed(42)

x=pd.DataFrame(x)
y=pd.DataFrame(y)
#print(x)
y.columns = [178]
print(y)

import pandas as pd
z = [x, y]
dataset = pd.concat(z, axis=1)
print(dataset)

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
target = dataset[178]
#print(target)
one_hot = to_categorical(target, num_classes = 2)
#print(one_hot)
print(dataset.shape)

TrainSet, TestSet = train_test_split(dataset, test_size=0.2, random_state=42)

xTrain = TrainSet.iloc[:, :-1]
yTrain = TrainSet.iloc[:,-1]
yTrain = to_categorical(yTrain, num_classes = 2)
print(xTrain.shape)
print(yTrain.shape)

xTest = TestSet.iloc[:, :-1]
yTest = TestSet.iloc[:,-1]
yTest = to_categorical(yTest, num_classes = 2)
print(xTest.shape)
print(yTest.shape)

print(dataset.shape)
print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape, )

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

from sklearn import decomposition
dataset.shape

pca = decomposition.PCA(n_components=4)

X_std_pca = pca.fit_transform(xTrain)

X_std_pca.shape

X_std_pca



#Building time series model
model = Sequential()

#First layer
model.add(Dropout(0.2,  input_shape = (178,)))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))

#2nd layer
#model.add(Dense(80, activation = 'relu'))
#model.add(Dropout(0.2))

#3rd layer
#model.add(Dense(80, activation = 'relu'))
#model.add(Dropout(0.2))

#4th layer
#model.add(Dense(output_dim = 10, activation = 'relu'))

#5th layer
model.add(Dense(2, activation = 'softmax'))

#Compile
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Summary
print(model.summary())

history = model.fit(xTrain, yTrain, validation_split = 0.33, epochs=150, batch_size=20)

#target = model.predict(xTest)
score = model.evaluate(xTest, yTest, batch_size = 20)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print(model.metrics_names)
print(score)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(xTest, yTest, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

history = loaded_model.fit(xTrain, yTrain, validation_split = 0.33, epochs=100, batch_size=20)

