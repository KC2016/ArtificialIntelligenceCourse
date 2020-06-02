import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree # NEW, was missing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydotplus as pydot
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image 
from sklearn.metrics import accuracy_score # NEW, was missing
import itertools # NEW, was missing

# download the dataset
import urllib.request
    
filename = 'dataset_einstein.csv'
relativPath = 'data'

url = 'https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/ArvoreDecis%C3%A3o/' + filename
urllib.request.urlretrieve(url, relativPath+'/'+filename)

# import the dataset to a dataframe
df = pd.read_csv(relativPath+'/'+'dataset_einstein.csv', delimiter=';')

# show the 5 first line
# print(df.head(5))

count_row = df.shape[0]  # count rows: recorded patients
count_col = df.shape[1]  # count columns
# print(count_row)
# print(count_col)

# remove rows with at least one blank cell
df = df.dropna()

# print(df.head(5))
print('Number of cells(colunms): ', df.shape[1])
print('Total of recorded patients:', df.shape[0])

# check if the database is balanced or unbalanced
print('Total of recorded patients with negative results: ', df[df['SARS-Cov-2 exam result'] =='negative'].shape[0])
print('Total of recorded patients with positive results: ', df[df['SARS-Cov-2 exam result'] =='positive'].shape[0])

# IMPLEMENTATION
print('''We need to convert the Dataframe to an Numpy Array, which is the type of data
that we will use in the training. 
We will also already separate the Dataset in two. 
One with input features, and one with labels (tags, record labels).

In this case, we are trying to make a classifier for the Covid test, in this case, 
we want to train our model with the tag present in the field 'SARS-Cov-2 exam result''')

# create the stickers for Y
Y = df['SARS-Cov-2 exam result'].values 
print(Y)

# X will be our matrix with features
# let's take the training fields (hemoglobin, leukocytes, basophils, c-reactive protein mg / dl)

X = df[['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']].values
print(X)

# divide our Dataset in two: one for training (80% of data) and another for testing (20% of data)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=3)

print('''Now let's create and train our model.
Remember the difference between algortimo and model? 
There is the training algorithm (which in this case is the decision tree) that 
will export a trained model (which is also an algorithm).''')

print('''now on my_tree i have associated with her the training algorithm,
basically the recipe we see in the theory.''')

#create an algorithm that will be of the type of decision tree
algortimo_arvore = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5) 
# max_deth = 5 to avoid overfitting

# train the model
modelo = algortimo_arvore.fit(X_treino, Y_treino)   # model in whitebox

print(''' The decision tree can be considered a White Box model, that is,
a model that we can better understand what he learned and how he decides. 
We can show the tree for that.''')

# show the most important feature (white box?)
print(modelo.feature_importances_)

nome_features = ['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']
nome_classes = modelo.classes_

# VISUALIZATION
from sklearn.externals.six import StringIO 
from IPython.display import Image
import pydotplus as pydot

# assemble the tree image
dot_data = StringIO()
# dot_data = tree.export_graphviz(my_tree_one, out_file=None, feature_names=featureNames)
export_graphviz(modelo, out_file=dot_data, filled=True, feature_names=nome_features, class_names=nome_classes, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png(relativPath+'/'+'arvore.png')
Image(relativPath+'/'+'arvore.png')   # it should show the image

# alternatively
from subprocess import call
call(['open', relativPath+'/'+'arvore.png'])

print('''And we can also understand which features are most important for the trained model''')

importances = modelo.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
# plt.show()

print('''
#Indice das features
# 0 - 'Hemoglobin', 
# 1 - 'Leukocytes'
# 2 - 'Basophils',
# 3 - 'Proteina C reativa mg/dL']
''')

print('''We will test the model by making predictions in the test dataset.''')

# applying the model to the test base and storing the result in y_predicoes
Y_predicoes = modelo.predict (X_teste)

# model evaluation
# let's assess the dataset y_teste's real value with predictions
print("TREE ACCURACY:", accuracy_score(Y_teste, Y_predicoes))
print(classification_report (Y_teste, Y_predicoes))

print('''accuracy: of the ratings the model made for a certain class, how many effectively were correct?
recall: of the possible datapoints belonging to a certain class, how many the model was the classified 
correctly?''')

print('''Let's understand the Confusion Matrix''')

print('''This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.''')

print('''A rercall of positives in 0.25 é too low. There are many false positives in the model.''')

def plot_confusion_matrix(cm, classes,        
                          normalize=False,
                          title='Matrix de Confusao',
                          cmap=plt.cm.Blues):        #color map used to specify colors
    plt.imshow(cm, interpolation='nearest', cmap=cmap) #numpy array generating the image
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Current label')
    plt.xlabel('Predicted label')
    plt.show()  # I needed to include this to plot
   

matrix_confusao = confusion_matrix(Y_teste, Y_predicoes)
plt.figure() 
plot_confusion_matrix(matrix_confusao, classes=nome_classes,
                      title='Matrix de Confusao')
