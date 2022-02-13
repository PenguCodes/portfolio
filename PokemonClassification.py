###
# This project was made in Python by Juan Camilo. #
###
# As a kid I always liked Pokemon and as a somewhat competitive person, I want to beat my fellow pokemon trainers
# in battle. For me to be able to that, I need to find the Pokemon that are most likely to be used professional battles
# before they do! This piece of code takes pokemon data of current professional pokemon in the OU and Ubers tier of play
# from generations 1-7 and tries to classify generation 8 into "OU" which is the assumption I am using for competitive
# play. Without further talk, lets dive into the world of competitive Pokemon!
#
# As a side note, you will need the pokemon_all.csv file from the GitHub and have both the files in the same folder
# for the code to work!
#
###

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import warnings
import openpyxl

###
# Above we are importing all our libraries we will use for our analysis!#
###

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

###
# Above we are setting some options for our analysis. I am telling python to ignore some of the warnings it might
# tell me about and for the Pandas dataframes to display all columns instead of truncating them when the dataframes
#dont fit in one screen
###

pokemon_all = pd.read_csv("pokemon_all.csv")
pokemon_all = pokemon_all.reset_index()

###
# Above we read our data set and reset its index, as some of the pokemon numbers are repeated to accommodate different
# forms of the same pokemon. Different Pokemon forms have wildly different statistics and thus I decided to treat them
# independently of each other
###

print(pokemon_all.dtypes)

###
# The dtypes method lets us take a look at each of our columns data types and also tells us our columns!
# By some miracle our columns all seem to be in their correct data type, although we do need to eliminate some of them!
###

pokemon_all = pokemon_all.drop(["No", "Height", "Weight", "Color", "Gender_Male", "Gender_Female", "Gender_Unknown",
                                "Base_EXP", "Final_EXP", "Ef_HP", "Ef_Attack", "Ef_Defense", "Ef_SP_Attack",
                                "Ef_SP_Defense", "Ef_Speed", "Egg_Steps", "Egg_Group1", "Egg_Group2", "Capture_Rate",
                                "Total", "index", "Original_Name", "Name", "index"], axis=1)

###
# All the above categories do not influence the competitive viability of a pokemon. Their egg group, gender and bonuses
# given when defeated do not affect their probability of being competitive.
###

pokemon_test = pokemon_all
pokemon_train = pokemon_all
pokemon_test = pokemon_test[pokemon_test["Generation"] == 8]
pokemon_train = pokemon_train[pokemon_all["Generation"] < 8]

###
# The above code separates our data into training and testing. Pokemon of the newest generation go into testing and the
# older pokemon go into testing.
###

print(pokemon_train.columns.tolist())
print(pokemon_test.columns.tolist())

###
# Just checking that our columns are the same going forward in order to avoid dimension errors!
###

pokemon_test = pokemon_test.reset_index()

###
# Just checking that our columns are the same going forward in order to avoid dimension errors!
###


print(pokemon_test.head())
print(pokemon_train.head())


###
# It seems our dataframes are looking good for now, except...
###


pokemon_all = pokemon_all.drop(["Generation"], axis=1)
pokemon_test = pokemon_test.drop(["Generation"], axis=1)
pokemon_train = pokemon_train.drop(["Generation"], axis=1)


###
# Looking better now without the generation column!
###

############################################### Try the drop here!


print(pokemon_test.columns)
print(pokemon_all.columns)
print(pokemon_train.columns)

###
# Just checking our columns yet again, make sure we are not having any errors!
###


def get_missings(df):

    labels, values = list(), list()
    if pokemon_all.isna().sum().sum() > 0:
        for column in df.columns:
            if df[column].isnull().sum():
                labels.append(column)
                values.append((df[column].isnull().sum() / len(df[column])) * 100)
        # Make a dataframe
        missing_information = pd.DataFrame({'Features': labels, 'MissingPercent': values}).sort_values(
            by='MissingPercent', ascending=False)
        plt.figure(figsize=(10, 7))
        sns.barplot(x=missing_information.Features, y=missing_information.MissingPercent).set_title(
            'The Percentage of Missing Values is:')
        return missing_information
    else:
        return False


###
# Above is a function to create a dataframe and a plot of our missing data points.
# After visualizing it, we can analyze how to better deal with our missing values
# ###

print(get_missings(pokemon_all))
plt.show()

###
# We can see that we are missing some values, we will fill them with 0s as it just means we have no value for those
# observations
# ###

pokemon_all = pokemon_all.fillna("0")
pokemon_test = pokemon_test.fillna("0")
pokemon_train = pokemon_train.fillna("0")
pokemon_all["Mega_Evolution _Flag"] = pokemon_all["Mega_Evolution _Flag"].replace({"Mega": 1}).astype(int)
pokemon_test["Mega_Evolution _Flag"] = pokemon_test["Mega_Evolution _Flag"].replace({"Mega": 1}).astype(int)
pokemon_train["Mega_Evolution _Flag"] = pokemon_train["Mega_Evolution _Flag"].replace({"Mega": 1}).astype(int)

###
# Filling up the missing values and replacing the text in our Mega Evoluion flag into an integer
# ###

print(pokemon_all[:].corrwith(pokemon_train['OU']).sort_values(ascending=False))

###
# Just checking our correlations and seeing which features are best correlated with our target! Special Attack and
# regular Attack seem to have the highest correlation with OU!
# ###

print(pokemon_all.dtypes)

###
# Checking to see that all our columns can be used by our model. We still have some object(text) dtypes, that we need to
# convert
# ##

pokemon_all_copy = pd.get_dummies(pokemon_all, columns=["Type1", "Type2", "Region_Forme", "Category","Ability1", "Ability2", "Ability_Hidden"])
pokemon_test_copy = pd.get_dummies(pokemon_test, columns=["Type1", "Type2", "Region_Forme", "Category", "Ability1", "Ability2", "Ability_Hidden"])
pokemon_train_copy = pd.get_dummies(pokemon_train, columns=["Type1", "Type2", "Region_Forme", "Category","Ability1", "Ability2", "Ability_Hidden"])

###
# Filling up our dataframes with pokemon types, formes, categories and ability dummies. All these were categorical and
# now are numerical, which is what we can use!
# ###

print(pokemon_train_copy.columns.tolist())
print(pokemon_test_copy.columns.tolist())

###
# Checking one last time that our columns are the same!###

pokemon_test_copy = pokemon_test_copy.drop(["index"], axis=1)

###
# We can see that some dummy columns do not exist in our test data, as those combinations do not exist in it. One way
# to deal with it is to create a new dataframe that contains all combinations and run the analysis on that frame!
###

pokemon_test_new = pd.DataFrame(pokemon_test_copy, columns=pokemon_train_copy.columns)

###
# We created the dataframe mentioned before, now lets check that our columns are the finally the same!
###

print(pokemon_train_copy.columns.tolist())
print(pokemon_test_new.columns.tolist())

###
# Our columns are matched up and ready to go! The new columns should have NaN values, lets see if that is the case.
###

print(pokemon_test_new.sample(5))

###
# It is, lets fill them with 0s as before.
###

pokemon_test_new = pokemon_test_new.fillna("0")

###
# We are set up and ready to go
###

# Independent variables
X = pokemon_train_copy.drop("OU", axis=1)

# Target
y = pokemon_train_copy["OU"]

# Split the train df to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2277)


def get_models():
    models = dict()
    models['LGBMClassifier'] = LGBMClassifier()
    models['LogisticRegression'] = LogisticRegression()
    models['DecisionTree'] = DecisionTreeClassifier(max_depth=8)  # Tuned
    models['RandomForest'] = RandomForestClassifier(max_depth=32)  # Tuned
    models['GradientBoosting'] = GradientBoostingClassifier(max_depth=5)  # Tuned
    models['svc'] = SVC(C=100, gamma=0.001, kernel='sigmoid')  # Tuned
    return models

###
# The above gives us a function to get our models with our specified parameters. Using multiple models can lead to
# better predictions if used properly
###



# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

###
# The above function gives us an idea of how our model(s) perform once run!
###

models = get_models()
results, names = list(), list()

for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

###
# The above loops gets us our models and our results, it seems we have pretty good scores all around!
###

def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('LogisticR', LogisticRegression()))
    level0.append(('LGBMClassifier', LGBMClassifier()))
    level0.append(('DecisionTree', DecisionTreeClassifier(max_depth=8)))
    level0.append(('RandomForest', RandomForestClassifier(max_depth=32)))
    level0.append(('GBoost', GradientBoostingClassifier()))
    level0.append(('svc', SVC(C=100, gamma=0.001, kernel='sigmoid')))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10, n_jobs=-1)
    return model


model = get_stacking()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Validation accuracy : ", accuracy_score(y_pred,y_test))

confusion_ma = confusion_matrix(y_pred, y_test, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_ma, display_labels=model.classes_)
disp.plot(cmap='viridis')
plt.grid(None)
plt.show()

###
# The above gets our predictions, fits our models, gets our validation accuracy and a confusion matrix to boot! It seems
# our model works!
###

predictions = model.predict(pokemon_test_new.drop("OU", axis=1))
sub = pd.DataFrame({"OU Real": pokemon_test_new["OU"], "OU Predicted": predictions})

sub.to_excel("output_pokemon.xlsx", engine="openpyxl")

###
# The above saves our work and saves it to an xlsx file. xlsx is a great format as a lot of users know how to easily
# manipulate it and maybe gather further insights
###

###
# Further steps:
# Although I feel satisfied on this project as a showcase of code, there are still some improvements that could be made
# Items: Competitive Pokemon use items, and these could heavily influence the chance of pokemon being chosen. There are
# a lot of items and it would be a bit tedious albeit not impossible to match every pokemon to every item and see if
# that pokemon with that item is competitive. This could lead to better insights and more predictive power
# Small Sample:
# The quantity of competitive Pokemon to non-competitive ones is a bit lopsided. There are less than 100 competitive
# pokemon to the 1000+ total Pokemon. A method such as synthethic sampling could be used if the small sample becomes
# a concern to tackle.
###

