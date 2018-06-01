## IMPORTS. YA GOTTA DO IT. YA JUST GOTTA.

import flask
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pickle

#---------- MODEL IN MEMORY ----------------#


# Read the data on pitch rates,
# Build a LogisticRegression predictor on it

#pitches = pd.read_csv("pitchpredictApp").drop('Unnamed: 0',1)
#
#y = pitches['hit_type']
#X = pitches.drop('hit_type',1)

#
#categorical_features_indices = np.where(X.dtypes != np.float)[0]
#Cat = CatBoostClassifier(learning_rate=0.1, loss_function = 'Logloss', class_weights = [1,4])


with open("Baseball_pickle2.sav", 'rb') as picklefile:
    PREDICTOR = pickle.load(picklefile)
#PREDICTOR = Cat.fit(X, y, cat_features=categorical_features_indices);

#PREDICTOR = LogisticRegression().fit(X,y, class_weight = {0:1,1:5})

### THIS IS JUST A CLASSIC MODEL FIT WITH YOUR DATA!
### LOAD THE CSV... ASSIGN COLUMNS ETC. FIT TO YA MODEL YA MODEL!


#---------- URLS AND WEB PAGES -------------#

# Initialize the app

app = flask.Flask(__name__)

# HOMEPAGE
# Opening up awesomeness.htmlself.
# Hey world, Here's the html!
#At app is a wrapper that the system knows.

##Now we're just reading the HTML. Simple!

@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("pitches.html", 'r') as viz_file:
        return viz_file.read()


# Get an example and return it's score from the predictor model
#HERE'S HOW TO RESPOND TO A REQUEST
@app.route("/score", methods=["POST"])
def score():

    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json

    x = data["example"]
    #x is the predicted inputs

    #x = (data["example"])
    score = PREDICTOR.predict_proba([x])
    # score = round((score[0,0]),4)
    print(score)


    ### NEED TO FIGURE OUT HOW TO GET BUTTON RESULTS INTO A THING THAT CAN BE PREDICTED!!!
    ### FIGURED IT OUT! "data" (in the other file)... will be a dictionary with "example" as it's key. The values will be
    ### the list of values (in order!) It will then turn it into a matrix, find the proba, and send back the estimation.
    ### HOW DO WE GET IT TO SAVE UNSELECTED BUTTONS AS INPUTS??? ... I think we have to make a variable for every fricken thing...
    ### And the variables will default as zero and we'll always return all of them and only if it's clicked will it switch to 1...
    ### Then our example will have everything in it. And it can be matrixed, and probaed, and sent back as a result.

    ### You could also maybe try Catboost...??? Not sure if it's worth it. YESS!!! SEEMS LIKE CATBOOST IS WORTH IT! WAY SIMPLER!

    ## Now we're predicting the probability of x
    # Put the result in a nice dict (called results) so we can send it as json

    results = {"score": score[0,0]}
    print(results)
    return flask.jsonify(results)


    ## Turning this dictionary into json

        ##For "example"... need to figure out how to add the other dummies back?
        ## Just so you know you'll probably need this...
        #PD.to_json(orient = 'records')

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
