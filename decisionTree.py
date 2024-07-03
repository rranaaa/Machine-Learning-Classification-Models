import pandas as pd , numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from data import load_and_preprocess_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

"""
  Decision Tree Classifier
"""
def run_decision_tree():
    # Get The Training Data (Exists In data,py)
    x_train , y_train , x_test , y_test = load_and_preprocess_data()

    # [1] Build The Model
    # We Take Instance From tree.DecisionTreeClassifier Class
    # fit()  This Method Used For Training The Decision Tree Classifier Model
    # (Accepting The Training Data For The Features And It's Target(Class))

    model = tree.DecisionTreeClassifier().fit(x_train , y_train)


    # [2] Testing The Model
    predicted_y = model.predict(x_test)

    # [3] Let's Compare Between The Ground Truth And Predicted Value
    # The First Way (Comparing One By One)
    test_predict = predicted_y + y_test
    print(test_predict.head(20))

    # Second Way (train_test_split)
    # How This Function Works ?
    # The function iterates through the corresponding elements .
    # For each pair of elements (Ground Truth label and predicted label), it checks if they are equal.
    # If It Is Equal Then Count_correct_prediction By 1
    # Calculate Accuracy By Dividing The  Count_correct_prediction By The Total Number Of Samples
    accuracy = accuracy_score(y_test, predicted_y)
    print("=" * 40)
    print("Accuracy of the Decision Tree Classifier:", accuracy , " --> " , accuracy * 100 , "%") # 0.7899327186643409
    # We Can Improve The Accuracy By Adding More Samples For Training Or Splitting The Test Data to
    # 15% For Validation And 15% For Testing To Prevent Overfitting
    print("=" * 40)

    # Third Way: Let's Take One Random Sample
    # Record Number (13064)
    # [178.305	115.879	3.5584	0.173	0.0925	-151.135	-170.381	-52.7482	53.1597	224.667] , class = h
    list = [178.305	,115.879,	3.5584,	0.173,	0.0925,	-151.135,	-170.381,	-52.7482,	53.1597	,224.667]
    list = pd.DataFrame(np.resize(np.array(list) , (1,10)))
    # Give The Data Feature Names
    list.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
    print(model.predict(list)) # ==> ['h']

    # Confusion Matrix
    print("="*40)
    print("Confusion Matrix")
    confusion_mat = confusion_matrix(y_test, predicted_y)
    print(confusion_mat)
    print("="*40)

    # Calculate Precision And recall 
    print("="*40)
    print("Report")
    report = classification_report(y_test, predicted_y)
    print(report)
    print("=" * 40)


    '''
    Cross Validation to evaluate the performance
    Step-by-step explanation of how the data is split into 5 folds:
        a)The entire dataset is divided into 5 equal-sized subsets.
        b)In the first iteration, the first fold is used as the validation set, and the remaining 4 folds are used for training.
        c)In the second iteration, the second fold is used as the validation set, and the other 4 folds are used for training.
        d)This process continues for the remaining folds until all 5 folds have been used as the validation set once.
        e)The performance of the model is evaluated on each fold, and the results are typically averaged to obtain an overall performance measure.
    '''
    print("="*40)
    dt_cv = cross_val_score(model, x_train , y_train, cv=5)  # cv=5 specifies 5 fold cross validation
    print("Decision Tree Cross-Validation Accuracy: {:.4f}%".format(dt_cv.mean() * 100))
    print("="*40)