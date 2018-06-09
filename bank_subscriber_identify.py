# Name      : Ganesa Rukmani Ramdas Pillai
# USC ID    : 9712700659
# email-id  : gpillai@usc.edu
# Term      : Spring 2018

import pandas as pd
import numpy as np
import seaborn as sns
from seaborn import set_style
set_style("darkgrid")
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from prettytable import PrettyTable
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import time
#get_ipython().run_line_magic('matplotlib', 'inline')


# # User-defined functions

# Function    : plot histogram<br />
# Input       : var, x_len, y_len, title, y_label, x_label = "None"<br />
# Output      : None<br /> 
# Description : Plot the histogram of var; also saves as PNG to image folder



def plot_histogram(var, label, x_len, y_len, title, y_label, x_label = "None"):
    fig = plt.figure(figsize = (x_len,y_len))
    ax = fig.gca()
    var.value_counts().plot(kind='bar', width=0.5);
    plt.title(title)
    if(x_label != "None"):
        plt.xlabel(x_label)
    plt.ylabel(y_label)
#     plt.savefig('images/' + title + '.png')
    pd.crosstab(var,label).plot(kind='bar');
    plt.title('Purchase Frequency for ' + x_label);
    if(x_label != "None"):
        plt.xlabel(x_label)
    plt.ylabel('y_label')
#     plt.savefig('images/' + 'crosstab' + title + '.png')
    return


# Function    : standardize_data<br />
# Input       : train, test<br />
# Output      : train, test<br /> 
# Description : Normalize the training and testing data



def standardize_data(train, test):
    test = (test-train.mean())/train.std()
    train = (train-train.mean())/train.std()
    return train, test


# Function    : print_bold<br />
# Input       : data<br />
# Output      : None<br /> 
# Description : Print the string in bold



def print_bold(data):
    print('\033[1m' + str(data) + '\033[0m')
    return


# Function    : plot_roc_curve<br />
# Input       : fpr, tpr, rocauc<br />
# Output      : None<br /> 
# Description : Plot ROC curve



def plot_roc_curve(fpr, tpr, rocauc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rocauc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# Function    : print_metrics<br />
# Input       : title, score_train, score_test, cm, f1, cr<br />
# Output      : None<br /> 
# Description : Print the metrics of a classifier



def print_plot_metrics(title, score_train, score_test, cm, f1, cr, fpr, tpr, rocauc, acc_score, tr_time, ts_time):   
    print_bold(title);
    print("\n Time to train: ", round(tr_time, 5), " seconds");
    print("\n Time to test: ", round(ts_time, 5), " seconds");
    print("\nConfusion Matrix");
    print(pd.DataFrame(cm))
    print("\nTraining Accuracy: ", round(score_train * 100, 2), "%");
    print("\nTesting Accuracy: ", round(score_test * 100, 2), "%");
    print("\nF1 score is: ", round(f1, 5));
    print("\nAccuracy score: ", round(acc_score * 100, 2), "%");
    print("\nClassification Report");
    print(cr);
    print("\nArea Under the Receiver Operating Characteristic Curve: \n", round(rocauc, 5));
    plot_roc_curve(fpr, tpr, rocauc)


# Function    : display_histogram_features<br />
# Input       : csvdata<br />
# Output      : None<br /> 
# Description : Display the histogram of all features of dataset



def display_histogram_features(csvdata):
    plot_histogram(csvdata.age, csvdata.y, 15, 5, "Histogram for Age", "Count", "Age")
    plot_histogram(csvdata.job, csvdata.y, 15, 5, "Histogram for Job", "Count", "Job")
    plot_histogram(csvdata.marital, csvdata.y, 10, 5, "Histogram for Marital", "Count", "Marital")
    plot_histogram(csvdata.education, csvdata.y, 10, 5, "Histogram for education", "Count", "education")
    plot_histogram(csvdata.default, csvdata.y, 5, 5, "Histogram for default", "Count", "default")
    plot_histogram(csvdata.housing, csvdata.y, 5, 5, "Histogram for housing", "Count", "housing")
    plot_histogram(csvdata.loan, csvdata.y, 5, 5, "Histogram for loan", "Count", "loan")
    plot_histogram(csvdata.contact, csvdata.y, 5, 5, "Histogram for contact", "Count", "contact")
    plot_histogram(csvdata.month, csvdata.y, 15, 5, "Histogram for month", "Count", "month")
    plot_histogram(csvdata.day_of_week, csvdata.y, 10, 5, "Histogram for day_of_week", "Count", "day_of_week")
    plot_histogram(csvdata.campaign, csvdata.y, 15, 5, "Histogram for campaign", "Count", "Campaigns")
    plot_histogram(csvdata.pdays, csvdata.y, 10, 5, "Histogram for pdays", "Count", "pdays")
    plot_histogram(csvdata.previous, csvdata.y, 10, 5, "Histogram for previous", "Count", "previous")
    plot_histogram(csvdata.poutcome, csvdata.y, 10, 5, "Histogram for poutcome", "Count", "poutcome")
    plot_histogram(csvdata.emp_var_rate, csvdata.y, 10, 5, "Histogram for emp_var_rate", "Count", "emp_var_rate")
    plot_histogram(csvdata.cons_price_idx, csvdata.y, 15, 5, "Histogram for cons_price_idx", "Count", "cons_price_idx")
    plot_histogram(csvdata.cons_conf_idx, csvdata.y, 15, 5, "Histogram for cons_conf_idx", "Count", "cons_conf_idx")
    plot_histogram(csvdata.euribor3m, csvdata.y, 50, 5, "Histogram for euribor3m", "Count", "euribor3m")
    plot_histogram(csvdata.nr_employed, csvdata.y, 10, 5, "Histogram for nr_employed", "Count", "nr_employed")
    return
    


# Function    : data_cleaning<br />
# Input       : csvdata<br />
# Output      : csvdata<br /> 
# Description : Clean the data of unknown and lesser important data points and features



def data_cleaning(csvdata):
    # Remove unknown from job feature as less than 10% are unknown
    csvdata = csvdata[csvdata.job != 'unknown']
    # Remove unknown from marital feature as only 11 sample points are unknown
    csvdata = csvdata[csvdata.marital != 'unknown']
    # Remove illiterate from education feature as only 1 sample point is illiterate
    csvdata = csvdata[csvdata.education != 'illiterate']
    # Remove yes from education feature as only 1 sample point is yes
    csvdata = csvdata[csvdata.default != 'yes']
    # Remove uknown from housing
    csvdata = csvdata[csvdata.housing != 'unknown']
    # Remove uknown from loan
    csvdata = csvdata[csvdata.loan != 'unknown']
    # Drop pdays feature as it is not a good predictor
    csvdata = csvdata.drop('pdays', axis=1)
    return csvdata


# Function    : split_categorical_strings<br />
# Input       : csvdata<br />
# Output      : dataset<br /> 
# Description : Create dataset by spliting categorical strings



def split_categorical_strings(csvdata):
    # Split Age into three categories and add to dataset
    bins = [0,28,55,120]
    temp = pd.cut(csvdata.age, 3,bins, labels = ["young", "middle", "old"])
    dataset = pd.get_dummies(temp, prefix="age")
    # Split Job into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.job, prefix="job")], axis=1)
    # Split marital into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.marital, prefix="marital")], axis=1)
    # Split education into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.education, prefix="education")], axis=1)
    # Split default into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.default, prefix="default")], axis=1)
    # Split housing into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.housing, prefix="housing")], axis=1)
    # Split loan into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.loan, prefix="loan")], axis=1)
    # Split contact into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.contact, prefix="contact")], axis=1)
    # Split month into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.month, prefix="month")], axis=1)
    # Split day_of_week into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.day_of_week, prefix="day")], axis=1)
    # Split campaign into dummies
    dataset = pd.concat([dataset, csvdata.campaign], axis=1)
    # Split previous into dummies
    dataset = pd.concat([dataset, csvdata.previous], axis=1)
    # Split poutcome into dummies
    dataset = pd.concat([dataset, pd.get_dummies(csvdata.poutcome, prefix="poutcome")], axis=1)
    # Split emp_var_rate into dummies
    dataset = pd.concat([dataset, csvdata.emp_var_rate], axis=1)
    # Split cons_price_idx into dummies
    dataset = pd.concat([dataset, csvdata.cons_price_idx], axis=1)
    # Split cons_conf_idx into dummies
    dataset = pd.concat([dataset, csvdata.cons_conf_idx], axis=1)
    # Split euribor3m into dummies
    dataset = pd.concat([dataset, csvdata.euribor3m], axis=1)
    # Split nr_employed into dummies
    dataset = pd.concat([dataset, csvdata.nr_employed], axis=1)
    return dataset


# Function    : extract_label_from_dataset<br />
# Input       : csvdata<br />
# Output      : label<br /> 
# Description : Extract the labels from the dataset



def extract_label_from_dataset(csvdata):
    label = pd.DataFrame(csvdata.y, columns=['y'])
    label.rename(columns={'y': 'Class'}, inplace=True)
    label = (label == 'yes').astype(int)
    return label


# Function    : balanced_test_sample<br />
# Input       : features, label, subsample_size<br />
# Output      : X_train, X_test, y_train, y_test<br /> 
# Description : Create balanced test data



def balanced_test_sample(features, label, subsample_size = 1.0):
    class_counts = label.Class.value_counts()
    min_class_id = class_counts.idxmin()
    max_class_id = class_counts.idxmax()
    min_class_num = class_counts.min()
    max_class_num = class_counts.max()
    
    label_min = pd.DataFrame(np.ones((min_class_num,), dtype=int), columns=['Class']) * min_class_id
    min_class = dataset[label.Class == min_class_id]
    
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(min_class, label_min['Class'], test_size=subsample_size, random_state=100)
    
    label_max = pd.DataFrame(np.ones((max_class_num,), dtype=int), columns=['Class']) * max_class_id
    max_class = dataset[label.Class == max_class_id]
    
    max_subsample_ratio = (min_class_num * subsample_size) / max_class_num;
    X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(max_class, label_max['Class'], test_size=max_subsample_ratio, random_state=100)
    
    
    X_train = pd.concat([X_train_min, X_train_max], axis=0)
    y_train = pd.concat([y_train_min, y_train_max], axis=0)
    X_test  = pd.concat([X_test_min, X_test_max], axis=0)
    y_test  = pd.concat([y_test_min, y_test_max], axis=0)
    
    X_train, y_train = shuffle(X_train, y_train)
    X_test , y_test  = shuffle(X_test, y_test)  
    
    return X_train, X_test, y_train, y_test
    


# Class: classifier_info<br />
# Description : The classifier_info class has all methods to classify and print metrics 



class classifier_info:
    
    def __init__(self, clf, X_train, X_test, y_train, y_test, title):
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.title = title
        
    def classify(self):
        start = time.clock()
        self.clf.fit(self.X_train, self.y_train);
        time_diff = time.clock() - start
        self.train_time = time_diff;
        self.train_score = self.clf.score(self.X_train, self.y_train)
        self.test_score = self.clf.score(self.X_test, self.y_test)
        start = time.clock()
        self.y_pred = self.clf.predict(self.X_test)
        time_diff = time.clock() - start
        self.test_time = time_diff;
        self.acc_score = metrics.accuracy_score(self.y_test, self.y_pred);
        self.cr = classification_report(self.y_test, self.y_pred)
        self.cm = metrics.confusion_matrix(self.y_test, self.y_pred)
        self.f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        self.fpr, self.tpr, self.threshold = metrics.roc_curve(self.y_test, self.y_pred)
        self.rocauc = metrics.auc(self.fpr, self.tpr)

        
    def metrics(self, PRINT_FLAG="yes"):
        print_plot_metrics(self.title, self.train_score, self.test_score, self.cm, self.f1, self.cr, self.fpr, self.tpr, self.rocauc, self.acc_score, self.train_time, self.test_time)
        if(PRINT_FLAG == "yes"):
            x.add_row([self.title, str(round(self.train_time, 5)) + " s", str(round((self.train_score * 100), 2)) + "%", str(round(self.test_time, 5)) + " s", str(round((self.test_score * 100), 2)) + "%" , str(round(self.rocauc, 5)), str(round(self.f1, 5))])
    
        
    
    
    
    


# # Reading the csv file



csvdata = pd.read_csv('bank-additional.csv')
csvdata.rename(columns={'emp.var.rate': 'emp_var_rate', 'cons.price.idx': 'cons_price_idx', 'cons.conf.idx' : 'cons_conf_idx', 'nr.employed' : 'nr_employed'}, inplace=True)
orig_csvdata = csvdata
csvdata.head()


# # Plot histogram of all features



#display_histogram_features(csvdata)


# # Data Cleaning



csvdata = data_cleaning(csvdata)


# # Data segregation of categorical strings



dataset = split_categorical_strings(csvdata)


# # Create the label dataframe



label = extract_label_from_dataset(csvdata)


# # Split the dataset into training and testing partitions



X_train, X_test, y_train, y_test = train_test_split(dataset, label['Class'], test_size=0.10, random_state=100)
X_train = X_train.copy()
X_test = X_test.copy()


# # Normalize the X_train and X_test



column_norm_str=["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed", "campaign", "previous"]
column_norm_str
for i in column_norm_str:
    X_train[i], X_test[i] = standardize_data(X_train[i], X_test[i])


# # Initialize the Pretty Table



x = PrettyTable()
x.field_names = ["Classifier", "Training Time", "Training Score", "Testing Time", "Testing Score", "ROC-AUC", "F1-score"]


# # 1. Basic Classifiers

# # 1.1 KNN Classification



clf = KNeighborsClassifier(n_neighbors=3)
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "KNN");
ci.classify();
ci.metrics();


# # 1.2 SVM Classification



clf = svm.SVC(kernel = 'linear', C=1)
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "SVM");
ci.classify();
ci.metrics();


# # 1.3 Naive Bayesian Classifier



clf = GaussianNB()
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "Naive Bayesian");
ci.classify();
ci.metrics();


# # 1.4 MLP Classifier



clf = MLPClassifier(hidden_layer_sizes=(57,57,57), max_iter=500)
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "MLP");
ci.classify();
ci.metrics();


# # 1.5 Decision Tree Classifier



clf = DecisionTreeClassifier(random_state=0, max_depth=2)
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "Decision Tree");
ci.classify();
ci.metrics();


# # 1.6 Random Forest Classifier



clf = RandomForestClassifier()
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "Random Forest");
ci.classify();
ci.metrics();


# # 1.7 Logistic Regression Classifier



clf = LogisticRegression()
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "Logistic Regression");
ci.classify();
ci.metrics();


# # 2. Other tricks to improve performance

# # 2.1 Dimensionality reduction with PCA 



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
X_train_pca = pd.DataFrame(principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalComponents = pca.transform(X_test)
X_test_pca = pd.DataFrame(principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# # 2.1.1 Random Forest Classifier with PCA



clf = RandomForestClassifier()
ci = classifier_info(clf, X_train_pca, X_test_pca, y_train, y_test, "Random Forest (PCA)");
ci.classify();
ci.metrics();


# # 2.1.2 Naive Bayesian Classifier with PCA



clf = GaussianNB()
ci = classifier_info(clf, X_train_pca, X_test_pca, y_train, y_test, "Naive Bayesian (PCA)");
ci.classify();
ci.metrics();


# # 2.1.3 MLP Classifier with PCA



clf = MLPClassifier(hidden_layer_sizes=(57,57,57), max_iter=500)
ci = classifier_info(clf, X_train_pca, X_test_pca, y_train, y_test, "MLP (PCA)");
ci.classify();
ci.metrics();


# # 2.2 Feature Selection using SelectKBest



fs = SelectKBest(f_classif, k=11)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train) 
X_test_fs = fs.transform(X_test)


# # 2.2.1 SVM Classifier with SelectKBest



clf = svm.SVC(kernel = 'linear', C=1)
ci = classifier_info(clf, X_train_fs, X_test_fs, y_train, y_test, "SVM (SelectKBest)");
ci.classify();
ci.metrics();


# # 2.2.2 Naive Bayesian Classifier with SelectKBest



clf = GaussianNB()
ci = classifier_info(clf, X_train_fs, X_test_fs, y_train, y_test, "Naive Bayesian (SelectKBest)");
ci.classify();
ci.metrics();


# # 2.2.3 MLP Classifier with SelectKBest



clf = MLPClassifier(hidden_layer_sizes=(57,57,57), max_iter=500)
ci = classifier_info(clf, X_train_fs, X_test_fs, y_train, y_test, "MLP (SelectKBest)");
ci.classify();
ci.metrics();


# # 2.2.4 Random Forest Classifier with SelectKBest



clf = RandomForestClassifier()
ci = classifier_info(clf, X_train_fs, X_test_fs, y_train, y_test, "Random Forest (SelectKBest)");
ci.classify();
ci.metrics();


# # 2.3 Training Dataset oversampling with SMOTE



method = SMOTE(kind='svm')
X_train_smote, y_train_smote = method.fit_sample(X_train, y_train)


# # 2.3.1 SVM Classifier with SMOTE



clf = svm.SVC(kernel = 'linear', C=1)
ci = classifier_info(clf, X_train_smote, X_test, y_train_smote, y_test, "SVM (SMOTE)");
ci.classify();
ci.metrics();


# # 2.3.2 Naive Bayesian Classifier with SMOTE



clf = GaussianNB()
ci = classifier_info(clf, X_train_smote, X_test, y_train_smote, y_test, "Naive Bayesian (SMOTE)");
ci.classify();
ci.metrics();


# # 2.3.3 MLP Classifier with SMOTE



clf = MLPClassifier(hidden_layer_sizes=(57,57,57), max_iter=500)
ci = classifier_info(clf, X_train_smote, X_test, y_train_smote, y_test, "MLP (SMOTE)");
ci.classify();
ci.metrics();


# # 2.3.4 Random Forest Classifier  with SMOTE



clf = RandomForestClassifier()
ci = classifier_info(clf, X_train_smote, X_test, y_train_smote, y_test, "Random Forest (SMOTE)");
ci.classify();
ci.metrics();


# # 2.4 Feature Selection using SelectKBest and training data oversampling with SMOTE



method = SMOTE(kind='svm')
X_train_fs_smote, y_train_fs_smote = method.fit_sample(X_train_fs, y_train)


# # 2.4.1 SVM Classifier with SelectKBest and SMOTE



clf = svm.SVC(kernel = 'linear', C=1)
ci = classifier_info(clf, X_train_fs_smote, X_test_fs, y_train_fs_smote, y_test, "SVM (SelectKBest + SMOTE)");
ci.classify();
ci.metrics();


# # 2.4.2 Naive Bayesian Classifier with SelectKBest and SMOTE



clf = GaussianNB()
ci = classifier_info(clf, X_train_fs_smote, X_test_fs, y_train_fs_smote, y_test, "Naive Bayesian (SelectKBest + SMOTE)");
ci.classify();
ci.metrics();


# # 2.4.3 MLP Classifier with SelectKBest and SMOTE



clf = MLPClassifier(hidden_layer_sizes=(57,57,57), max_iter=500)
ci = classifier_info(clf, X_train_fs_smote, X_test_fs, y_train_fs_smote, y_test, "MLP (SelectKBest + SMOTE)");
ci.classify();
ci.metrics();


# # 2.4.4 Random Forest Classifier with SelectKBest and SMOTE



clf = RandomForestClassifier()
ci = classifier_info(clf, X_train_fs_smote, X_test_fs, y_train_fs_smote, y_test, "Random Forest (SelectKBest + SMOTE)");
ci.classify();
ci.metrics();


# # 2.5 RFE for logistic regression



clf = LogisticRegression()
rfe = RFE(clf, 18)
rfe = rfe.fit(X_train, y_train)
print("[RFE] Features support")
print(rfe.support_)
print("[RFE] Features Ranking")
print(rfe.ranking_)
i, = np.where(rfe.support_ == 1)
X_train_rfe = X_train.iloc[:, i]
X_test_rfe = X_test.iloc[:,i]


# # 2.5.1 Logistic Regression classifier with RFE



clf = LogisticRegression()
ci = classifier_info(clf, X_train_rfe, X_test_rfe, y_train, y_test, "Logistic Regression (RFE)");
ci.classify();
ci.metrics();


# # 2.6 Feature Selection using SelectKBest, balanced test set extract, and training data oversampled with SMOTE



X_train_bal, X_test_bal, y_train_bal, y_test_bal = balanced_test_sample(dataset, label, 0.3)
fs = SelectKBest(f_classif, k=11)
fs.fit(X_train_bal, y_train_bal)
X_train_bal_fs = fs.transform(X_train_bal) 
X_test_bal_fs = fs.transform(X_test_bal)
method = SMOTE(kind='svm')
X_train_bal_fs_smote, y_train_bal_fs_smote = method.fit_sample(X_train_bal_fs, y_train_bal)


# # 2.6.1 SVM Classifier with SelectKBest, balanced test set, SMOTE



clf = svm.SVC(kernel = 'linear', C=1)
ci = classifier_info(clf, X_train_bal_fs_smote, X_test_bal_fs, y_train_bal_fs_smote, y_test_bal, "SVM (SelectKBest + Balanced Test + SMOTE)");
ci.classify();
ci.metrics();


# # 2.6.2 Naive Bayesian Classifier with SelectKBest, balanced test set, SMOTE



clf = GaussianNB()
ci = classifier_info(clf, X_train_bal_fs_smote, X_test_bal_fs, y_train_bal_fs_smote, y_test_bal, "Naive Bayesian (SelectKBest + Balanced Test + SMOTE)");
ci.classify();
ci.metrics();


# # 2.6.3 MLP Classifier with SelectKBest, balanced test set, SMOTE



clf = MLPClassifier(hidden_layer_sizes=(57,57,57), max_iter=500)
ci = classifier_info(clf, X_train_bal_fs_smote, X_test_bal_fs, y_train_bal_fs_smote, y_test_bal, "MLP (SelectKBest + Balanced Test + SMOTE)");
ci.classify();
ci.metrics();


# # 2.6.4 Random Forest Classifier with SelectKBest, balanced test set, SMOTE



clf = RandomForestClassifier()
ci = classifier_info(clf, X_train_bal_fs_smote, X_test_bal_fs, y_train_bal_fs_smote, y_test_bal, "Random Forest (SelectKBest + Balanced Test + SMOTE)");
ci.classify();
ci.metrics();


# # 2.7 SVM Classifier with class weights



clf = svm.SVC(kernel = 'linear', C=1, class_weight='balanced')
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "SVM (Class Weight Balanced)");
ci.classify();
ci.metrics();


# # 2.8 Cross validation using GridSearchCV

# # 2.8.1 KNN classification using GridSearchCV



k = np.arange(20)+1
parameters = {'n_neighbors': k}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn,parameters,cv=10);
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "kNN (GridSearchCV)");
ci.classify();
print("KNN best k: ", clf.best_params_['n_neighbors'])
ci.metrics();


# # 2.8.2 SVM classification using GridSearchCN



parameter_candidates = [
  {'C': [1, 10, 100], 'kernel': ['linear']},
  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 2, 10], 'kernel': ['rbf']},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1);
ci = classifier_info(clf, X_train, X_test, y_train, y_test, "SVM (GridSearchCV)");
ci.classify();
print('[SVM] Best C:',clf.best_estimator_.C) 
print('[SVM] Best Kernel:',clf.best_estimator_.kernel)
print('[SVM] Best Gamma:',clf.best_estimator_.gamma)
ci.metrics();


# # Print the accumulated results



print(x)

