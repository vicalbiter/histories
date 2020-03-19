# Base libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Libraries for ROC plots
from scipy import interp
from sklearn.metrics import roc_curve, auc

# Libraries for Cross Validation
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold

class ProbModel:
    
    ##### FUNCTIONS FOR CLUSTERIZATION #####

    #Function to clusterize categories of a certain feature, and add the new clusterized feature as a new column
    # Clusters should be an input of the form {cluster_A: {categories}, cluster_B: [categories]}
    # Ex. obesity = {0:[1,2,3], 1:[4,5,6]}
    def clusterize_discrete(self, feature, clusters, new_name, data):
        new_data = data.copy()
        original_list = data.loc[data.index[0]:data.index[-1], feature]
        new_list = []
        for index in original_list.index:
            cat = False
            if original_list[index] == -1 or original_list[index] == "-1":
                new_list.append("N")
                continue
            for cluster in clusters:
                if original_list[index] in clusters[cluster]:
                    new_list.append(cluster)
                    cat = True
            if cat == False:
                new_list.append("N")
        new_data[new_name] = new_list
        return new_data

    # Function to clusterize categories of a certain continous feature, and add the new clusterized feature as a
    # new column
    # Clusters shoud be an input of the form {cluster_A: {lambdaFunction1}, cluster_B: lambdaFunction2}
    def clusterize_continuous(self, feature, clusters, new_name, data):
        new_data = data.copy()
        original_list = data.loc[data.index[0]:data.index[-1], feature]
        new_list = []
        for index in original_list.index:
            if original_list[index] == -1 or original_list[index] == "-1":
                new_list.append("N")
                continue
            for cluster in clusters:
                if eval("self." + clusters[cluster])(original_list[index]):
                    new_list.append(cluster)
                    break
        new_data[new_name] = new_list
        return new_data

    # Set of auxiliary high-order functions that will evaluate the conditions to binarize a history
    def less_than(self, num):
        return lambda n: n < num

    def lesseq_than(self, num):
        return lambda n: n <= num

    def greater_than(self, num):
        return lambda n: n > num

    def greatereq_than(self, num):
        return lambda n: n >= num

    def between(self, num1, num2):
        return lambda n: n <= num2 and n >= num1
    
    
    ##### FUNCTIONS FOR GROUPING VARIABLES #####

    # Function to build a composite random variable from several variables (to create a history, for example)
    def group_list_of_variables(self, list_of_features, new_name, data):
        new_data = data.copy()
        new_list = []
        i = 0
        for feature in list_of_features:
            buffer_list = data.loc[data.index[0]:data.index[-1], feature]
            if i == 0:
                for index in buffer_list.index:
                    new_list.append(str(buffer_list[index]))
                    i = i + 1
            else:
                k = 0
                for index in buffer_list.index:
                    new_list[k] = new_list[k] + str(buffer_list[index])
                    k = k + 1
        new_data[new_name] = new_list
        return new_data
    
    ##### FUNCTIONS FOR COUNTING / CALCULATING PROBABILITIES #####
    
    # Functions to count ocurrences for a category of a feature (NX)

    # Count the number of instances inside the database whose feature X = category
    # Ex: get_nx('AIMC', 3, fdata)
    def get_nx(self, feature, category, data):
        count = 0
        for index in data.index:
            if data.loc[index][feature] == category:
                count = count + 1
            elif self.match_structure(data.loc[index][feature], category):
                count = count + 1
        return count

    # Count the number of instances inside the database whose feature X_1 = category_1 and feature X_2 = category_2
    # Ex: get_ncx('AIMC', 3, 'Obesidad', 1, fdata)
    def get_ncx(self, feature_1, category_1, feature_2, category_2, data):
        count = 0
        for index in data.index:
            if data.loc[index][feature_1] == category_1 and data.loc[index][feature_2] == category_2:
                count = count + 1
            elif self.match_structure(data.loc[index][feature_1], category_1) and data.loc[index][feature_2] == category_2:
                count = count + 1
            elif self.match_structure(data.loc[index][feature_1], category_1) and self.match_structure(data.loc[index][feature_2], category_2):
                count = count + 1
            elif data.loc[index][feature_1] == category_1 and self.match_structure(data.loc[index][feature_2], category_2):
                count = count + 1
        return count

    # Get a conditional probability P(F_1 = C_1 | F_2 = C_2)
    def get_cond_prob(self, feature_1, category_1, feature_2, category_2, data):
        ncx = self.get_ncx(feature_1, category_1, feature_2, category_2, data)
        nx = self.get_nx(feature_2, category_2, data)
        if nx != 0:
            p = ncx / float(nx)
        else:
            p = 0
        #return {'P': p, 'nx': nx, 'ncx': ncx}
        return p
    
    
    # Get the epsilon of a feature-category in relation to a classFeature-classCategory
    def get_epsilon(self, feature, category, classFeature, classCategory, data):
        n = len(data)
        nx = self.get_nx(feature, category, data)
        nc = self.get_nx(classFeature, classCategory, data)
        ncx = self.get_ncx(feature, category, classFeature, classCategory, data)
        
        # Without smoothing
        # if n != 0 and nx != 0:
        #     pc = nc / float(n)
        #     pcx = ncx / float(nx)
        #     epsilon = nx * (pcx - pc) / math.sqrt(nx * pc * (1 - pc))
        # else:
        #     epsilon = 0

        # With smoothing
        pc = (nc + 1) / (float(n) + 2)
        pcx = (ncx + 1) / (float(nx) + 2)
        epsilon = nx * (pcx - pc) / math.sqrt(nx * pc * (1 - pc))
        
        #print 'Epsilon :' + str(epsilon)
        return {'feat': feature, 'cat': category, 'class': classFeature, 'classcat': classCategory, 'epsilon': epsilon, 'nx': nx, 'ncx': ncx, 'nc': nc}


    ##### FUNCTIONS FOR CALCULATING EPSILONS #####

    # Get the epsilons for all the categories in a single feature, in relation to a classFeature-classCategory
    def get_all_epsilons(self, feature, classFeature, classCategory, data):
        catlist = self.get_categories(feature, data)
        epsilons = []
        for category in catlist:
            epsilons.append(self.get_epsilon(feature, category, classFeature, classCategory, data))
        return pd.DataFrame(epsilons)

    # Get a list of all the categories that a single feature has in the dataset
    def get_categories(self, feature, data):
        index_list = data.loc[data.index[0]:data.index[-1], feature]
        categories = {}
        for index in index_list.index:
            categories[index_list[index]] = "1"
        return categories.keys()

    # Get the epsilons of a list of history patterns (categories), in relation to a classFeature-classCategory
    def get_epsilons_from_feature_and_catlist(self, feature, categories, classFeature, classCategory, data):
        epsilons = []
        for category in categories:
            epsilons.append(self.get_epsilon(feature, category, classFeature, classCategory, data))
        return epsilons

    # Get the epilons of a list of history patterns (histories), in relation to all the categories in classFeature 
    def get_epsilons_from_feature_and_catlist(self, feature, histories, classFeature, data):
        catlist = self.get_categories(classFeature, data)
        epsilons = []
        for category in catlist:
            epsilons = epsilons + self.get_epsilons_from_feature_and_catlist(feature, histories, classFeature, category, data)
        return pd.DataFrame(epsilons)
    
    
    ##### AUXILIARY FUNCTIONS #####
    
    # Function to determine if a certain history matches a general pattern
    def match_structure(self, history, structure):
        if type(history) is str: 
            if len(history) != len(structure):
                return False
            for i in range(len(structure)):
                if history[i] != structure[i] and structure[i] != '*':
                    return False
            return True
        else:
            return False

    def string_generator(self, string, seeds, num):
        strings = []
        if num == 0:
            return string
        else:
            for seed in seeds:
                strings.append(string_generator(string + seed, seeds, num - 1))
        return strings

    def string_padding(self, pre_padding, strings, post_padding):
        new_strings = []
        for string in strings:
            new_strings.append(pre_padding + string + post_padding)
        return new_strings

class BayesianModel(ProbModel):
    
    # Get the score of a single feature-category in relation to a classFeature-classCategory
    def get_score(self, feature, category, classFeature, classCategory, data):
        n = len(data)
        nx = self.get_nx(feature, category, data)
        nc = self.get_nx(classFeature, classCategory, data)
        ncx = self.get_ncx(feature, category, classFeature, classCategory, data)
        
        # Without smoothing
        # pxc = ncx / float(nc)
        # pxnc = (nx - ncx) / float(n - nc)
        # if pxc != 0 and pxnc != 0:
        #     score = math.log(pxc/pxnc)
        # else:
        #     score = 0
    
        # With smoothing
        pxc = (ncx + 1) / float(nc)
        pxnc = (nx - ncx + 1) / float(n - nc + 2)
        score = math.log(pxc/pxnc)
        return score

    
    ##### Score calculation, given a feature or a list of features #####
    ##### Only the features are explicitely stated; the categories are automatically extracted from the data ######
    
    # Get all the scores from all the feature-cagetory combinations of a list of features
    def get_scores_from_featlist(self, list_of_features, classFeature, classCategory, data):
        scores = {}
        for feature in list_of_features:
            scores[feature] = self.get_scores_from_feature(feature, classFeature, classCategory, data)
        return scores
    
    # Get all the scores of the list of categories associated to a single feature
    def get_scores_from_feature(self, feature, classFeature, classCategory, data):
        catlist = self.get_categories(feature, data)
        scores = {}
        for category in catlist:
            scores[category] = self.get_score(feature, category, classFeature, classCategory, data)
        return scores
    
    
    ###### Score calculation, given a list of features and categories #####
    ###### Both the features and the categories are explicitly stated #####
    
    
    # Get all the scores associated to the feature and category combinations in the "queries" object
    # The "queries" object shoud have the following structure:
    # queries = {"feature1": ["category1", "category2" ..., "categoryM"],
    #            "feature2": ["category1", "category2" ..., "categoryN"],
    #             .
    #             .
    #             .
    #            "featureX": ["category1", "category2" ..., "categoryL"]
    #           }
    def get_scores_from_featlist_and_catlist(self, queries, classFeature, classCategory, data):
        scores = {}
        for feature in queries:
            scores[feature] = self.get_scores_from_feature_and_catlist(feature, queries[feature], classFeature, classCategory, data)
        return scores

    # Get all the scores of the list of categories associated to a single feature
    def get_scores_from_feature_and_catlist(self, feature, catlist, classFeature, classCategory, data):
        scores = {}
        for category in catlist:
            #scores.append(get_score(feature, category, classFeature, classCategory, data))
            scores[category] = self.get_score(feature, category, classFeature, classCategory, data)
        return scores
    
    
    ###### Scores - Full representation #####
    
    # Get the score of a single feature-category in relation to a classFeature-classCategory
    def get_score_full(self, feature, category, classFeature, classCategory, data):
        n = len(data)
        nx = self.get_nx(feature, category, data)
        nc = self.get_nx(classFeature, classCategory, data)
        ncx = self.get_ncx(feature, category, classFeature, classCategory, data)
    
        # Without smoothing
    #   pxc = ncx / float(nc)
    #   pxnc = (nx - ncx) / float(n - nc)
    #     if pxc != 0 and pxnc != 0:
    #         score = math.log(pxc/pxnc)
    #     else:
    #         score = 0
        
        # With smoothing
        pxc = (ncx + 1) / float(nc)
        pxnc = (nx - ncx + 1) / float(n - nc + 2)
        score = math.log(pxc / pxnc)
        
        return {'feat': feature, 'cat': category, 'class': classFeature, 'classcat': classCategory, 'score': score, 'nx': nx, 'ncx': ncx, 'nc': nc}


    # Get all the scores from all the feature-cagetory combinations of a list of features
    def get_scores_from_featlist_full(self, list_of_features, classFeature, classCategory, data):
        scores = []
        for feature in list_of_features:
            scores = scores + self.get_scores_from_feature_full(feature, classFeature, classCategory, data)
        return pd.DataFrame(scores)
    
    # Get all the scores of the list of categories associated to a single feature
    def get_scores_from_feature_full(self, feature, classFeature, classCategory, data):
        catlist = self.get_categories(feature, data)
        scores = []
        for category in catlist:
            scores.append(self.get_score_full(feature, category, classFeature, classCategory, data))
        return scores
    
    def get_scores_from_featlist_and_catlist_full(self, queries, classFeature, classCategory, data):
        scores = []
        for feature in queries:
            scores = scores + self.get_scores_from_feature_and_catlist_full(feature, queries[feature], classFeature, classCategory, data)
        return pd.DataFrame(scores)

    # Get all the scores of the list of categories associated to a single feature
    def get_scores_from_feature_and_catlist_full(self, feature, catlist, classFeature, classCategory, data):
        scores = []
        for category in catlist:
            #scores.append(get_score(feature, category, classFeature, classCategory, data))
            scores.append(self.get_score_full(feature, category, classFeature, classCategory, data))
        return scores
    
    
    ##### Auxiliary functions for the actual implementation of the above functions in a classification model
    
    # Use the scores dictionary to replace the individual scores in the original data
    def get_scores_from_featlist_per_user(self, query_features, scores_dictionary, data):
        user_data = data[query_features]
        for index in user_data.index:
            for col in user_data.columns:
                category = user_data.at[index, col]
                feature = col
                try:
                    user_data.at[index, col] = scores_dictionary[feature][category]
                except:
                    print("The score for a feature-category combination was not set - It may have not been in the training set")
                    print("The score for index " + str(index) + " will be set to 0")
                    user_data.at[index, col] = 0
        return user_data
    
    def get_scores_from_featlist_and_catlist_per_user(self, queries, scores_dictionary, data):
        query_features = queries.keys()
        user_data = data[query_features]
        for index in user_data.index:
            for col in user_data.columns:
                category = user_data.at[index, col]
                feature = col
                try:
                    user_data.at[index, col] = scores_dictionary[feature][category]
                except:
                    print("The score for a feature-category combination was not set - It may have not been in the training set")
                    print("The score for index " + str(index) + " will be set to 0")
                    user_data.at[index, col] = 0
        return user_data

class NBA(BayesianModel):
    
    # Class constructor
    def __init__(self):
        self.scores_dictionary = {}
        self.predicted_scores = {}
        self.query_features = {}
        self.classFeature = ""
        self.classCategory = ""
    
    # Train the model, given certain query_features, a classFeature-classCategory and some training data Xt
    def train(self, query_features, classFeature, classCategory, Xt):
        # Get the dictionary of scores, in relation to classFeature-classCategory
        self.query_features = query_features
        self.classFeature = classFeature
        self.classCategory = classCategory
        self.scores_dictionary = self.get_scores_from_featlist(self.query_features, self.classFeature, self.classCategory, Xt)
    
    # Use the scores dictionary of the already trained model to classify the test data Xv
    def predict(self, Xv):

        # Use the dictionary of scores to calculate the associated sum of scores for every user
        user_scores = self.get_scores_from_featlist_per_user(self.query_features, self.scores_dictionary, Xv)
        
        # Store all the score-related information in a dataframe
        sum_scores = pd.DataFrame(user_scores.sum(axis=1))
        user_scores = user_scores.join(sum_scores)
        user_scores = user_scores.rename(columns={0: "total_score"})
        self.predicted_scores = user_scores.copy()
    
    # Use the scores dictionary of the already trained model to predict the scores of the test data Xv
    def get_predicted_scores(self):
        # Return the predicted scores from the self.predicted_scores table
        return pd.DataFrame(self.predicted_scores["total_score"])
    
    # Use the scores dictionary of the already trained model to perform a classification of the test data Xv
    def get_predicted_labels(self):
        # Evaluate whether the total_score > 0. Classify as 1 if total_score > 0, or 0 if total_score <= 0
        labels = pd.DataFrame(self.predicted_scores.eval('total_score > 0').replace(True, 1).replace(False, 0))
        label_name = "predicted_" + self.classFeature
        return labels.rename(columns={0: label_name})

class GNB(BayesianModel):
    
    # Class constructor
    def __init__(self):
        self.scores_dictionary = {}
        self.predicted_scores = {}
        self.queries = {}
        self.classFeature = ""
        self.classCategory = ""
    
    # Train the model, given certain query_features, a classFeature-classCategory and some training data Xt
    def train(self, queries, classFeature, classCategory, Xt):
        # Get the dictionary of scores, in relation to classFeature-classCategory
        self.queries = queries
        self.classFeature = classFeature
        self.classCategory = classCategory
        self.scores_dictionary = self.get_scores_from_featlist_and_catlist(self.queries, self.classFeature, self.classCategory, Xt)
    
    # Use the scores dictionary of the already trained model to classify the test data Xv
    def predict(self, Xv):

        # Use the dictionary of scores to calculate the associated sum of scores for every user
        user_scores = self.get_scores_from_featlist_and_catlist_per_user(self.queries, self.scores_dictionary, Xv)
        
        # Store all the score-related information in a dataframe
        sum_scores = pd.DataFrame(user_scores.sum(axis=1))
        user_scores = user_scores.join(sum_scores)
        user_scores = user_scores.rename(columns={0: "total_score"})
        self.predicted_scores = user_scores.copy()
    
    # Use the scores dictionary of the already trained model to predict the scores of the test data Xv
    def get_predicted_scores(self):
        # Return the predicted scores from the self.predicted_scores table
        return pd.DataFrame(self.predicted_scores["total_score"])
    
    # Use the scores dictionary of the already trained model to perform a classification of the test data Xv
    def get_predicted_labels(self):
        # Evaluate whether the total_score > 0. Classify as 1 if total_score > 0, or 0 if total_score <= 0
        labels = pd.DataFrame(self.predicted_scores.eval('total_score > 0').replace(True, 1).replace(False, 0))
        label_name = "predicted_" + self.classFeature
        return labels.rename(columns={0: label_name})

class Validation:
    
    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__
        
    # Run a CV of the GNB model and perform a decile analysis of its results
    def run_cv(self, X, qF, cF, cC, folds, plot_name):

        kf = KFold(n_splits = 5, shuffle=True)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        sum_actual_tps = np.repeat(0.0, 10)
        sum_expected_tps = np.repeat(0.0, 10)

        j = 0
        for train_index, test_index in kf.split(X):
            train = pd.DataFrame()
            test = pd.DataFrame()
            for index in train_index:
                name = X.iloc[index].name
                train = train.append(X.loc[name])
            for index in test_index:
                name = X.iloc[index].name
                test = test.append(X.loc[name])

            self.model.train(qF, cF, cC, train)
            self.model.predict(test)
            Yprob = self.model.get_predicted_scores()

            fpr, tpr, thresholds = roc_curve(test[cF].values, Yprob.values[:, -1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold %d (AUC=%0.2f)" % (j + 1, roc_auc))
            j = j + 1

            # Decile analysis
            scores = Yprob.copy()
            scores[cF] = test[cF]
            bins = self.split_scores(scores, 10)
            actual_tps = self.get_decile_frequencies(bins, cF)      
            expected_tps = self.get_expected_tps(scores, cF, 10)

            sum_actual_tps += actual_tps
            sum_expected_tps += expected_tps

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, lw=2, color='b', alpha=0.8, label='Mean ROC (AUC=%0.2f)' % (mean_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(plot_name)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
               label='Random', alpha=.8)
        plt.legend(loc="lower right")
        plt.show()

        # Plot the results of the decile analysis
        mean_actual_tps = sum_actual_tps / (folds * 1.0)
        mean_expected_tps = sum_expected_tps / (folds * 1.0)
        self.plot_decile_bars(mean_actual_tps, mean_expected_tps, 10)
        
    # Function to plot the score distribution of the classified data, by deciles
    def plot_score_distribution(self, scores, classFeature, num_bins):
        # Sort by score and split the data into buckets (deciles)
        bins = self.split_scores(scores, num_bins)

        # Get the number of true positives by decile
        frequencies = self.get_decile_frequencies(bins, classFeature)

        # Get the expected value of true positives by decile, if the classification was made randomly
        expected = self.get_expected_tps(scores, classFeature, num_bins)

        # Plot the results
        self.plot_decile_bars(frequencies, expected, num_bins)

    # Split the scores dataframe in bins
    def split_scores(self, scores, num_bins):
        results = scores.copy()
        sorted_results = results.sort_values(by=["total_score"], ascending=False)
        bins = np.array_split(sorted_results, num_bins)
        return bins

    # Get the count of true positives for every bin
    def get_decile_frequencies(self, bins, classFeature):
        frequencies = []
        for binid in range(len(bins)):
            frequencies.append(bins[binid][classFeature].sum())
        return frequencies

    # Get the expected count of true positives per decile
    def get_expected_tps(self, scores, classFeature, num_bins):
        expected = np.repeat(scores[classFeature].sum(), num_bins) * 1.0 / num_bins
        return expected

    # Plot the results
    def plot_decile_bars(self, frequencies, expected, num_bins):
        plt.bar(np.arange(1, num_bins + 1, 1), frequencies, color='b', label="Actual True Positives")
        plt.plot(np.arange(1, num_bins + 1, 1), expected, color='r', linestyle='dashed', label="Expected True Positives")
        plt.legend(loc="upper right")
        plt.xlabel("Score Decile")
        plt.ylabel("# of True Positives")
        plt.title("True Positives Distribution")
        plt.show()

