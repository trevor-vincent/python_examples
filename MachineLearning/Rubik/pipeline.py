import csv
import numpy as np
import math
import luigi
import pickle        
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# [[104   2   1]
#  [ 22   2   1]
#  [ 35   2   2]]
#               precision    recall  f1-score   support

#            0       0.65      0.97      0.78       107
#            1       0.33      0.08      0.13        25
#            2       0.50      0.05      0.09        39

#     accuracy                           0.63       171
#    macro avg       0.49      0.37      0.33       171
# weighted avg       0.57      0.63      0.53       171

# 0.631578947368421


class CleanDataTask(luigi.Task):
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def run(self):
        with open(self.tweet_file, errors='ignore') as csv_file, \
             self.output().open('w') as cleaned_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_writer = csv.writer(cleaned_file, delimiter=',')
            for row in csv_reader:
                # skip all rows that have no geo-coords or geo-coords = 0,0
                if row[-5] != "" and row[-5] != "[0.0, 0.0]":
                    csv_writer.writerow(row)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

    Output file will have columns corresponding to the training data:
    - X = a one-hot coded column for each city in "cities.csv"
    - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)

    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    cities_1hot_file = luigi.Parameter(default='cities_1hot.pkl')
    output_file = luigi.Parameter(default='features.csv')
    sentiment_dict = {"positive": 2, "neutral": 1, "negative": 0}
    
    def requires(self):
        return CleanDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def get_one_hot_encoding(self):
        cities = []
        with open(self.cities_file, errors='ignore') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    cities.append([row[1], [float(row[4]), float(row[5])]])
                line_count += 1
        return np.eye(len(cities)), cities

    def get_eucl_dist_sqrd(self, latlon1, latlon2):
        '''Computes Euclidean distance between points on the Earth

        Parameters
        ----------
        latlon1 : float array
            Geo-coords for point 1
        latlon2 : float array
            Geo-coords for point 2

        Returns
        -------
        distance
            Euclidean distance between point 1 and 2
            We set the Earth radius = 1 (i.e. length units where the Earth radius = 1)
            See Ref: https://vvvv.org/blog/polar-spherical-and-geographic-coordinates
        '''
        lat1 = latlon1[0]
        lon1 = latlon1[1]
        lat2 = latlon2[0]
        lon2 = latlon2[1]
        x1 = math.cos(lat1) * math.cos(lon1)
        y1 = math.cos(lat1) * math.sin(lon1)
        z1 = math.sin(lat1)
        x2 = math.cos(lat2) * math.cos(lon2)
        y2 = math.cos(lat2) * math.sin(lon2)
        z2 = math.sin(lat2)
        return (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

    def get_closest_city(self, cities, tweet_location):
        min_dist = math.pi  # initialize to otherside of earth
        min_dist_i = -1
        for i in range(len(cities)):
            dist = self.get_eucl_dist_sqrd(cities[i][1], tweet_location)
            if dist < min_dist:
                min_dist = dist
                min_dist_i = i
        return min_dist_i

    def run(self):
        onehot, cities = self.get_one_hot_encoding()
        with open(self.cities_1hot_file, 'wb') as fid:
            pickle.dump(cities, fid)        
        line_count = 0
        with open(self.input().path, errors='ignore') as input_file, self.output().open('w') as output_file:
            csv_reader = csv.reader(input_file, delimiter=',')
            csv_writer = csv.writer(output_file, delimiter=',')
            for row in csv_reader:
                if line_count != 0:
                    sent_int = self.sentiment_dict[row[5]]
                    tweet_latlon = [float(i) for i in (
                        (row[-5])[1:])[:-2].split(", ")]
                    city_id = self.get_closest_city(cities, tweet_latlon)
                    rowoh = list(onehot[city_id, :])
                    rowoh.append(sent_int)
                    # print(rowoh.size)
                    # csv_writer.writerow(np.append(rowoh, [int(sent_int)], 0))
                    csv_writer.writerow(rowoh)
                else:
                    line_count += 1


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickled model.
    """

    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')
    
    def requires(self):
        return TrainingDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        X = []
        y = []
        with open(self.input().path, errors='ignore') as input_file:
            csv_reader = csv.reader(input_file, delimiter=',')
            for row in csv_reader:
                X.append(row[:-1])
                y.append(row[-1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        print(accuracy_score(y_test, predictions))

        
        # save the classifier
        with open(self.output().path, 'wb') as fid:
            pickle.dump(classifier, fid)



            
class ScoreTask(luigi.Task):
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')
    cities_1hot_file = luigi.Parameter(default='cities_1hot.pkl')
    
    def requires(self):
        return TrainModelTask(tweet_file=self.tweet_file)
    
    def run(self):
        # pred = clf.predict_proba(X_test)
        with open(self.input().path, 'rb') as pickle_file:
            classifier = pickle.load(pickle_file)
        with open(self.cities_1hot_file, 'rb') as pickle_file2:
            cities = pickle.load(pickle_file2)
        predict_out = []
            
        one_hot = np.eye(len(cities))

        for i in range(len(cities)):
            predict_out.append(np.append(cities[i][0],classifier.predict_proba(one_hot[i, :].reshape(1, len(cities)))))
            # predict_out.append(np.concatenate([cities[i][0], [1],axis=0))
            # predict_out.append(np.append([cities[i][0]],classifier.predict_proba(one_hot[i, :].reshape(1, 11)))

        with open(self.output_file, 'w') as score_file:
            csv_writer = csv.writer(score_file, delimiter=',')
            csv_writer.writerow(["city_name","negative_prob","neutral_prob","positive_prob"])
            predict_out = sorted(predict_out, key = lambda x: x[3])
            for i in predict_out:
                csv_writer.writerow(i)
            
if __name__ == "__main__":
    luigi.run()
