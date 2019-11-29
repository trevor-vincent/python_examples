import csv
import numpy as np
import math
import luigi
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from math import sin, cos, pi

class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Input
        ----------
        tweet_file (Luigi parameter)
            data file containing tweets, geo_coords, sentiment and other data

        Output
        -------
        clean_data.csv
            cleaned tweet file with only rows that contain geo-coords
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def requires(self):
        return None

    def output(self):
        # outputs cleam_data.csv
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
    '''Extracts features/outcome variable in preparation for training a model.

        Input
        ----------
        clean_data.csv    
            The output of CleanDataTask (cleaned tweet file)
        cities.csv
            A file containing cities and geo-coords

        Output
        -------
        features.csv
            An output file containing the training data
            X = a one-hot coded column for each city in "cities.csv"
            y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        cities.pkl
            Convenience file containing list of city names and their geo-coords, stored in a pickle object
    '''

    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')
    cities_pkl = luigi.Parameter(default='cities.pkl')
    sentiment_dict = {"positive": 2, "neutral": 1, "negative": 0}
    
    def requires(self):
        # requires cleaned tweet file
        return CleanDataTask(tweet_file=self.tweet_file)

    def output(self):
        # outputs features.csv
        return luigi.LocalTarget(self.output_file)

    def get_eucl_dist_sqrd(self, lat_lon_1, lat_lon_2):
        '''Computes Euclidean distance squared between points on the Earth

        Input
        ----------
        lat_lon_1 : float array
            Geo-coords [lattitude, longitude] for point 1
        lat_lon_2 : float array
            Geo-coords [lattitude, longitude] for point 2

        Output
        -------
        distance
            Euclidean distance squared between point 1 and 2
            We set the Earth radius = 1 (i.e. length units where the Earth radius = 1)
            See Ref: https://vvvv.org/blog/polar-spherical-and-geographic-coordinates
        '''
        # convert from (lat,lon) to xyz coords using spherical polar transformation (see Ref)
        x1 = cos(lat_lon_1[0]) * cos(lat_lon_1[1])
        y1 = cos(lat_lon_1[0]) * sin(lat_lon_1[1])
        z1 = sin(lat_lon_1[0])
        x2 = cos(lat_lon_2[0]) * cos(lat_lon_2[1])
        y2 = cos(lat_lon_2[0]) * sin(lat_lon_2[1])
        z2 = sin(lat_lon_2[0])
        return (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

    def run(self):
        cities = []
        #open cities.csv and store city names and geo-coords
        with open(self.cities_file, errors='ignore') as csv_file:
            # skip first line
            next(csv_file)
            csv_reader = csv.reader(csv_file, delimiter=',')
            # append city name and geo_coords to cities list
            for row in csv_reader:
                cities.append([row[1], [float(row[4]), float(row[5])]])

        # save cities and geo_coords to a pkl file for convenience
        with open(self.cities_pkl, 'wb') as fid:
            pickle.dump(cities, fid)       

        # create one hot encoding for cities
        one_hot = np.eye(len(cities))

        # initialize distancs array
        dist = np.zeros(len(cities))

        # open cleaned_data.csv for reading and features.csv for writing
        with open(self.input().path, errors='ignore') as input_file, self.output().open('w') as output_file:
            # skip first header line of cleaned_data.csv
            next(input_file) 
            csv_reader = csv.reader(input_file, delimiter=',')
            csv_writer = csv.writer(output_file, delimiter=',')
            for row in csv_reader:
                    # translate {negative,neutral,positive} to integer {0,1,2} from dictionary
                    sent_int = self.sentiment_dict[row[5]]
                    
                    # remove "[]" from string and then split on "," to get the geo-coords from row
                    tweet_latlon = [float(i) for i in ((row[-5])[1:])[:-2].split(", ")]

                    # find closest city to this tweet using Euclidean distance
                    for i in range(len(cities)):
                        dist[i] = self.get_eucl_dist_sqrd(cities[i][1], tweet_latlon)    
                    city_id = dist.argmin()

                    # append one hot encoding for city and sentiment integer to features.csv
                    Xy_row = list(one_hot[city_id, :])
                    Xy_row.append(sent_int)
                    csv_writer.writerow(Xy_row)                    


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive

        Input
        ----------
        features.csv from TrainingDataTask   
            csv file containing feature vectors and labels for training

        Output
        -------
        model.pkl
            pickled classifier trained to predict tweet sentiment from tweet location
    """
    

    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')
    
    def requires(self):
        #requires features.csv
        return TrainingDataTask(tweet_file=self.tweet_file)

    def output(self):
        #outputs model.pkl
        return luigi.LocalTarget(self.output_file)

    def run(self):
        X = []
        y = []
        #open features.csv and store feature vectors and labels in X and y respectively
        with open(self.input().path, errors='ignore') as input_file:
            csv_reader = csv.reader(input_file, delimiter=',')
            for row in csv_reader:
                X.append(row[:-1])
                y.append(row[-1])

        # create training, test datasets
        # we are not worried about accuracy, so X_test and y_test will not be used
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # fit a RandomForest classifier to the training data with 50 trees
        classifier = RandomForestClassifier(n_estimators=50, random_state=0)
        classifier.fit(X_train, y_train)
        
        # save the classifier
        with open(self.output().path, 'wb') as fid:
            pickle.dump(classifier, fid)

            
class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Input
        ----------
        model.pkl from TrainModelTask   
            trained classifier which predicts tweet sentiment based on city

        Output
        -------
        scores.csv
            Four column soutput file containing the probability scores for each city
            Columns are:
            - city name
            - negative probability
            - neutral probability
            - positive probability (sorted lowest to highest)
    """
    
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')
    cities_pkl = luigi.Parameter(default='cities.pkl')
    
    def requires(self):
        #requires model.plk
        return TrainModelTask(tweet_file=self.tweet_file)

    def output(self):
        #outputs scores.csv
        return luigi.LocalTarget(self.output_file)
    
    def run(self):
        # pred = clf.predict_proba(X_test)
        with open(self.input().path, 'rb') as pickle_file:
            classifier = pickle.load(pickle_file)

        #open convenience file containing city info
        with open(self.cities_pkl, 'rb') as pickle_file2:
            cities = pickle.load(pickle_file2)

        #initialize list to hold probability predictions and city name
        predict_out = []
        
        #create one hot encoding matrix for cities
        one_hot = np.eye(classifier.n_features_)
        
        # compute probabilities for each city and append these probabilities along with city name to predict_out list
        for i in range(len(cities)):
            predict_out.append(np.append(cities[i][0],classifier.predict_proba(one_hot[i, :].reshape(1, len(cities) ) ) ) )

        # open scores file and write out probabilities sorted by positive probability for each city            
        with open(self.output_file, 'w') as score_file:
            csv_writer = csv.writer(score_file, delimiter=',')
            csv_writer.writerow(["city_name", "negative_prob", "neutral_prob", "positive_prob"])
            predict_out = sorted(predict_out, key = lambda x: x[3])
            for i in predict_out:
                csv_writer.writerow(i)

if __name__ == "__main__":
    luigi.run()
