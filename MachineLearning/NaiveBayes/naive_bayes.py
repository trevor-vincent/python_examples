import pandas
import numpy as np
emails = pandas.read_csv('emails.csv')

def process_email(text):
     return list(set(text.split()))

def make_model():
    # print(emails['words'][0])
    for i in range(len(emails['words'])):
        print("email " + str(i))
        for word in emails['words'][i]:
            if word not in model:
                model[word] = {'spam': 1, 'ham': 1}
            if word in model:
                if emails['spam'][i]:
                    model[word]['spam'] += 1
                else:
                    model[word]['ham'] += 1

def predict_naive_bayes(email):
     words = set(email.split())
     spams = []
     hams = []
     for word in words:
         if word in model:
             spams.append(model[word]['spam'])
             hams.append(model[word]['ham'])
     prod_spams = np.prod(spams)
     prod_hams = np.prod(hams)
     return prod_spams/(prod_spams + prod_hams)
 
emails['words'] = emails['text'].apply(process_email)
model = {}

make_model()
print("done making model")
print(predict_naive_bayes('hi mom how are you'))
