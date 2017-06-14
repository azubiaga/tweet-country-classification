from __future__ import print_function
import numpy as np
from scipy.sparse import csc_matrix
from sklearn import metrics
import sys, os
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import json
import operator
from pprint import pprint
import gensim
import re

def get_top_labels_and_feats(filename, top, topinc):
    featids = {'content': [], 'name': [], 'tz': [], 'ulang': [], 'uloc': []}
    topcountries = {}
    if os.path.exists('feats/topcountries' + str(top)) and os.path.exists('feats/content' + str(topinc)) and os.path.exists('feats/name' + str(topinc)) and os.path.exists('feats/tz' + str(topinc)) and os.path.exists('feats/ulang' + str(topinc)) and os.path.exists('feats/uloc' + str(topinc)):
        with open('feats/topcountries' + str(top), 'r') as fh:
            for line in fh:
                data = line.strip().split('\t')
                topcountries[data[0]] = int(data[1])
        with open('feats/content' + str(topinc), 'r') as fh:
            featids['content'] = fh.read().splitlines()
        with open('feats/name' + str(topinc), 'r') as fh:
            featids['name'] = fh.read().splitlines()
        with open('feats/tz' + str(topinc), 'r') as fh:
            featids['tz'] = fh.read().splitlines()
        with open('feats/ulang' + str(topinc), 'r') as fh:
            featids['ulang'] = fh.read().splitlines()
        with open('feats/uloc' + str(topinc), 'r') as fh:
            featids['uloc'] = fh.read().splitlines()
    else:
        countries = {}
        feats = {'content': {}, 'name': {}, 'tz': {}, 'ulang': {}, 'uloc': {}}
        linecount = 0
        with open(filename, 'r') as fh:
            for line in fh:
                linecount += 1
                tweet = json.loads(line)

                cc = tweet['coordinates']['location']['address']['country_code']
                countries[cc] = 1 + countries.get(cc, 0)

                for ctoken in re.sub(r'([^\s\w]|_)+', '', tweet['text'].lower()).split(' '):
                    if ctoken != 'rt' and not 'http' in ctoken:
                        feats['content'][ctoken] = 1 + feats['content'].get(ctoken, 0)

                for nametoken in re.sub(r'([^\s\w]|_)+', '', tweet['user']['name'].lower()).split(' '):
                    feats['name'][nametoken] = 1 + feats['name'].get(nametoken, 0)

                feats['tz'][str(tweet['user']['time_zone'])] = 1 + feats['tz'].get(str(tweet['user']['time_zone']), 0)

                feats['ulang'][tweet['user']['lang']] = 1 + feats['ulang'].get(tweet['user']['lang'], 0)

                for loctoken in re.sub(r'([^\s\w]|_)+', '', tweet['user']['location'].lower()).split(' '):
                    feats['uloc'][loctoken] = 1 + feats['uloc'].get(loctoken, 0)

                print('Reading features: ' + str(linecount), end='\r')
        print('Reading features: ' + str(linecount), end='\n')

        cfeatids = sorted(feats['content'].items(), key=operator.itemgetter(1), reverse=True)
        inc = 0
        for feat, count in cfeatids:
            inc += 1
            if inc <= topinc:
                featids['content'].append(feat)

        namefeatids = sorted(feats['name'].items(), key=operator.itemgetter(1), reverse=True)
        inc = 0
        for feat, count in namefeatids:
            inc += 1
            if inc <= topinc:
                featids['name'].append(feat)

        featids['tz'] = feats['tz'].keys()

        featids['ulang'] = feats['ulang'].keys()

        ulocfeatids = sorted(feats['uloc'].items(), key=operator.itemgetter(1), reverse=True)
        inc = 0
        for feat, count in ulocfeatids:
            inc += 1
            if inc <= topinc:
                featids['uloc'].append(feat)

        with open('feats/content' + str(topinc), 'wb') as fw:
            for f in featids['content']:
                fw.write(f + '\n')

        with open('feats/name' + str(topinc), 'wb') as fw:
            for f in featids['name']:
                fw.write(f + '\n')

        with open('feats/tz' + str(topinc), 'wb') as fw:
            for f in featids['tz']:
                fw.write(f + '\n')

        with open('feats/ulang' + str(topinc), 'wb') as fw:
            for f in featids['ulang']:
                fw.write(f + '\n')

        with open('feats/uloc' + str(topinc), 'wb') as fw:
            for f in featids['uloc']:
                fw.write(f + '\n')

        scountries = sorted(countries.items(), key=operator.itemgetter(1), reverse=True)
        ccount = 0
        with open('feats/topcountries' + str(top), 'wb') as fw:
            for scountry, value in scountries:
                ccount += 1
                if ccount <= top:
                    topcountries[scountry] = ccount
                    fw.write(scountry + '\t' + str(ccount) + '\n')

    return topcountries, featids

def build_matrix(filename, featureids, featurelengths, toplabels, istest = 0):
    colcount = featurelengths['content'] + featurelengths['name'] + featurelengths['tz'] + featurelengths['ulang'] + featurelengths['uloc']

    linecount = 0
    rows = []
    cols = []
    gt = []
    values = []
    with open(filename, 'r') as fh:
        for line in fh:
            tweet = json.loads(line)
            if istest == 1 or tweet['coordinates']['location']['address']['country_code'] in toplabels:
                for ctoken in list(set(re.sub(r'([^\s\w]|_)+', '', tweet['text'].lower()).split(' '))):
                     if ctoken in featureids['content']:
                         rows.append(linecount)
                         cols.append(featureids['content'][ctoken])
                         values.append(1)

                for nametoken in list(set(re.sub(r'([^\s\w]|_)+', '', tweet['user']['name'].lower()).split(' '))):
                    if nametoken in featureids['name']:
                        rows.append(linecount)
                        cols.append(featureids['name'][nametoken] + featurelengths['content'])
                        values.append(1)

                if str(tweet['user']['time_zone']) in featureids['tz']:
                    rows.append(linecount)
                    cols.append(featureids['tz'][str(tweet['user']['time_zone'])] + featurelengths['content'] + featurelengths['name'])
                    values.append(1)

                if tweet['user']['lang'] in featureids['ulang']:
                    rows.append(linecount)
                    cols.append(featureids['ulang'][tweet['user']['lang']] + featurelengths['content'] + featurelengths['name'] + featurelengths['tz'])
                    values.append(1)

                for loctoken in list(set(re.sub(r'([^\s\w]|_)+', '', tweet['user']['location'].lower()).split(' '))):
                    if loctoken in featureids['uloc']:
                        rows.append(linecount)
                        cols.append(featureids['uloc'][loctoken] + featurelengths['content'] + featurelengths['name'] + featurelengths['tz'] + featurelengths['ulang'])
                        values.append(1)

                if istest == 0:
                    gt.append(toplabels[tweet['coordinates']['location']['address']['country_code']])
                else:
                    gt.append(0)

                linecount += 1
            print('Loading data: ' + str(linecount), end='\r')
            sys.stdout.flush()
    print('Loading data: ' + str(linecount), end='\n')

    # TODO: tlang

    row = np.asarray(rows)
    col = np.asarray(cols)
    data = np.asarray(values)

    return (csc_matrix((data, (row, col)), shape=(linecount, colcount)), gt)

if len(sys.argv) < 4:
    print("No dataset and classifier specified.")
    sys.exit()

testdata = sys.argv[1]
classifier = sys.argv[2]
output = sys.argv[3]
top = int(sys.argv[4])
topinc = int(sys.argv[5])

if not os.path.exists(testdata):
    print("Dataset not found.")
    sys.exit()

trainfile = 'training-data.json'

toplabels, featids = get_top_labels_and_feats(trainfile, top, topinc)
featureids = {'content': {}, 'name': {}, 'tz': {}, 'ulang': {}, 'uloc': {}}
featurelengths = {'content': 0, 'name': 0, 'tz': 0, 'ulang': 0, 'uloc': 0}
for ftype in ['content', 'name', 'tz', 'ulang', 'uloc']:
    fid = 0
    for feat in featids[ftype]:
        featureids[ftype][feat] = fid
        fid = len(featureids[ftype])
    featurelengths[ftype] = len(featureids[ftype])

if not os.path.exists('models/' + classifier + '-' + str(top) + '-' + str(topinc) + '.pkl'):
    train_data, train_gt = build_matrix(trainfile, featureids, featurelengths, toplabels)

    print("#1 Train data loaded.")

    if classifier == "svm":
        cmodel = LinearSVC()
    if classifier == "nb":
        cmodel = GaussianNB()
    if classifier == "knn":
        cmodel = KNeighborsClassifier()
    if classifier == "decisiontree":
        cmodel = DecisionTreeClassifier()
    if classifier == "randomforest":
        cmodel = RandomForestClassifier()
    if classifier == "maxent":
        cmodel = LogisticRegression()

    cmodel.fit(train_data, train_gt)
    _ = joblib.dump(cmodel, 'models/' + classifier + '-' + str(top) + '-' + str(topinc) + '.pkl', compress=9)
    print("#2 Model created.")
else:
    cmodel = joblib.load('models/' + classifier + '-' + str(top) + '-' + str(topinc) + '.pkl')

test_data, test_gt = build_matrix(testdata, featureids, featurelengths, toplabels, 1)

print("#3 Test data loaded.")

predicted = cmodel.predict(test_data)
print("#4 Classification done.")

key = 0
allgt = 0
matchinggt = 0
with open(testdata, 'r') as fh, open(output, 'wb') as fw:
    for line in fh:
        tweet = json.loads(line)
        tweet['country'] = list(toplabels.keys())[list(toplabels.values()).index(predicted[key])]
        fw.write(json.dumps(tweet) + '\n')
        key += 1

        if 'place' in tweet and tweet['place'] != None and 'country_code' in tweet['place']:
            allgt += 1
            if tweet['country'].lower() == tweet['place']['country_code'].lower():
                matchinggt += 1

print('Accuracy: ' + str(matchinggt) + '/' + str(allgt) + ' -- ' + str(float(matchinggt) / allgt))
