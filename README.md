# tweet-country-classification

This is code that classifies tweets by country, used for a research project and reported in a publication in the IEEE TKDE journal (see reference below).

The classifier assumes that the input file with tweets includes one JSON-formatted tweet per line. The command to run the classifier is as follows:

python run.classifier.py [input-file] maxent [output-file] 217 10000

This will use the trained model that we provide. Alternatively, you can also train your own model from your training data with country-labelled tweets.

The dataset we used for these experiments and for building the model can be found here:
https://figshare.com/articles/Tweet_geolocation_5m/3168529

(to be completed)

Reference:
Zubiaga, A., Voss, A., Procter, R., Liakata, M., Wang, B., & Tsakalidis, A. (2017). Towards real-time, country-level location classification of worldwide tweets. IEEE Transactions on Knowledge and Data Engineering.
