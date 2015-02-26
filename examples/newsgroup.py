from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)

print len(newsgroups_train.data), len(newsgroups_test.data)

from sklearn import metrics
from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN

tokenizer = Tokenizer(min_df=10, max_features=50000)
X_train = tokenizer.fit_transform(newsgroups_train.data)
X_test  = tokenizer.transform(newsgroups_test.data)
Y_train = newsgroups_train.target
Y_test  = newsgroups_test.target

print tokenizer.n_features

layers = [
    Embedding(size=128, n_features=tokenizer.n_features),
    GatedRecurrent(size=256, activation='tanh', gate_activation='steeper_sigmoid',
    			   init='orthogonal', seq_output=False),
    Dense(size=1, activation='sigmoid', init='orthogonal') # sigmoid for binary classification
]

model = RNN(layers=layers, cost='bce') # bce is classification loss for binary classification and sigmoid output
for i in range(2):
    model.fit(X_train, Y_train, n_epochs=1)
    tr_preds = model.predict(X_train[:len(Y_test)])
    te_preds = model.predict(X_test)

    tr_acc = metrics.accuracy_score(Y_train[:len(Y_test)], tr_preds > 0.5)
    te_acc = metrics.accuracy_score(Y_test, te_preds > 0.5)

    print i, tr_acc, te_acc # dataset too small to fully utilize Passage

save(model, 'model.pkl')