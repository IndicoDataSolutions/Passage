# Passage
A little library for text analysis with RNNs.

Warning: very alpha, work in progress.

Install
```
python setup.py develop
```

Using Passage to do binary classification of text, this example:

* Tokenizes some training text, converting it to a format Passage can use.
* Defines the model's structure as a list of layers.
* Creates the model with that structure and a cost to be optimized.
* Trains the model for one iteration over the training text.
* Uses the model and tokenizer to predict on new text.

```
from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.model import RNN

tokenizer = Tokenizer()
train_tokens = tokenizer.fit_transform(train_text)

layers = [
	Embedding(size=128, n_features=tokenizer.n_features),
	GatedRecurrent(size=128),
	Dense(size=1, activation='sigmoid')
]

model = RNN(layers=layers, cost='BinaryCrossEntropy')
model.fit(train_tokens, train_labels)

model.predict(tokenizer.transform(test_text))
```

Where: 

* train_text is a list of strings ['hello world', 'foo bar']
* train_labels is a list of labels [0, 1]
* test_text is another list of strings