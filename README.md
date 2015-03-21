**Passage**
===================
A little library for text analysis with RNNs.

Warning: very alpha, work in progress.

## Install

via Github (version under active development)
```
git clone http://github.com/IndicoDataSolutions/passage.git
python setup.py develop
```
or via pip
```
sudo pip install passage
```

## Example
Using Passage to do binary classification of text, this example:

* Tokenizes some training text, converting it to a format Passage can use.
* Defines the model's structure as a list of layers.
* Creates the model with that structure and a cost to be optimized.
* Trains the model for one iteration over the training text.
* Uses the model and tokenizer to predict on new text.
* Saves and loads the model.

```
from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import save, load

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
save(model, 'save_test.pkl')
model = load('save_test.pkl')
```

Where: 

* train_text is a list of strings ['hello world', 'foo bar']
* train_labels is a list of labels [0, 1]
* test_text is another list of strings

## Datasets

Without sizeable datasets RNNs have difficulty achieving results better than traditional sparse linear models. Below are a few datasets that are appropriately sized, useful for experimentation. Hopefully this list will grow over time, please feel free to propose new datasets for inclusion through either an issue or a pull request.

**__Note__**: __None of these datasets were created by indico, nor should their inclusion here indicate any kind of endorsement__

Blogger Dataset: http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip (Age and gender data)

