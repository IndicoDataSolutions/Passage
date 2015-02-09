from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.model import RNN


train_text = ["this is good", "quite good", "it's bad", "not too bad"]
train_labels = [1,               1,            0,         1            ]
test_text = ["rather good", "quite bad"]

tokenizer = Tokenizer()
train_tokens = tokenizer.fit_transform(train_text)

layers = [
    Embedding(size=128, n_features=tokenizer.n_features),
    GatedRecurrent(size=128),
    Dense(size=1, activation='sigmoid')
]

model = RNN(layers=layers, cost='BinaryCrossEntropy')


model.fit(train_tokens, train_labels)

print model.predict(tokenizer.transform(test_text))
