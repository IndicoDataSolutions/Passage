import numpy as np
import pandas as pd
from lxml import html

from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import Tokenizer

# download data at kaggle.com/c/word2vec-nlp-tutorial/data
 
def clean(texts):
	return [html.fromstring(text).text_content().lower().strip() for text in texts]
 
if __name__ == "__main__":
	tr_data = pd.read_csv('labeledTrainData.tsv', delimiter='\t') 
	trX = clean(tr_data['review'].values)
	trY = tr_data['sentiment'].values

	print("Training data loaded and cleaned.")

	tokenizer = Tokenizer(min_df=10, max_features=100000)
	trX = tokenizer.fit_transform(trX)

	print("Training data tokenized.")

	layers = [
		Embedding(size=256, n_features=tokenizer.n_features),
		GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', seq_output=False, p_drop=0.75),
		Dense(size=1, activation='sigmoid', init='orthogonal')
	]

	model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
	model.fit(trX, trY, n_epochs=10)

	te_data = pd.read_csv('testData.tsv', delimiter='\t')
	ids = te_data['id'].values
	teX = clean(te_data['review'].values)
	teX = tokenizer.transform(teX)
	pr_teX = model.predict(teX).flatten()
	 
	pd.DataFrame(np.asarray([ids, pr_teX]).T).to_csv('submission.csv', index=False, header=["id", "sentiment"])
