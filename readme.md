# Machine-Learning Tutorial: ML Sentiment Checker

This is a collection of files that I will use to learn more about ML. This specific project is a tool that will let a user check the sentiment of some text as positive or negative, etc.

## Install Dependencies

Below are the steps to get up and running. Run the following in your terminal:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## What will we learn

- Load and prep text data.
- Train a simple ML model with scikit-learn.
- Save the model so you can reuse it later.
- Run predictions in a script.

### Step 1: Install Needed Libraries

```
pip install scikit-learn pandas

```

### Step 2: Dataset

We need text labeled as positive/negative. For simplicity we'll start with a tiny hand-made CSV.

```
text,label
"I love this movie, it was fantastic!",positive
"Terrible film. Waste of time.",negative
"Absolutely brilliant acting.",positive
"The plot was boring and predictable.",negative
"Loved the soundtrack!",positive
"Not my type of movie, I didn’t enjoy it.",negative

```

### Step 3: Model Training

#### Step 1: CountVectorizer CountVectorizer() looks at all sentences in the training data and builds a vocabulary:

```
vocabulary = ["i", "love", "this", "movie", "it", "was", "fantastic", "terrible", "film", "waste", "of", "time", "absolutely", "brilliant", "acting", "the", "plot", "boring", "and", "predictable", "loved", "soundtrack", "not", "my", "type", "didn’t", "enjoy"]
```

Then it converts each sentence into a vector of word counts. Example:

```
"I love this movie" → [1, 1, 1, 1, 0, 0, 0, 0, 0, ...]
```

Each position corresponds to a word in the vocabulary. The number is how many times that word appears in the sentence.

#### Step 2: MultinomialNB

MultinomialNB looks at these word counts for each label:

- Positive sentences: it calculates how often each word appears in positive sentences.
- Negative sentences: it calculates how often each word appears in negative sentences.

Then, when it sees a new sentence, it computes the probability that the sentence belongs to each class using Bayes’ theorem.

- Words like "love" and "fantastic" push the probability toward positive.
- Words like "boring" and "waste" push it toward negative.

#### Step 3: Pipeline in Action

model.fit(X, y) CountVectorizer learns the vocabulary and converts all training sentences into vectors. MultinomialNB trains on those vectors. model.predict(["I love this movie"]) The pipeline automatically: Converts "I love this movie" → vector using the same vocabulary. Feeds it to MultinomialNB. Returns "positive".

### Step 4: Using The Model

To test the model:

```
python3 train.py
python3 predict.py
```


Expected Output:

```
What a wonderful film! -> positive
This was the worst experience ever. -> negative
```

There is always a chance for errors to occur. This due to needing more data to fine-tune the training of the model.


### Step 5: Improving the model

1. More training data - Even a few hundred labeled sentences will improve accuracy drastically.
2. Text preprocessing - Lowercasing, removing punctuation, removing stopwords (the, a, and). This reduces noise and helps the model generalise.
3. Better vectorization - Instead of simple counts (CountVectorizer), we will use TF-IDF (TfidfVectorizer) to give more weight to important words.
