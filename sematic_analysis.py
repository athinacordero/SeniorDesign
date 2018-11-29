import json
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim.util
import gensim.models

subreddits = ['depression', 'Anxiety', 'foreveralone', 'socialanxiety', 'SuicideWatch', 
                    'berkeley', 'PowerLedger', 'TalesFromYourServer', 'tifu']

total = []
for subreddit in subreddits:
    with open('datasets/'+subreddit+'_text_samples_extended.json') as f:
        data = json.load(f)
    
    for x in range(0, len(data['data'])):
        if x % 100 == 0:
            print(x)
        if 'selftext' in data['data'][x]:
            
            total.append(gensim.util.simple_preprocess(data.get('data')[x].get('selftext')))


model = gensim.models.Word2Vec(
        total,
        size=150,
        window=10,
        min_count=2,
        workers=10)
model.train(total, total_examples=len(total), epochs=10)

w1 = "sad"
model.wv.most_similar(positive=w1)