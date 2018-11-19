from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy
import sys
import json
import nltk
import numpy as np

def get_pos_counts_for_subreddit(subreddit="depression"):
    post_dict = {}

    with open('datasets/'+subreddit+'_text_samples_extended.json') as f:
        data = json.load(f)

    print("Loading data for subreddit " + subreddit)
    for x in range(0, len(data['data'])):
        if x % 100 == 0:
            print(x)
        if 'selftext' in data['data'][x]:
            tok = nltk.word_tokenize(data.get('data')[x].get('selftext'))
            post_dict[data.get('data')[x].get('title')] = nltk.pos_tag(tok)

    # list all parts of speech, and count instances of each

    pos = post_dict.values()
    counts = dict()
    # poss = pos.split()

    for post in pos:
        for word_pos_pair in post:
            if word_pos_pair[1] in counts:
                counts[word_pos_pair[1]] += 1
            else:
                counts[word_pos_pair[1]] = 1

    return counts


def main():

    # Get pos counts for each subreddit
    subreddits = ['depression', 'Anxiety', 'foreveralone', 'socialanxiety', 'SuicideWatch', 
                    'berkeley', 'PowerLedger', 'TalesFromYourServer', 'tifu']

    totalCounts = []
    for subreddit in subreddits:
        counts = get_pos_counts_for_subreddit(subreddit=subreddit)
        countsArray = []
        for pos in counts:
            countsArray.append(counts[pos])
        totalCounts.append(countsArray)

    # Assure each subreddit is an array of the same size
    vec = []
    longest_row_len = max(len(row) for row in totalCounts)

    for row in totalCounts:
        row = np.append(np.array(row), np.zeros(longest_row_len-len(row)))
        vec = np.append(vec, row)

    # Reshape array to be 2D array
    vec = np.reshape(vec, (len(subreddits), int(vec.shape[0]/len(subreddits))))

# begin MDS plotting

    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(vec)
    X_transformed.shape
    print(X_transformed)

    # ax = plt.axes([0., 0., 1., 1.])

    s = 100
    # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color='navy', s=s, lw=0)

    fix, ax = plt.subplots()
    ax.scatter(vec[:, 0], vec[:, 1])
    for i, subreddit in enumerate(subreddits):
        ax.annotate(subreddit, (vec[i][0], vec[i][1]))

    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    # segments = [[X_transformed[i, :], X_transformed[j, :]]
    #             for i in range(len(X_transformed)) for j in range(len(X_transformed))]
    # values = numpy.abs(vec)
    # lc = LineCollection(segments, zorder=0, cmap=plt.cm.Blues, norm=plt.Normalize(0, values.max()))
    # lc.set_array(X_transformed.flatten())
    # lc.set_linewidths(numpy.full(len(segments), 0.5))
    # ax.add_collection(lc)

    plt.show()


if __name__ == '__main__':

        main()
