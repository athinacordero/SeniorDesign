from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy
import sys
import json
import nltk


def main():

    post_dict = {}
    vec = []

# Open json and tokenize, POS tag posts

    with open(sys.argv[1]) as f:
        data = json.load(f)

    for x in range(0, len(data['data'])):
        if 'selftext' in data['data'][x]:
            tok = nltk.word_tokenize(data.get('data')[x].get('selftext'))
            post_dict[data.get('data')[x].get('title')] = nltk.pos_tag(tok)

# list all parts of speech, and count instances of each

    pos = post_dict.values()
    counts = dict()
    poss = pos.split()

    for word in poss:
        if word in poss:
            counts[word] += 1
        else:
            counts[word] = 1

# append part of speech counts to empty array

    vec.extend(counts.values())

# begin MDS plotting

    embedding = MDS(n_components=35)
    X_transformed = embedding.fit_transform(vec)
    X_transformed.shape

    ax = plt.axes([0., 0., 1., 1.])

    s = 100
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color='navy', s=s, lw=0,
                label='Depression')

    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[X_transformed[i, :], X_transformed[j, :]]
                for i in range(len(pos)) for j in range(len(X_transformed))]
    values = numpy.abs(vec)
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.Blues,
                        norm=plt.Normalize(0, values.max()))
    lc.set_array(X_transformed.flatten())
    lc.set_linewidths(numpy.full(len(segments), 0.5))
    ax.add_collection(lc)

    plt.show()


if __name__ == '__main__':

        main()
