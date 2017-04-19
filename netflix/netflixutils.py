import _pickle as pickle
import numpy as np
import pandas as pd
import pudb
from sklearn.decomposition import PCA
from toolz import pipe
from tqdm import tqdm


DATA_ = 'ml-20m/'
MOVIES = 'movies.csv'
LINKS = 'links.csv'
RATINGS = 'ratings.csv'
TAGS = 'tags.csv'
GENOME_SCORES = 'genome-scores.csv'
GENOME_TAGS = 'genome-tags.csv'




def get_users_by_moviecount(lb, ub):
    try:
        userRatings = pickle.load(open('pickles/users_by_moviecount_{}_to_{}.p'.format(lb, ub), 'rb'))
        print ('Precomputed user data found.')
    except:
        print ('Precomputed user data not found.')
        df = pd.read_csv(DATA_ + RATINGS)
        ratingsPerUser = df.groupby('userId').size().sort_values(ascending=False)
        ratingsPerUser = ratingsPerUser[ ratingsPerUser <= ub ]
        ratingsPerUser = ratingsPerUser[ ratingsPerUser >= lb ]
        userRatings = []
        print ('There are {} users to collect.'.format(len(ratingsPerUser)))
        for userId, count in tqdm(ratingsPerUser.iteritems()):
            userRatings.append(df[df.userId == userId])
        pickle.dump(userRatings, open('pickles/users_by_moviecount_{}_to_{}.p'.format(lb, ub), 'wb'))

    return userRatings
    

def get_tags_dict(row, df, nFeatures):
    """
    takes a row of ratingsDf and genome-scoresDf to tags-relevance dict.
    """
    movie = row.movieId
    df = df[df.movieId == movie]
    if len(df) != nFeatures:
        return np.NaN
    relevances = df.drop('movieId', axis=1)\
            .set_index('tagId')\
            .to_dict()
    return relevances['relevance']

def get_relevance_scores_for_movie(row, dfg, nFeatures):
    movie = row.movieId
    df = dfg[dfg.movieId == movie]
    if len(df) != nFeatures:
        return np.nan
    relevance = list(df.relevance.values)
    return relevance



def get_feature_scores(dfu, nFeatures, method='top'):
    print ('Loading genome')
    try:
        dfGenome = pickle.load(open('pickles/dfGenome.p', 'rb'))
    except:
        dfGenome = pd.read_csv(DATA_ + GENOME_SCORES)
        pickle.dump(dfGenome, open('pickles/dfGenome.p', 'wb'))
    ratedMovies = dfu.movieId.values
    dfGenome = dfGenome[ dfGenome.movieId.isin(ratedMovies) ]

    if method == 'top':
        topTags = dfGenome.groupby('tagId').mean().relevance\
                .nlargest(nFeatures)\
                .index.values
        dfGenome = dfGenome[ dfGenome.tagId.isin(topTags) ]
        dfu['scores'] = dfu.apply(lambda x: get_relevance_scores_for_movie(x, dfGenome, nFeatures), axis=1)
    
    elif method == 'pca':
        allTags = list(set(dfGenome.tagId.values))
        nTags = len(allTags)
        dfu['raw_scores'] = dfu.apply(lambda x: get_relevance_scores_for_movie(x, dfGenome, nTags), axis=1)
        dfu = dfu.dropna(subset=['raw_scores'])
        allScores = pipe(dfu.raw_scores.values,
                list,
                np.array)
        pca = PCA(n_components=nFeatures)
        pca.fit(allScores)
        allScores = pipe(allScores,
                pca.transform,
                list)
        dfu['scores'] = allScores
    
    else:
        print ('Method not recognized. Exiting')
        sys.exit()
    return dfu
    
   



