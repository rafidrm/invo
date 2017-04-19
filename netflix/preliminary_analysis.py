import numpy as np
import pandas as pd
import pudb

from toolz import pipe
from scipy.spatial import ConvexHull
from sklearn.metrics import r2_score 
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from tqdm import tqdm
tqdm.pandas(desc="~")

from gmio import LinearAbsoluteDualityGap






DATA_ = 'ml-20m/'
MOVIES = 'movies.csv'
LINKS = 'links.csv'
RATINGS = 'ratings.csv'
TAGS = 'tags.csv'
GENOME_SCORES = 'genome-scores.csv'
GENOME_TAGS = 'genome-tags.csv'



def get_user():
    print('loading df')
    df = pd.read_csv(DATA_ + RATINGS)
    ratingsPerUser = df.groupby('userId').size().sort_values(ascending=False)
    # userid | count
    # --------------
    # 118205 | 9254
    # 8405   | 7515
    # 121535 | 5640
    # 125794 | 5491
    # ...
    # 54478  | 1251
    # 27387  | 620
    # 128110 | 211
    # 111047 | 45
    user = 111047
    dfu = df[df.userId == user]
    return dfu

def get_tags_dict(row, df, nFeatures):
    """
    takes a row of ratingsDf and genome-scoresDf to tags-relevance dict.
    """
    movie = row.movieId
    df = df[df.movieId == movie]
    if len(df) != nFeatures:
        return np.NaN
    #assert len(df) == 1128, 'every tag must have a relevance score'
    relevances = df.drop('movieId', axis=1)\
            .set_index('tagId')\
            .to_dict()
    return relevances['relevance']

def get_genomes(ratedMovies, nFeatures):
    print ('loading genome')
    dfGenome = pd.read_csv(DATA_ + GENOME_SCORES)
    dfGenome = dfGenome[ dfGenome.movieId.isin(ratedMovies) ]
    #I do feature selection by leaving only the top n tags
    topTags = dfGenome.groupby('tagId').mean().relevance\
            .nlargest(nFeatures)\
            .index.values
    dfGenome = dfGenome[ dfGenome.tagId.isin(topTags) ]
    return dfGenome

def df2features(df):
    scores = pipe(df.scores.values,
            list,
            np.array,
            )
    return scores

def get_convex_hull(pts):
    """
    returns A,b for P = { x | Ax >= b }.
    """
    print ('loading convex hull')
    hull = ConvexHull(pts)
    # hull returns poly in xA + b <= 0 form.
    m,n = hull.equations.shape
    b = hull.equations[:,n-1]
    A = hull.equations[:,0:n-1]
    A = -1 * A
    return np.mat(A),np.mat(b).T

def get_training_set(df, maxNumber):
    """
    There are several options. Can grab everything rated = 5.0, or >= 4.5
    if not a lot of data. If lots of data, then select top n%.
    """
    print ('loading training set')
    pts = df[df.rating == 5]
    if len(pts) > maxNumber:
        pu.db
    pts = df2features(pts)
    return [ np.mat(pt).T for pt in pts ]

def predict_invo_rating(cost, pts):
    ratings = np.zeros(len(pts))
    for i, pt in enumerate(pts):
        ratings[i] = (cost * pt).tolist()[0][0]
    return ratings




def first_experiment():
    nFeatures = 15
    maxTraining = 100
    dfu = get_user() # currently preset to a fixed user 
    ratedMovies = dfu.movieId.values
    dfg = get_genomes(ratedMovies, nFeatures) # truncated/relevant genomes matrix

    # get and clean relevance scores for tags
    dfu['scoresDict'] = dfu.progress_apply(lambda x: get_tags_dict(x, dfg, nFeatures), axis=1)
    dfu = dfu.dropna()
    dfu['tags'] = dfu.apply(lambda x: list(x.scoresDict.keys()), axis=1)
    dfu['scores'] = dfu.apply(lambda x: list(x.scoresDict.values()), axis=1)
    scores = df2features(dfu)
    AIneq, bIneq = get_convex_hull(scores)
    xTrain = get_training_set(dfu, maxTraining)

    # invo
    print ('attempting invo')
    invo = LinearAbsoluteDualityGap(AIneq, bIneq, verbose=True)
    res = invo.fit(xTrain)
    topRatings = predict_invo_rating(invo.c, xTrain)
    # TODO: the cost vector in invo is proportional to the feature values
    # and has no real bearing to the actual score. so if features ~ [0,1]
    # and score ~ [0,5] then we need to introduce some sort of 
    # normalization in order to be able to interpret the results.
    # One option is to 

    # lets just throw all points and see ratios?
    allPts = df2features(dfu)
    allPts = [ np.mat(pt).T for pt in allPts ]
    predRatings = predict_invo_rating(invo.c, allPts)
    trueRatings = dfu.rating.values
    trueRatings = trueRatings / 10
    r2 = r2_score(trueRatings, predRatings)
    print ('r2 score = {}'.format(r2))
    #pu.db
    
    # plot results
    xL = np.arange(len(predRatings))
    plt.figure(1)
    trueL = plt.plot(trueRatings, label='True')
    scaledL = plt.plot(predRatings, label='Scaled x10')
    #plt.legend(handles = [trueL, scaledL])
    plt.title('Scaled Predicted Rating vs True Rating (R2={})'.format(r2))
    plt.xlabel('Movie index')
    plt.ylabel('Rating')
    plt.show()



if __name__ == "__main__":
    first_experiment()
