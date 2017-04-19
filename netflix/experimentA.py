import numpy as np
import pandas as pd
import pudb

from matplotlib import pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import r2_score
from toolz import pipe


from invo.LinearModels.AbsoluteDualityGap import AbsoluteDualityGap 
from netflixutils import get_users_by_moviecount, get_feature_scores
from plotutils import plot_histogram




def get_training_set(df):
    """ There are several options. Can grab everything rated = 5.0, or >= 4.5
    if not a lot of data. If lots of data, then select top n%.
    """
    print ('Loading training set')
    df = df[df.rating == 5]
    points = list(df.scores.values)
    return points


def scale_and_quantize_ratings(rawPredRatings, trueRatings):
    rawPredRatings = [ float(rp) for rp in rawPredRatings ]
    rawPredRatings = rawPredRatings + np.abs(np.min(rawPredRatings))
    scaleFactor = np.mean(trueRatings) / np.mean(rawPredRatings)
    predRatings = rawPredRatings * scaleFactor
    predRatings[ predRatings >= 5 ] = 5
    predRatings[ predRatings <= 0 ] = 0
    predRatings = np.round(predRatings * 2) / 2
    return predRatings


def invo_experiment(dfu, nFeatures, plotResults=False):
    dfu = get_feature_scores(dfu, nFeatures, 'pca')
    allPoints = list(dfu.scores.values)
    adg = AbsoluteDualityGap() 
    adg.ForwardModel(forward='hull', points=allPoints)

    optimalPoints = get_training_set(dfu) 
    adg.solve(optimalPoints)
    results = {
            'solved': adg.solved,
            'cost': adg.c,
            'error': adg.error,
            'dual': adg.dual
            }

    rawPredRatings = [ point * adg.c.T for point in allPoints ]
    trueRatings = dfu.rating.values
    predRatings = scale_and_quantize_ratings(rawPredRatings, trueRatings)
    r2 = r2_score(trueRatings, predRatings)
    print ('r2 score = {}'.format(r2))

    if plotResults == True:
        # reorder movies
        trueRatings, predRatings = (list(x) for x in zip(*sorted(zip(trueRatings, predRatings))))
        xL = np.arange(len(predRatings))
        fig1 = plt.figure()
        plt.plot(trueRatings, label='True')
        plt.plot(predRatings, ls='dashed', label='Predicted')
        #plt.legend(handles = [trueL, scaledL])
        paramsTitle = 'USER={}\tFEATURES={}\tSIZE={}\tR2={}'\
                .format(PREDICT_USER,
                    NUMBER_OF_FEATURES,
                    len(optimalPoints),
                    r2)
        plt.title('Scaled Predicted Rating vs True Rating\n' + paramsTitle)
        plt.xlabel('Movie index')
        plt.ylabel('Rating')
        plt.ylim(-0.5, 5.5)

        fig1.savefig(paramsTitle, format='svg', dpi=700, bbox_inches='tight')
    return r2








def first_experiment():
    # experiment parameters
    nFeatures = 10
    maxTrainingSize = 100
    lb = 50
    ub = 50
    nUsers = 200 
    plotResults = False 

    # get user ratings data
    dfu = get_users_by_moviecount(lb, ub)
    dfu = dfu[:nUsers]

    r2s = []
    for ind, df in enumerate(dfu):
        print ('Testing user {} of {}'.format(ind, nUsers))
        score = invo_experiment(df, nFeatures, plotResults)
        r2s.append(score)
    
    print ('Summary statistics:')
    print ('Mean={}'.format(np.mean(r2s)))
    print ('Max={}'.format(np.max(r2s)))
    print ('Min={}'.format(np.min(r2s)))


    plotData = [ (r2s, 'R2 scores') ]
    plotKwargs = {
            'title': 'Histogram of R2 scores',
            'bins': 50,
            'save':'R2_histograms_for_{}_users_in_{}_to_{}_range'.format(
                nUsers,
                lb,
                ub)}
    plot_histogram(plotData, **plotKwargs)



if __name__ == "__main__":
    first_experiment()
