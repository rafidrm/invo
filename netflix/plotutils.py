import matplotlib.pyplot as plt
plt.style.use('ggplot')



def attach_common_kwargs(kwargs):
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 1
    if 'legend' not in kwargs:
        kwargs['legend'] = False
    return kwargs



def plot_ts(ts, **kwargs):
    ''' Plots time series.
        Args:
            ts (list) : list of timeseries. Each is a tuple (x, y, label)
        Usage:
            plotData = [ (allDates, portfolio.pr, portfolio.name) for portfolio in allPortfolios ]
            plotKwargs = { 'title': 'Daily Return',
                'rotation': 45,
                'alpha': 0.5,
                'legend': True,
                'save': 'fname'}
            plot_ts(plotData, **plotKwargs)
    '''
    kwargs = attach_common_kwargs(kwargs)
    fig1 = plt.figure()
    for x in ts:
        plt.plot(x[0], x[1], label=x[2], alpha = kwargs['alpha'])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'rotation' in kwargs:
        plt.xticks(rotation=kwargs['rotation'])
    if kwargs['legend'] == True:
        plt.legend()
    if 'save' in kwargs:
        fig1.savefig('{}.svg'.format(kwargs['save']), format='svg', dpi=1200, bbox_inches='tight')
        fig1.savefig('{}.eps'.format(kwargs['save']), format='eps', dpi=1200, bbox_inches='tight')


def plot_histogram(xs, **kwargs):
    ''' Plots histogram.
        Args:
            xs (list) : contains tuples (x, label), where x is the list for histogramming.
        Usage:
            plotData = [ (portfolio.pr, portfolio.name) for portfolio in allPortfolios ]
            plotKwargs = { 'title': 'Histogram of Daily Returns',
                'bins': 50,
                'alpha': 0.5,
                'legend': True,
                'save': 'fname'}
            plot_histogram(plotData, **plotKwargs)
    '''
    kwargs = attach_common_kwargs(kwargs)
    if 'bins' not in kwargs:
        kwargs['bins'] = 30
    fig1 = plt.figure()
    for x in xs:
        plt.hist(x[0], bins=kwargs['bins'], alpha=kwargs['alpha'], label=x[1])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'rotation' in kwargs:
        plt.xticks(rotation=kwargs['rotation'])
    if kwargs['legend'] == True:
        plt.legend()
    if 'save' in kwargs:
        fig1.savefig('{}.svg'.format(kwargs['save']), format='svg', dpi=1200, bbox_inches='tight')
        fig1.savefig('{}.eps'.format(kwargs['save']), format='eps', dpi=1200, bbox_inches='tight')

