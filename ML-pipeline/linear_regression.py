import pandas as pd
import statsmodels.formula.api as smf



def regression(data, i):
    harmonic = data[data['cluster'] == i]
    harmonic = harmonic[harmonic['cluster'].notnull()]

    model = smf.ols('frequency ~ onset', data = harmonic)
    results = model.fit()
    #print(results.summary())
    #plt.scatter(harmonic.power, harmonic.frequency)

    intercept = results.params[0] # intercept is the value of the predicted harmonic
    
    return intercept