'''
Data Set Analysis
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot
import statsmodels.formula.api as smf
import pylab

# Import supplied csv, convert to pandas dataframe
data = pd.read_csv('executiontest.csv')

# Display all columns in output
pd.set_option('max_columns', None)

# Rename columns for ease of access
data.rename(columns=({'trading_date': 'date',
                      'exchange_id': 'exchange',
                      'traded_volume': 'volume',
                      'pnl': 'pnl'}), inplace=True)

# Remove rows that have zero values for volume or pnl likely due to holidays or bad data. Comment out to include them
data = data.drop(data[data.pnl == 0].index)
data = data.drop(data[data.volume == 0].index)

# Convert string prices to floats
data['volume'] = pd.to_numeric(data['volume'])
data['pnl'] = pd.to_numeric(data['pnl'])
data['exchange'] = pd.to_numeric(data['exchange'])

# Groups the data based on exchange number for total volume and pnl
exch_pl = data.groupby(['exchange']).sum()

# Observe a profit/volume comparison to see if there is a correlation between the two
profit_volume = (exch_pl['pnl'] / exch_pl['volume'])
profit_volume_sorted = profit_volume.sort_values()
volume_sorted = exch_pl['volume'].sort_values()
pnl_sorted = exch_pl['pnl'].sort_values()
pnl_volume_corr = exch_pl['pnl'].corr(exch_pl['volume'])

exchanges = {}
for x in range(1,(data['exchange'].max() + 1)):
    exchanges['exch_' + str(x).format(x)] = data.loc[data['exchange'] == x]

# Calculate mean volume for each exchange
exch_vol_means = []
exch_vol_stdev = []
exch_pnl_means = []
exch_pnl_stdev = []

for x in range(1,(data['exchange'].max() + 1)):
    means = exchanges['exch_' + str(x).format(x)]['volume'].mean()
    stdev = exchanges['exch_' + str(x).format(x)]['volume'].std()
    pnl_means = exchanges['exch_' + str(x).format(x)]['pnl'].mean()
    pnl_stdev = exchanges['exch_' + str(x).format(x)]['pnl'].std()

    exch_vol_means.append(means)
    exch_vol_stdev.append(stdev)
    exch_pnl_means.append(pnl_means)
    exch_pnl_stdev.append(pnl_stdev)

# Correlation of all pnl and volume data
all_corr = data['volume'].corr(data['pnl'])
pv_corr = []
for x in range(1,(data['exchange'].max() + 1)):
    corr = exchanges['exch_' + str(x).format(x)]['volume'].corr(exchanges['exch_' + str(x).format(x)]['pnl'])
    pv_corr.append(corr)

# New dataframe for mean, stdev of each exchange
exch_data = pd.DataFrame({'volume': exch_pl['volume'], 'pnl': exch_pl['pnl'], 'pv ratio': profit_volume,
                          'pv_corr': pv_corr, 'vol_mean': exch_vol_means, 'vol_stdev': exch_vol_stdev,
                          'pnl_mean': exch_pnl_means, 'pnl_stdev': exch_pnl_stdev})

# Creating charts for data on pnl_stdev and pv ratio
by_stdev = exch_data.sort_values(by = 'pnl_stdev')
by_stdev_desc = exch_data.sort_values(by = 'pnl_stdev', ascending=False)

# Individual total P/L over time for each exchange
pnl_cumsum = {}
for x in range(1,(data['exchange'].max() + 1)):
    pnl_cumsum['exch_' + str(x).format(x) + '_pnl'] = exchanges['exch_' + str(x).format(x)]['pnl'].cumsum().dropna().reset_index(drop=True)

# Create the chart showing pnl for all exchanges
pnl_plots = {}
for x in range(1,(data['exchange'].max() + 1)):
    pnl_plots['exch_' + str(x).format(x) + '_plots'] = plot.plot(pnl_cumsum['exch_' + str(x) + '_pnl'], label='Exch ' + str(x))

plot.xlabel('Time in Days (8/2/18 - 8/6/20)')
plot.ylabel('P/L Over Time (Millions)')
plot.legend(bbox_to_anchor =(1.05, 1.15), ncol = 7)
# plot.show()

# Statistical comparison of the exchanges
print(exch_data.describe())

# Regression of pnl on volume
exch_ols = smf.ols(formula = "exch_data['pnl']~exch_data['volume']", data=exch_data).fit()
print(exch_ols.params)
print(exch_ols.summary())

# scatter-plot data
ax = exch_data.plot(x='volume', y='pnl', kind='scatter')

# plot regression line
abline_plot(model_results=exch_ols, ax=ax)
ax.set_title('OLS Regression')
ax.set_ylim([-1000000,7000000])
# plot.show()

# Regression of Exchange 4 and 5
exch45_ols = smf.ols(formula = "pnl_cumsum['exch_4_pnl']~pnl_cumsum['exch_5_pnl'].iloc[:-2]", data=data['pnl']).fit()
print(exch45_ols.params)
print(exch45_ols.summary())

# Autocorrelations for each exchange
autocorrs = []
for x in range(1,(data['exchange'].max() + 1)):
    corrs = pnl_cumsum['exch_' + str(x) + '_pnl'].autocorr(lag=3)
    autocorrs.append(corrs)
print('Autocorrelations = ' + str(autocorrs))

