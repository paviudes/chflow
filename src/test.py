import numpy as np
from analyze import plots as pl

# test the Varians bins plotting system
# Create X and Y values. Compute the Variance by bins
# Plot the bar chart showing the Variance by bins
npoints = 100
nbins = 10
testX = np.exp(np.random.uniform(low=-5, high=0, size=npoints))
testY = np.exp(np.random.uniform(low=-5, high=0, size=npoints))

print("X Y\n%s" % (np.array_str(np.column_stack((testX, testY)))))

# Compute bins
bins = pl.ComputeBinVariance(testX, testY, nbins)
print("bins\n%s" % (np.array_str(bins)))

# Plot bins
pl.AddBinVariancePlot(bins, 0, "infid", "infid", "infid", pdf = None, plotfname = "testplot.pdf")