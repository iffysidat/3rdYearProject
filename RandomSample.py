from numpy.random import exponential as exp
import numpy as np
lengthOfYourDataset = 17000

# Tune these parameters
pointsToSample = 1000 # Number of points we will end up with (i.e. rows of your data / events)
exponentialFalloff = 2000 # Essentially tunes the falloff of the distribution
                          # Thus the weighting of the points
# ---------------------

indicesToUse = set()
while len(indicesToUse) < pointsToSample:
    index = lengthOfYourDataset - int(exp(exponentialFalloff)) # Get a point, and flip the distribution
    index = np.max(index, 0) # Might draw a point > the length of the data set - be ready for this
    indicesToUse.add(index)

print(len(indicesToUse))
print(indicesToUse)
# Plot the histogram of sampled points to make irfan happy
import matplotlib.pyplot as plt
plt.hist(list(indicesToUse), bins=30)
plt.show()