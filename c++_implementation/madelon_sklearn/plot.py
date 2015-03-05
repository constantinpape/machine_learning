import numpy as np
import pylab as plt
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
     description = "Pass filepaths to generated data for plotting.")
  parser.add_argument('filepaths', metavar='generated_data', type=str,
                     nargs='+', help="filepaths to generated_data")
  args = parser.parse_args().filepaths
  
  #plot for each feature 
  for filename in args:
    gen = np.genfromtxt(filename)
    for i in range(gen.shape[1]): 
      plt.hist(gen[:,i],50)
      #plt.plot(gen[:,i])
      #plt.show()
      plt.ylabel("counts")
      plt.xlabel("feature_dim " + str(i+1))
      plt.savefig(filename + str(i+1) + ".png")
