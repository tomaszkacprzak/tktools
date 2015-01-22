import tktools
import numpy as np
sub_set  = np.array([6,1,2,1,5])
full_set = np.array([12,1,2,5,7,6,8,9,10]);
indices=tktools.search_unsorted(sub_set,full_set)
print sub_set
print full_set
print indices