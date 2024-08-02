import numpy as np

array = [(2,5), (4,18), (12,9), (2,7), (4, 20)]

def new_average_consecutive(array):
    table = {}
    for n,v in array:
      try:
          table[n].append(v)
      except:
          table[n] = [v]

    output = []
    for n in sorted(table):
       output.append(np.average(table[n]))
    
    return output

print(new_average_consecutive(array))