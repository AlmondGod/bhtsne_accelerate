# import numpy as np
# import bhtsne
# import sys
# import os



# data = np.loadtxt("mnist2500_X.txt", skiprows=1)
# print("Data shape:", data.shape)

# embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

# print("Embedding shape:", embedding_array.shape)

import numpy as np
import bhtsne
import sys
import os
import time

# Redirect stdout and stderr to a file
# output_file_time = time.strftime("%Y%m%d-%H%M%S")
# output_file = 'tsne_output' + output_file_time + '.log'
dataset = "mnist2500_X"
output_file = 'tsne_output_' + dataset + '2.log'
sys.stdout = sys.stderr = open(output_file, 'w')

try:
    data = np.loadtxt(dataset + ".txt", skiprows=1)
    print("Data shape:", data.shape)

    embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

    print("Embedding shape:", embedding_array.shape)

finally:
    # Restore stdout and stderr
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Print a message to the console to indicate where the output was saved
print(f"Output has been saved to {output_file}")