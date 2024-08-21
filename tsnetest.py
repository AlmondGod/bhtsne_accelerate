# # import numpy as np
# # import bhtsne
# # import sys
# # import os



# # data = np.loadtxt("mnist2500_X.txt", skiprows=1)
# # print("Data shape:", data.shape)

# # embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

# # print("Embedding shape:", embedding_array.shape)

# import numpy as np
# import bhtsne
# import sys
# import os
# import time

# # Redirect stdout and stderr to a file
# # output_file_time = time.strftime("%Y%m%d-%H%M%S")
# # output_file = 'tsne_output' + output_file_time + '.log'
# dataset = "mnist2500_X"
# # output_file = 'tsne_output_' + dataset + '3.log'
# output_file = 'testagain.log'
# sys.stdout = sys.stderr = open(output_file, 'w')

# try:
#     data = np.loadtxt(dataset + ".txt", skiprows=1)
#     print("Data shape:", data.shape)

#     embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

#     print("Embedding shape:", embedding_array.shape)

# finally:
#     # Restore stdout and stderr
#     sys.stdout.close()
#     sys.stdout = sys.__stdout__
#     sys.stderr = sys.__stderr__

# # Print a message to the console to indicate where the output was saved
# print(f"Output has been saved to {output_file}")

import numpy as np
import bhtsne
import sys
import os
import time

dataset = "mnist2500_X"
output_file_time = time.strftime("%Y%m%d-%H%M%S")
output_file = 'second_tsne_output' + output_file_time + '.log'

# Open the file outside the try block
with open(output_file, 'w') as f:
    # Redirect stdout and stderr to the file
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = sys.stderr = f

    try:
        data = np.loadtxt(dataset + ".txt", skiprows=1)
        print("Data shape:", data.shape)

        embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

        print("Embedding shape:", embedding_array.shape)

    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Print a message to the console to indicate where the output was saved
print(f"Output has been saved to {output_file}")