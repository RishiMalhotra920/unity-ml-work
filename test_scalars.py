from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
# Use a simple directory name without spaces or special characters
writer = SummaryWriter(f"runs/test_{datetime.now()}")
bin_edges = np.arange(0.5, 11.5, 1)

# Log histogram data
writer.add_histogram('this_tag', np.array([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), bins=bin_edges, global_step=0)
writer.add_histogram('this_tag', np.array([1, 1, 1, 1, 1, 1, 1, 1, 6, 7, 8, 9, 10]), bins=bin_edges, global_step=1)
writer.add_histogram('this_tag', np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 9, 10]), bins=bin_edges, global_step=2)

writer.close()


# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# writer = SummaryWriter("runs/experiment_5")
# for i in range(10):
#     x = np.random.random(1000)
#     writer.add_histogram('distribution centers', x + i, i)
# writer.close()