import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()
ROOT = os.environ.get("ROOT")

labels = ['URLs', 'Code', 'Logs']
train = [0.856, 0.460, 0.223]
test = [0.800, 0.420, 0.198]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train, width, label='train')
rects2 = ax.bar(x + width/2, test, width, label='test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('% of Machine information in train and test sets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig(f"{ROOT}/results/eda.png")
