"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 8
# count
# precision = (77, 82, 92, 78, 91, 68, 82, 94)
# recall = (73, 74, 91, 69, 89, 68, 83, 94)

# no lemma
precision = (70, 85, 87, 71, 88, 68, 84, 92)
recall = (70, 81, 86, 84, 94, 68, 81, 91)
# lemma
# precision = (69, 85, 85, 69, 92, 70, 79, 87)
# recall = (67, 81, 82, 83, 96, 71, 79, 93)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, precision, width, color='r')
rects2 = ax.bar(ind + width, recall, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Precision and recall')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Alcohol', 'Delivery', 'GoodForKids', 'NoiseLevel', 'WheelChairAccessible', 'WiFi', 'BikeParking', 'RestaurantsTakeOut'))

ax.legend((rects1[0], rects2[0]), ('Precision', 'Recall'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
