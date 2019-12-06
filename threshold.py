from skimage import io, filters
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import measure

image = io.imread('pusheen.jpg')
#work on extra copy of image to keep the original untouched
im = image
#convert image to gray
im = rgb2gray(im)
#threshold segmentation
val = filters.threshold_otsu(im)
drops = ndimage.binary_fill_holes(im < val)

labels = measure.label(drops)
#prints max value label which is equal to total num.
print(labels.max())

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(drops, cmap="gray")
ax[1].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()