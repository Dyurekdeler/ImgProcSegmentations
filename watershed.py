from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import watershed, disk
from skimage.filters import rank
from skimage import io, measure

image_o = io.imread("pusheen.jpg")
#convert image to grayscale
image = rgb2gray(image_o)

# denoise image
denoised = rank.median(image, disk(2))
# find continuous region (low gradient - where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]
# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))
# process the watershed
labels = watershed(gradient, markers)

count = measure.label(labels)
#print label with max value , which is equal to total num
print(count.max())

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image_o, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(image, cmap=plt.cm.gray)
ax[1].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.7)
ax[1].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()