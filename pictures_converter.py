from PIL import Image
import glob
from numpy.random import choice
import os

EXTENSION = "jpg"
PROB = 0.7

# remove files from test/training dirs if exists
for direc in ['training/colour', 'training/grey', 'test/colour', 'test/grey']:
  for fname in os.listdir(direc):
    if fname.endswith(EXTENSION):
      os.remove(direc + "/" + fname)

# process all images from pictures directory
file_name = {"test": 0, "training": 0}
for filename in glob.glob("pictures/*." + EXTENSION):
    im = Image.open(filename)
    if not im.mode in ['RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV']:
      print(im.mode)
      continue
    # image modes https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#modes
    im_rgb, im_grey = im.convert("RGB"), im.convert("L")
    # randomly assign to training or test sets with givrn prob
    set_dir = choice(["training", "test"], 1, p=[PROB, 1- PROB])[0]
    # save grey and colour image version
    im_rgb.save(set_dir + "/colour/" + str(file_name[set_dir]) + "." + EXTENSION)
    im_grey.save(set_dir + "/grey/" + str(file_name[set_dir]) + "." + EXTENSION)
    # increment file name
    file_name[set_dir] += 1
