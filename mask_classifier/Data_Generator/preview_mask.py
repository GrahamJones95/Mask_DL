from matplotlib import pyplot as plt
import mask
import os
import random

folder_path = "my_images/source"
dest_path = "my_images/"
#dist_path = "/home/preeth/Downloads"

#c = 0
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i in range(len(images)):
    print("the path of the image is", images[i])
    type = random.choice(list(mask.FaceMasker.Mask_Type))
    mask.create_mask(images[i],dest_path,type = type)
    