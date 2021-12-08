from matplotlib import pyplot as plt
import os
import csv
import shutil

experiments_path = 'my_data/'
data_path = 'my_images/'


def create_data_files(experiments_path, data_path, train_percent):
    folder_names = ["with_mask_correct","with_mask_incorrect","without_mask"] 
    with open(experiments_path+"/train.csv",'w') as file:
        wr = csv.writer(file)
        wr.writerow(["filename","class"])

    with open(experiments_path+"/test.csv",'w') as file:
        wr = csv.writer(file)
        wr.writerow(["filename","class"])

    for folder in folder_names:
        print(os.listdir())
        images = [os.path.join(data_path+folder,f) for f in os.listdir(data_path+folder)]
        
        print("Number of images in {} = {}".format(data_path+folder,len(images)))
        #Write some images to the train folder
        with open(experiments_path+"/train.csv",'a') as file:
            for image in images[:round(len(images)*train_percent)]:
                wr = csv.writer(file)
                wr.writerow([image,folder])
                shutil.copy(image,experiments_path+"/train")

                
        #Write remaining images to test folder
        with open(experiments_path+"/test.csv",'a') as file:
            for image in images[round(len(images)*train_percent)+1:]:
                wr = csv.writer(file)
                wr.writerow([image,folder])
                
create_data_files(experiments_path, data_path, 0.8)
# TO DO Still need to copy files over to new directory
                