from PIL import Image # Image manipulations
import matplotlib.pyplot as plt
import os

#show test images
for image_path in os.listdir("TestResults"):
  image_path = os.path.join("TestResults",image_path)
  # Open the image to show it in the first column of the plot 
  image = Image.open(image_path)  
  # Create the figure 
  fig = plt.figure(figsize=(50,5))
  ax = fig.add_subplot(1, 1, 1) 
  # Plot the image in the first axe with it's category name
  ax.axis('off')
  ax.set_title(image_path)
  ax.imshow(image) 
#show result
for image_path in os.listdir("TestResults"):
  image_path = os.path.join("TestResults",image_path)
  # Open the image to show it in the first column of the plot 
  image = Image.open(image_path)  
  # Create the figure 
  fig = plt.figure(figsize=(50,5))
  ax = fig.add_subplot(1, 1, 1) 
  # Plot the image in the first axe with it's category name
  ax.axis('off')
  ax.set_title(image_path)
  ax.imshow(image) 
