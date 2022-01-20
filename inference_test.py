# Test inference against a random selection of files
import os, random, shutil, torch
from yolo.utils import select_device
from datetime import datetime
from pathlib import Path

# Image list, from which things are randomly selected.

# Read in the image filename list.
with open('./dv_images.txt','r') as f:
	image_list = f.readlines()
random_num = 20

# Set the device
device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else 'cpu')

# Create and load the model
model = torch.hub.load('chrisgilldc/delivery_vehicles','delivery_vehicles',force_reload=True,device=device)

# Create an output directory
output_dir = './inference_test_' + datetime.now().strftime('%Y%m%d_%H%M')
os.mkdir(output_dir)

# Index counter
i = 0

# Make a number of random inferences on the files.
while i < random_num:
	# Select a random file
	image_filename = image_list[random.randrange(0,len(image_list))].strip()

	ifp = Path(image_filename)
	original_dest_filename = output_dir + '/' + ifp.stem + '_orig' + ifp.suffix
	# Copy the input file to the output directory for easy reference
	shutil.copy2(image_filename,original_dest_filename)

	# Run the inference
	result = model(image_filename)
	# Inferred output
	#f = open(output_dir + '/' + str(i) + '_result.txt','w')
	#f.write(result.print())
	#f.close()

	result.save(save_dir=output_dir)

	# Increate the counter
	i += 1
