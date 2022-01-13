####
# Delivery Vehicles - Hubconf)
####

# Model to identify delivery vehicles!

import torch


def delivery_vehicles(verbose=True,device=None):
	""" Creates a Delivery Vehicle model
	This is a YOLOv5 based model to detect various types of delivery vehicles - UPS, US Postal Service, etc
	"""

	from pathlib import Path
	from models.yolo import Model
	from utils.downloads import attempt_download
	from utils.general import check_requirements, intersect_dicts, set_logging
	from utils.torch_utils import select_device

	check_requirements(exclude=('tensorboard','thop','opencv-python'))

	channels = 1
	classes = 4

	# Path to the checkpoint
	path = Path('delivery_vehicles.pt')

	# Try to create the model
	try:
		# Do we use a CUDA device?
		device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
		cfg = list((Path(__file__).parent).rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
		model = Model(cfg,channels,classes)
		#chkpt = torch.load(attempt_download(path), map_location=device)
		#csd = ckpt['model'].float().state_dict() # Checkpoint state_dict as FP32
		#csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors']) # intersect
		#model.load_state_dict(csd, strict=False) # Load
		checkpoint = "https://drive.google.com/file/d/1cV1O7hlKjQZBtbdWFxfvuBBN6b0UjpmY/view?usp=sharing"
		model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,progress=True))
		if len(ckpt['model'].names) == classes:
			model.names = ckpt['model'].names # Set the class names attribute
		return model.to(device)

	except Exception as e:
		raise Exception(s) from e
