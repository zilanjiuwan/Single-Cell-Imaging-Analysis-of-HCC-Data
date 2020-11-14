import numpy as np
import cv2
import math


# def rotate_bound(image, angle):
# 	# grab the dimensions of the image and then determine the
# 	# center
# 	if angle%90==0:
# 		return np.rot90(image, -angle//90)

# 	(h, w) = image.shape[:2]
# 	(cX, cY) = (w // 2, h // 2)
 
# 	# grab the rotation matrix (applying the negative of the
# 	# angle to rotate clockwise), then grab the sine and cosine
# 	# (i.e., the rotation components of the matrix)
# 	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
# 	cos = np.abs(M[0, 0])
# 	sin = np.abs(M[0, 1])
 
# 	# compute the new bounding dimensions of the image
# 	nW = int((h * sin) + (w * cos))
# 	nH = int((h * cos) + (w * sin))
 
# 	# adjust the rotation matrix to take into account translation
# 	M[0, 2] += (nW / 2) - cX
# 	M[1, 2] += (nH / 2) - cY
 
# 	# perform the actual rotation and return the image
# 	image_rotated = []
# 	for slc in range(int(np.ceil(image.shape[2]/500))):
# 		image_rotated.append(cv2.warpAffine(image[:,:,slc*500:(slc+1)*500], M, (nW, nH)))

# 	return np.concatenate(image_rotated, axis=-1)


# def rotate(image, mask, angle_step):
	
# 	assert mask.ndim==3

# 	for angle in range(0, 360, angle_step):
# 		image_rotated = rotate_bound(image, angle).astype('uint8')
# 		mask_rotated = rotate_bound(mask.astype('uint8'), angle).astype(bool)
# 		if mask_rotated.ndim<3:
# 			mask_rotated = np.expand_dims(mask_rotated, -1)
# 		yield image_rotated, mask_rotated

def rotate(image, mask, angle_step):

	assert image.ndim==3
	assert mask.ndim==3
	assert mask.dtype==bool

	rows, cols = image.shape[:2]

	for angle in range(0, 360, angle_step):
		if angle%90==0:
			image_rotated = np.rot90(image, angle//90)
			mask_rotated = np.rot90(mask, angle//90)
		else:
			M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
			image_rotated = cv2.warpAffine(image, M, (cols,rows))
			mask_rotated = cv2.warpAffine(mask.astype('uint8'), M, (cols,rows)).astype(bool)
			mask_rotated = np.expand_dims(mask_rotated, -1) if mask_rotated.ndim==2 else mask_rotated

		assert image_rotated.ndim==3
		assert mask_rotated.ndim==3
		assert mask_rotated.dtype==bool

		yield image_rotated, mask_rotated



def flip(image, mask, mode=0):

	yield image, mask
	yield image[::-1,:], mask[::-1,:]
	if mode==1:
		yield image[:,::-1], mask[:,::-1]
		yield image[::-1,::-1], mask[::-1,::-1]


def augment(image, mask, angle_step):

	for image_flipped, mask_flipped in flip(image, mask):
		for image_rotated, mask_rotated in rotate(image_flipped, mask_flipped, angle_step):
			yield image_rotated, mask_rotated

#填充切割
def crop(image, mask, H, W):

	# if image size is smaller than crop size, pad the image
	n_y, n_x = image.shape[:2]
	pad_y = max(H-n_y, 0)
	pad_x = max(W-n_x, 0)
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask = np.pad(mask, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])

	n_y, n_x = image.shape[:2]
	num_steps_y = math.ceil(n_y/H)
	num_steps_x = math.ceil(n_x/W)
	for i in np.linspace(0, n_y-H, num_steps_y, dtype=int):
		for j in np.linspace(0, n_x-W, num_steps_x, dtype=int):
			yield image[i:i+H, j:j+W], mask[i:i+H, j:j+W]
#重叠切割
def overlap_crop(image, mask, H, W):

	# pad the image to multiple of H and W
	n_y, n_x = image.shape[:2]
	pad_y = math.ceil(n_y/H)*H-n_y
	pad_x = math.ceil(n_x/W)*W-n_x
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask = np.pad(mask, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])

	n_y, n_x = image.shape[:2]
	for i in range(0, n_y-H+1, H//2):
		for j in range(0, n_x-W+1, W//2):
			yield image[i:i+H, j:j+W], mask[i:i+H, j:j+W]

#填充切割
def crop2(image, mask1,mask2, H, W):

	# if image size is smaller than crop size, pad the image
	n_y, n_x = image.shape[:2]
	pad_y = max(H-n_y, 0)
	pad_x = max(W-n_x, 0)
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask1 = np.pad(mask1, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask2 = np.pad(mask2, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	n_y, n_x = image.shape[:2]
	num_steps_y = math.ceil(n_y/H)
	num_steps_x = math.ceil(n_x/W)
	for i in np.linspace(0, n_y-H, num_steps_y, dtype=int):
		for j in np.linspace(0, n_x-W, num_steps_x, dtype=int):
			yield image[i:i+H, j:j+W], mask1[i:i+H, j:j+W],mask2[i:i+H, j:j+W]
#重叠切割
def overlap_crop2(image,  mask1,mask2, H, W):

	# pad the image to multiple of H and W
	n_y, n_x = image.shape[:2]
	pad_y = math.ceil(n_y/H)*H-n_y
	pad_x = math.ceil(n_x/W)*W-n_x
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask1 = np.pad(mask1, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask2 = np.pad(mask2, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	n_y, n_x = image.shape[:2]
	for i in range(0, n_y-H+1, H//2):
		for j in range(0, n_x-W+1, W//2):
			yield image[i:i+H, j:j+W], mask1[i:i+H, j:j+W],mask2[i:i+H, j:j+W]

#填充切割
def crop3(image, mask1,mask2,mask3, H, W):

	# if image size is smaller than crop size, pad the image
	n_y, n_x = image.shape[:2]
	pad_y = max(H-n_y, 0)
	pad_x = max(W-n_x, 0)
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask1 = np.pad(mask1, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask2 = np.pad(mask2, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask3 = np.pad(mask3, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])    
	n_y, n_x = image.shape[:2]
	num_steps_y = math.ceil(n_y/H)
	num_steps_x = math.ceil(n_x/W)
	for i in np.linspace(0, n_y-H, num_steps_y, dtype=int):
		for j in np.linspace(0, n_x-W, num_steps_x, dtype=int):
			yield image[i:i+H, j:j+W], mask1[i:i+H, j:j+W], mask2[i:i+H, j:j+W], mask3[i:i+H, j:j+W]
#重叠切割
def overlap_crop3(image, mask1,mask2,mask3, H, W):

	# pad the image to multiple of H and W
	n_y, n_x = image.shape[:2]
	pad_y = math.ceil(n_y/H)*H-n_y
	pad_x = math.ceil(n_x/W)*W-n_x
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask1 = np.pad(mask1, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask2 = np.pad(mask2, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask3 = np.pad(mask3, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])    
	n_y, n_x = image.shape[:2]
	for i in range(0, n_y-H+1, H//2):
		for j in range(0, n_x-W+1, W//2):
			yield image[i:i+H, j:j+W], mask1[i:i+H, j:j+W],mask2[i:i+H, j:j+W],mask3[i:i+H, j:j+W]


            
            
            
def split(image, mask, H, W):

	assert image.ndim==3
	assert mask.ndim==3
	n_y, n_x = image.shape[:2]
	N_y = math.ceil(n_y/H)
	N_x = math.ceil(n_x/W)
	pad_y = N_y*H-n_y
	pad_x = N_x*W-n_x
	image = np.pad(image, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	mask = np.pad(mask, [(pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)], 'constant', constant_values=[(0,0)])
	for i in range(N_y):
		for j in range(N_x):
			yield image[i*H:(i+1)*H, j*W:(j+1)*W], mask[i*H:(i+1)*H, j*W:(j+1)*W]


def mask_to_label(mask):
	'''
	mask must be a binary image and have 3 dims
	'''
	assert mask.ndim==3
	mask_sizes = np.sum(mask, axis=(0,1))
	mask_sizes = np.argsort(mask_sizes)[::-1]
	mask = mask[:,:,mask_sizes]
	label = np.zeros(mask.shape[:2], dtype='uint16')
	n_instances = mask.shape[2]
	for j in range(n_instances):
		label[mask[:,:,j]!=0] = j+1
	
	return label.astype(np.uint16)


def label_to_mask(label):
	n_instances = np.max(label)
	mask = np.zeros(label.shape+(n_instances,), dtype=bool)
	for k in range(n_instances):
		mask[:,:,k] = label==(k+1)
	return mask


def normalize_image(image):
	assert image.dtype==np.uint8
	image = (image-np.min(image)) / (np.max(image)-np.min(image)) * 255
	return image.astype('uint8')


def rle_encode(mask):
	pixels = mask.T.flatten()
	# We need to allow for cases where there is a '1' at either end of the sequence.
	# We do this by padding with a zero at each end when needed.
	use_padding = False
	if pixels[0] or pixels[-1]:
		use_padding = True
		pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
		pixel_padded[1:-1] = pixels
		pixels = pixel_padded
	rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
	if use_padding:
		rle = rle - 1
	rle[1::2] = rle[1::2] - rle[:-1:2]
	return rle


def rle_to_string(runs):
	return ' '.join(str(x) for x in runs)


def rle_decode(rle_str, mask_shape, mask_dtype):
	s = rle_str.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
	for lo, hi in zip(starts, ends):
		mask[lo:hi] = 1
	return mask.reshape(mask_shape[::-1]).T



def mask_to_coordinate(mask):
    coordinate = []
    for slc in range(mask.shape[2]):
        coordinate.append(set(zip(*np.nonzero(mask[:,:,slc]))))
    return coordinate


def evaluate(mask_true, mask_pred, threshes=np.arange(0.5, 1, 0.05)):

	coord_true = mask_to_coordinate(mask_true)
	coord_pred = mask_to_coordinate(mask_pred)
	iou = np.zeros([len(coord_true), len(coord_pred)], dtype=float)
	for i in range(len(coord_true)):
		for j in range(len(coord_pred)):
			iou[i,j] = len(coord_true[i]&coord_pred[j])/len(coord_true[i]|coord_pred[j])
	TP = np.zeros(len(threshes))
	FP = np.zeros(len(threshes))
	FN = np.zeros(len(threshes))
	for ix, t in enumerate(threshes):
		I, J = np.nonzero(iou>t)
		TP[ix] = len(I)
		FP[ix] = len(coord_pred) - len(J)
		FN[ix] = len(coord_true) - len(I)

	return np.mean(TP/(TP+FP+FN))


def prune_mask(mask):
    
    ## remove empty mask
    sizes = np.sum(mask, axis=(0,1))
    mask = mask[:,:,sizes>0]
    assert mask.ndim == 3
    
    ## remove empty bounding box
    n_instances = mask.shape[2]
    areas = np.zeros(n_instances)
    for k in range(n_instances):
        rows, cols = np.nonzero(mask[:,:,k])
        width, height = np.max(cols)-np.min(cols), np.max(rows)-np.min(rows)
        areas[k] = width*height
        
    mask = mask[:,:,areas>0]
    assert mask.ndim == 3
        
    return mask