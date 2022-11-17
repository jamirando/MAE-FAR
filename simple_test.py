import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from MAE.util import misc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(imaage, title=''):
	assert image.shape[2] == 3
	plt.imshow(torch.clip((image * imagenet_std + imagenet_mean)* 255,0,255).int())
	plt,title(title, fontsize=16)
	plt.axis('off')
	return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', random_mask=False, finetune=False):
	model = misc.get_mae_model(arch, random_mask=random_mask, finetune=finetune)
	checkpoint = torch.load(chkpt_dir, map_location='cpu')
	msg = model.load_state_dict(checkpoint['model'], strict=False)
	print(msg)
	return model

def run_one_image(img, mask, model):
	x = torch.tensor(img, dtype=torch.float32).cuda()
	mask = torch.tensor(mask, dtype=torch.float32).cuda()

	x = x.unsqueeze(dim=0)
	x = torch.einsum('nhwc->nchw',x)
	mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])

	y, mask2 = model.forward_return_image(x.float(), mask)
	y = torch.einsum('nchw->nhwc',y).detach()

	mask2 = mask2.detach()
	mask2 = mask2.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
	mask2 = model.unpatchify(mask2)
	mask2 = torch.einsum('nchw->nhwc',mask).detach()

	x = torch.einsum('nchw->nhwc', x)

	im_masked = x * (1 - mask)
	im_masked = im_masked.cpu()

	im_masked2 = x * (1 - mask2)
	im_masked2 = im_masked2.cpu()

	im_paste = x * (1 - mask) + y*mask
	im_paste = im_paste.cpu()
	x = x.cpu()
	y = y.cpu()

	plt.rcParams['figure.figsize'] = [24, 24]
	plt.subplot(1, 5, 1)
	show_image(x[0], "original")

	plt.subplot(1, 5, 2)
	show_image(im_masked[0], "masked")

	plt.subplot(1, 5, 3)
	show_image(im_masked2[0], "enlarged_masked")

	plt.subplot(1, 5, 4)
	show_image(y[0], "reconstruction")

	plt.subplot(1, 5, 5)
	show_image(im_paste[0], "reconstruction + visible")


	plt.savefig('test.png')
	#plt.show()
	plt.close()

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', metavar='img', type=str, help='Path to image')
    parser.add_argument('--mask', metavar='mask', type=str, help='Path to mask')
    parser.add_argument('--chkpt', metavar='chkpt', type=str, default='./pretrained/mae_pretrain_vit_large.pth', help='Path to checkpoint')
    return parser.parse_args()

def main(args):

	img = cv2.imread(args.image)[:,:,::-1]
	img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
	img = np.array(img) / 255.

#	assert image.shape == (256, 256, 3)

	img = img - imagenet_mean
	img = img / imagenet_std


	mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
	mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
	mask = np.array(mask) / 255.

	#chkpt_dir = 'pretrained/mae_pretrain_vit_large.pth'
	model_mae = prepare_model(args.chkpt, random_mask=False, finetune=False)
	print('Model loaded.')

	torch.manual_seed(2)
	run_one_image(img, mask, model_mae)

if __name__ == '__main__':
    main(argparser())