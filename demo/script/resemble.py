# -*- coding: utf-8 -*-
import h5py
import nibabel as nib
import os
import sys
import numpy as np
import pandas as pd
import time
import logging
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.measure import compare_ssim
import skimage.transform as skitran
import keras.backend as K
K.set_image_data_format("channels_first")
import tensorflow as tf

from train import SR_UnetGAN
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

class Resemble():
	def __init__(self):
		self.loca = 'SH_Vision'
		# self.loca = 'Bern_Vision'
		# self.loca = 'Bern_Vision_cross_tracer'
		# self.loca = 'SH_Vision_cross_tracer'
		# self.loca = 'SH_GE'
		# self.loca = 'SH_UI'

		root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
		self.nii_data_path = os.path.join(root_dir, '../data/{}'.format(self.loca))
		self.center_max = [34, 367250]
		self.source_dir = self.nii_data_path

		self.save_dir = os.path.join(root_dir, '../result')
		self.model_path = os.path.join(root_dir, '../model/generator_epoch_100.hdf5')

		self.generator = SR_UnetGAN().build_generator()
		self.generator.load_weights(self.model_path)

		self.upsample_order = 5

	def de_normalization(self,volume, center_max):
		volume = np.array(volume)
		volume = np.clip(volume,0, np.max(volume))
		volume *= center_max
		return volume

	def find_ac_affine(self, pid):
		ac = nib.load(os.path.join(self.source_dir,pid,'AC.nii'))
		nac = np.array(nib.load(os.path.join(self.source_dir,pid,'NAC.nii')).dataobj)
		if 'GE' in self.loca:
			slice_num = int(nac.shape[2] // (2/2.79))
		elif 'UI' in self.loca:
			slice_num = int(nac.shape[2] // (2/2.68))
		return ac.affine, slice_num, nac

	def minmax_normalization_center(self, img, center_max):
		center_min = 0.0
		out = (img - center_min) / (center_max - center_min)
		return np.clip(out, 0, 1)

	def minmax_normalization_individual(self, img):
		center_min = 0.0
		return (img - center_min) / (np.max(img) - center_min)

	def cut_pad_downsample(self, image, downsample_order, scale, width):
		print('\tDownsampling')
		img = image.copy()
		### Adjusting slice thickness to be 2, which is consistent to SH_Vision
		if 'Vision' in self.loca:
			if 'Bern_Vision' in self.loca:
				slice_num = int(img.shape[2] // (2/1.65))
				img = skitran.resize(img, (img.shape[0], img.shape[1], slice_num), order=1)
			else:
				slice_num = int(img.shape[2])
			if img.shape[2] < 448:
				img = np.pad(img, ((4,4),(4,4),(0,448-img.shape[2])),'constant', constant_values=(0, 0))
			else:
				img = img[:,:,-448:]
				img = np.pad(img, ((4,4),(4,4),(0,0)),'constant', constant_values=(0, 0))
			img = skitran.resize(img, (img.shape[0] // scale, img.shape[1] // scale, img.shape[2] // scale), order=downsample_order)
			out = img
			# out = np.pad(img, ((0, width-img.shape[0]), (0, width-img.shape[1]), (0, width-img.shape[2])), 'constant', constant_values=(0,0))
		else:
			if 'GE' in self.loca:
				slice_num = int(img.shape[2] // (2/2.79))
			elif 'UI' in self.loca:
				slice_num = int(img.shape[2] // (2/2.68))
			img = skitran.resize(img, (img.shape[0], img.shape[1], slice_num), order=1)
			if img.shape[2] < 448:
				img = np.pad(img, ((0,0),(0,0),(0,448-img.shape[2])),'constant', constant_values=(0, 0))
			else:
				img = img[:,:,-448:]
			img = skitran.resize(img, (width-2, width-2, img.shape[2] // scale), order=downsample_order)
			out = np.pad(img, ((1,1),(1,1),(0,0)),'constant', constant_values=(0, 0))
		return out, slice_num

	def upsample_apply_ratio_map(self, network_out, slice_num, NAC):
		print('\tUpsampling')
		scale = 4
		low_res = network_out.copy()
		low_res = low_res[1:-1,1:-1,:]
		if 'Vision' in self.loca:
			high_res = skitran.resize(low_res, (low_res.shape[0] * scale, low_res.shape[1] * scale, low_res.shape[2] * scale), order=self.upsample_order)
			if slice_num <= 448:
				out_ratio = high_res[:, :, :slice_num]
			else:
				out_ratio = high_res[:, :, :]
			out_slice = out_ratio.shape[2]
			if 'Bern_Vision' in self.loca:
				if slice_num <= 448:
					out_slice = NAC.shape[2]
				else:
					out_slice = int(out_ratio.shape[2] // (1.65/2))
				out_ratio = skitran.resize(out_ratio, (out_ratio.shape[0], out_ratio.shape[1], out_slice), order=1)
		else:
			if 'GE' in self.loca:
				out_dim = 256
				out_slice_ratio = (2.79/2)
			elif 'UI' in self.loca:
				out_dim = 150
				out_slice_ratio = (2.68/2)
			high_res = skitran.resize(low_res, (out_dim, out_dim, low_res.shape[2] * scale), order=self.upsample_order)
			if slice_num <= 448:
				out_ratio = high_res[:, :, :slice_num]
				out_slice = NAC.shape[2]
			else:
				out_ratio = high_res[:, :, :]
				out_slice = int(out_ratio.shape[2] // out_slice_ratio)
			out_ratio = skitran.resize(out_ratio, (out_ratio.shape[0], out_ratio.shape[1], out_slice), order=1)
		nac = NAC[:,:,-out_slice:]
		out_ratio[nac < 1] = 1
		return nac*out_ratio

	def resemble_from_h5(self):
		save_path = os.path.join(self.save_dir,self.loca)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
			os.makedirs(os.path.join(save_path,'NAC'))
		data = h5py.File(self.data_path, 'r')
		filenames = np.array(data[self.domain+'_filenames']).flatten()
		filenames = [x.decode('utf-8') for x in filenames]

		for i in range(len(filenames)):
			pid = filenames[i]
			print('{}/{}--{}--start'.format(i+1,len(filenames),pid))
			h5_nac = np.array(data.get(self.domain+'_NAC_PET')[i])
			x_slice = np.expand_dims(h5_nac, axis=0)
			x_slice = np.expand_dims(x_slice, axis=0)
			out_slice = self.generator.predict(x_slice)[0,0,:,:,:]
			affine, slice_num, source_nac = self.find_ac_affine(pid)
			out_volume = self.de_normalization(out_slice, self.center_max[0])
			out_ac = self.upsample_apply_ratio_map(out_volume, slice_num, source_nac)
			nib.save(nib.Nifti1Image(out_ac, affine=affine),
					 os.path.join(save_path, '{}_ac_gen.nii'.format(pid)))
			print('{}/{}--{}--done'.format(i+1,len(filenames),pid))
		data.close()

	def resemble_from_nii(self):
		save_path = os.path.join(self.save_dir,self.loca)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		count = 0
		for pid in [x for x in os.listdir(self.nii_data_path) if not x.startswith('.')]:
			count += 1
			print('{}/{}--{}--start'.format(count,len([x for x in os.listdir(self.nii_data_path) if not x.startswith('.')]),pid))
			ac = nib.load(os.path.join(self.nii_data_path,pid,'AC.nii'))
			affine = ac.affine
			nac = np.array(nib.load(os.path.join(self.nii_data_path,pid,'NAC.nii')).dataobj)
			input_nac, slice_num = self.cut_pad_downsample(nac, downsample_order=4, scale=4, width=112)
			input_nac = self.minmax_normalization_center(input_nac, self.center_max[1])
			x_slice = np.expand_dims(input_nac, axis=0)
			x_slice = np.expand_dims(x_slice, axis=0)
			out_volume = self.generator.predict(x_slice)[0,0,:,:,:]
			out_volume = self.de_normalization(out_volume, self.center_max[0])
			out_ac = self.upsample_apply_ratio_map(out_volume, slice_num, nac)
			nib.save(nib.Nifti1Image(out_ac, affine=affine),
					 os.path.join(save_path, '{}_ac_gen.nii'.format(pid)))
			print('\t{}--done'.format(pid))

if __name__ == '__main__':
	eva = Resemble()
	eva.resemble_from_nii()
