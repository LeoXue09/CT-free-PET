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
import pydicom as dicom
from scipy import spatial
from matplotlib import cm
from scipy.interpolate import interpn
from sklearn.linear_model import SGDRegressor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import skimage.transform as skitran

class Evaluate():
	def __init__(self):
		self.loca = 'SH_Vision'
		# self.loca = 'Bern_Vision'
		# self.loca = 'SH_GE'
		# self.loca = 'SH_UI'

		root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
		self.source_nii_dir = self.source_dicom_dir = os.path.join(root_dir, '../data/{}'.format(self.loca))
		self.evaluate_dir = self.save_dir = os.path.join(root_dir, '../result')

	def compute_metrics(self, real_input, pred_input):
		real = real_input.copy()
		real[real<1] = 0
		pred = pred_input.copy()
		pred[real<1] = 0
		mse = np.mean(np.square(real-pred))
		nrmse = np.sqrt(mse) / (np.max(real)-np.min(real))
		ok_idx = np.where(real!=0)
		mape = np.mean(np.abs((real[ok_idx] - pred[ok_idx]) / real[ok_idx]))
		PIXEL_MAX = np.max(real)
		psnr = 20*np.log10(PIXEL_MAX / np.sqrt(np.mean(np.square(real-pred))))
		real_norm = real / float(np.max(real))
		pred_norm = pred / float(np.max(pred))
		ssim = compare_ssim(real_norm, pred_norm)
		return mse, nrmse, mape, psnr, ssim

	def downsample(self, img, downsample_order, scale):
		img = skitran.resize(img, (img.shape[0] // scale, img.shape[1] // scale, img.shape[2] // scale), order=downsample_order)
		return img

	def find_ac_affine(self, pid, cut_slice):
		ac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,'AC.nii')).dataobj)
		nac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,'NAC.nii')).dataobj)
		slice_num = ac.shape[2]
		if ac.shape[2] == cut_slice:
			pass
		else:
			ac = ac[:,:,-cut_slice:]
			nac = nac[:,:,-cut_slice:]
		return ac, nac

	def evalaute(self):
		eva_path = os.path.join(self.evaluate_dir,self.loca)
		nac_path = os.path.join(self.save_dir,self.loca,'NAC')
		general_save_path = os.path.join(self.save_dir,'general_result', self.loca)
		if not os.path.exists(general_save_path):
			os.makedirs(general_save_path)
		gen_ac_list = [x for x in os.listdir(eva_path) if os.path.isfile(os.path.join(eva_path,x))]
		pids, mse_list, nrmse_list, mape_list, psnr_list, ssim_list, similarity_list = [],[],[],[],[],[],[]
		ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list, ori_similarity_list = [],[],[],[],[],[]
		for count, file in enumerate(gen_ac_list):
			pid = file.split('_ac_gen')[0]
			print('{}/{}--{}--start'.format(count+1,len(gen_ac_list),pid))
			individual_save_path = os.path.join(self.save_dir,'individual_result', self.loca, pid)
			if not os.path.exists(individual_save_path):
				os.makedirs(individual_save_path)
			gen_ac = np.array(nib.load(os.path.join(eva_path,file)).dataobj)
			if self.loca in ['SH_GE', 'SH_UI']:
				gen_ac *= self.scanner_correction_ratio
			cut_slice = gen_ac.shape[2]
			ori_ac, ori_nac = self.find_ac_affine(pid, cut_slice)
			print('\tCalculating metrics')
			mse, nrmse, mape, psnr, ssim = self.compute_metrics(ori_ac, gen_ac)
			similarity = self.plot_dvh(ori_ac, gen_ac, ori_nac, pid, individual_save_path, cut_slice, modality='PRED')
			ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(ori_ac, ori_nac)
			ori_similarity = self.plot_dvh(ori_ac, ori_nac, ori_nac, pid, individual_save_path, cut_slice, modality='NAC')
			print(ori_nrmse, nrmse)
			mse_list.append(mse)
			nrmse_list.append(nrmse* 1e2)
			psnr_list.append(psnr)
			ssim_list.append(ssim)
			mape_list.append(mape)
			similarity_list.append(similarity)
			ori_mse_list.append(ori_mse)
			ori_nrmse_list.append(ori_nrmse* 1e2)
			ori_psnr_list.append(ori_psnr)
			ori_ssim_list.append(ori_ssim)
			ori_mape_list.append(ori_mape)
			ori_similarity_list.append(ori_similarity)
			pids.append(pid)

			df = pd.DataFrame({'PID': pids, 'MSE': mse_list, 'ori_MSE': ori_mse_list, 'NRMSE %': nrmse_list, 
							   'ori_NRMSE %': ori_nrmse_list, 'MAPE': mape_list, 'ori_MAPE': ori_mape_list, 
							   'PSNR': psnr_list, 'ori_PSNR': ori_psnr_list, 'SSIM': ssim_list, 'ori_SSIM': ori_ssim_list,
							   'DVH_similarity': similarity_list, 'ori_DVH_similarity': ori_similarity_list})
			df = df.append({'PID': 'Mean Value', 'MSE': np.mean(mse_list),'ori_MSE': np.mean(ori_mse_list),
							'NRMSE %': np.mean(nrmse_list), 'ori_NRMSE %': np.mean(ori_nrmse_list),
							'MAPE': np.mean(mape_list),'ori_MAPE': np.mean(ori_mape_list),
							'PSNR': np.mean(psnr_list), 'ori_PSNR': np.mean(ori_psnr_list),
							'SSIM': np.mean(ssim_list), 'ori_SSIM': np.mean(ori_ssim_list), 
							'DVH_similarity': np.mean(similarity_list), 'ori_DVH_similarity': np.mean(ori_similarity_list)}, ignore_index=True)
			df = df[['PID', 'MSE','ori_MSE', 'MAPE','ori_MAPE','NRMSE %','ori_NRMSE %', 'PSNR','ori_PSNR', 'SSIM','ori_SSIM','DVH_similarity','ori_DVH_similarity']]
			df.to_csv(os.path.join(general_save_path,'{}.csv'.format(self.loca)), index=False)

	def compute_DVH_points(self, mask, img, scaling=False, ori_max=None, stops=None):
		interval = 100
		voxels = []
		idx = np.argwhere(mask != 0)
		for i in idx:
			x,y,z = i
			voxels.append(img[x,y,z])
		voxels = np.array(voxels)
		if scaling == True:
			fac = ori_max / np.max(voxels)
			voxels = voxels * fac
		volumn = np.size(voxels)
		if stops is None:
			stops = np.arange(0,np.max(voxels)*(1+2/interval),np.max(voxels)/interval)
		x, y = [],[]
		for s in stops:
			cut = np.size(np.where(voxels >= s)[0])
			y.append(round(cut/volumn*100,1))
			x.append(round(s,1))
		return x,y, np.max(voxels), stops

	def plot_dvh(self, ori_dose, pre_dose, ori_nac, pid, save_path, cut_slice, modality):
		print('\tPlotting DVH---{}'.format(modality))
		save_path = os.path.join(self.save_dir,'individual_result', self.loca, pid, 'organ')
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		organ_list = ['liver', 'spleen', 'left_kidney', 'right_kidney']
		mask_label_list = [3, 13, 6, 7]
		color_list = ['red','green','blue','orange']
		fig1, ax1 = plt.subplots()
		ax1.set_title('Dose Volume Histogram')
		max_dic = {}

		try:
			all_mask = np.array(nib.load(os.path.join(self.source_nii_dir,'{}/seg.nii'.format(pid))).dataobj)
		except:
			print('{} without mask'.format(pid))
			return 1
		else:
			similarity = []
			for i, organ in enumerate(organ_list):
				if self.loca == 'Quadra':
					all_mask = all_mask[:,:,:cut_slice]
				elif 'Vision' in self.loca:
					all_mask = all_mask[:,:,-cut_slice:]
				mask = all_mask.copy()
				mask[np.where(mask>(mask_label_list[i]+0.001))] = 0
				mask[np.where(mask<(mask_label_list[i]-0.001))] = 0
				try:
					x1,y1,ori_max, stops = self.compute_DVH_points(mask, ori_dose, scaling=False)
				except:
					continue
				else:
					max_dic[organ] = ori_max
					ax1.plot(x1,y1,':',color=color_list[i])
					x2,y2,_,_ = self.compute_DVH_points(mask, pre_dose, scaling=False, ori_max=max_dic[organ], stops=stops)
					ax1.plot(x2,y2,'-',color=color_list[i])
					similarity.append(1 - spatial.distance.cosine(y1,y2))
			ax1.set_xlabel('Dose (Bq/mL)')
			ax1.set_ylabel('Volume percentage (%)')
			if modality == 'NAC':
				fig1.savefig(os.path.join(save_path, 'DVH_nac.png'),bbox_inches='tight')
			elif modality == 'PRED':
				fig1.savefig(os.path.join(save_path, 'DVH_pred.png'),bbox_inches='tight')
			plt.close()
			return np.mean(similarity)

	def normal_equation(self, x, y):
		print('\tNormal equation')
		# X = x.reshape(-1,1)
		x_bias = np.ones((len(x),1))
		x = np.reshape(x,(len(x),1))
		x = np.append(x_bias,x,axis=1)
		x_transpose = np.transpose(x)
		x_transpose_dot_x = x_transpose.dot(x)
		print('\t\tinversing')
		temp_1 = np.linalg.inv(x_transpose_dot_x)
		temp_2 = x_transpose.dot(y)
		theta = temp_1.dot(temp_2)
		theta_shaped = np.reshape(theta,(len(theta),1))
		y_hat = np.dot(x,theta_shaped)
		y_hat = y_hat.flatten()
		r2 = np.corrcoef(y_hat, y)[0, 1]**2
		return theta, r2

	def plot_scatterheat(self):
		eva_path = os.path.join(self.evaluate_dir,self.loca)
		gen_ac_list = [x for x in os.listdir(eva_path) if os.path.isfile(os.path.join(eva_path,x))]
		for count, file in enumerate(gen_ac_list):
			pid = file.split('_ac_gen')[0]
			print('{}/{}--{}--start'.format(count+1,len(gen_ac_list),pid))
			gen_ac = np.array(nib.load(os.path.join(eva_path,file)).dataobj)
			if self.loca in ['SH_GE', 'SH_UI']:
				gen_ac *= self.scanner_correction_ratio
			cut_slice = gen_ac.shape[2]
			ori_ac, ori_nac = self.find_ac_affine(pid, cut_slice)
			if 'Vision' in self.loca:
				downsample_scale = 4
			elif self.loca == 'SH_GE':
				downsample_scale = 2
			elif self.loca == 'SH_UI':
				downsample_scale = 1
			print('\tDownsampling')
			gen_ac = self.downsample(gen_ac, downsample_order=4, scale=downsample_scale)
			ori_ac = self.downsample(ori_ac, downsample_order=4, scale=downsample_scale)
			ori_nac = self.downsample(ori_nac, downsample_order=4, scale=downsample_scale)
			print('\tPlotting Scatterheat---PRED')
			save_path = os.path.join(self.save_dir,'individual_result', self.loca, pid)
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			x = ori_ac.flatten()
			X = x.reshape(-1,1)
			y1 = gen_ac.flatten()
			y2 = ori_nac.flatten()
			print('\t\tHistogramming')
			data , x_e, y_e = np.histogram2d(x, y1, bins = 20, density = True)
			print('\t\tInterpolating')
			z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y1]).T , method = "splinef2d", bounds_error = False)
			#To be sure to plot all data
			z[np.where(np.isnan(z))] = 0.0
			# Sort the points by density, so that the densest points are plotted last
			idx = z.argsort()
			sort_x, sort_y1, z = x[idx], y1[idx], z[idx]
			print('\t\tPlotting scatter')
			fig, ax1 = plt.subplots(figsize=[10,10])
			divider = make_axes_locatable(ax1)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			im = ax1.scatter(sort_x, sort_y1, c=z, s=10, cmap='rainbow', alpha=0.5)
			ax1.set_ylim(0, max(x))
			ax1.set_xlim(0, max(x))
			ax1.set_aspect('equal', adjustable='box')
			fig.colorbar(im, cax=cax, orientation='vertical')
			# plt.savefig(os.path.join(save_path, 'PRED_scatterheat_m: {},b: {}.png'.format(m,b)), bbox_inches='tight')
			fig.savefig(os.path.join(save_path, 'PRED_scatterheat.png'), bbox_inches='tight')
			plt.close()
			print('\tPlotting Scatterheat---NAC')
			print('\t\tHistogramming')
			data , x_e, y_e = np.histogram2d(x, y2, bins = 20, density = True)
			print('\t\tInterpolating')
			z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y2]).T , method = "splinef2d", bounds_error = False)
			z[np.where(np.isnan(z))] = 0.0
			idx = z.argsort()
			sort_x, sort_y2, z = x[idx], y2[idx], z[idx]
			print('\t\tPlotting scatter')
			fig, ax1 = plt.subplots(figsize=[10,10])
			divider = make_axes_locatable(ax1)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			im = ax1.scatter(sort_x, sort_y2, c=z, s=10, cmap='rainbow', alpha=0.5)
			ax1.set_ylim(0, max(x))
			ax1.set_xlim(0, max(x))
			ax1.set_aspect('equal', adjustable='box')
			fig.colorbar(im, cax=cax, orientation='vertical')
			# plt.savefig(os.path.join(save_path, 'NAC_scatterheat_m: {},b: {}.png'.format(m,b)), bbox_inches='tight')
			fig.savefig(os.path.join(save_path, 'NAC_scatterheat.png'), bbox_inches='tight')
			plt.close()
			print('\tFitting lines---PRED')
			theta1, r2_1 = self.normal_equation(x, y1)
			b1,w1 = theta1
			theta2, r2_2 = self.normal_equation(x, y2)
			b2, w2 = theta2
			df = pd.DataFrame({'PID': [pid], 'Pred_w': [w1], 'Pred_b': [b1], 'Pred_r2': [r2_1], 'NAC_w': [w2], 'NAC_b': [b2], 'NAC_r2': [r2_2]})
			df = df[['PID', 'Pred_w', 'Pred_b', 'Pred_r2', 'NAC_w', 'NAC_b', 'NAC_r2']]
			df.to_csv(os.path.join(save_path,'fitted_line_parameter.csv'), index=False)

if __name__ == '__main__':
	eva = Evaluate()
	eva.evalaute()
	eva.plot_scatterheat()