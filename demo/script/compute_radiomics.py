# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import nibabel as nib
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk 
import radiomics
import pydicom as dicom
radiomics.logger.setLevel('ERROR')

class Evaluate():
	def __init__(self):
		self.selected_feature_list = ['glrlm_HighGrayLevelRunEmphasis', 'glszm_ZonePercentage','firstorder_RootMeanSquared','firstorder_90Percentile',
									  'firstorder_Median','glcm_JointAverage']
		self.SUV_feature_list = ['SUV mean', 'SUV max', 'Total Lesion Glycolysis']
		root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
		self.dicom_header_dir = os.path.join(root_dir, '../data/dicom_header')
		self.test_PID_df = pd.read_csv(os.path.join(self.dicom_header_dir,'test_PID.csv'))
		self.evaluate_dir = os.path.join(root_dir, '../result')
		self.nii_dir = os.path.join(root_dir, '../data')
		self.save_dir = os.path.join(self.evaluate_dir, 'radiomics')
		self.organ_list = ['liver', 'lung', 'kidney']
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

	def get_data(self, loca, pid):
		gen_ac_dir = os.path.join(self.evaluate_dir, loca)
		nii_dir = os.path.join(self.nii_dir,loca)
		dicom_header_df = os.path.join(self.dicom_header_dir, loca,'{}.csv'.format(loca))
		df = pd.read_csv(dicom_header_df)
		SUV_ratio = df.loc[df['PID'] == pid, 'weight'].values * 1000 / df.loc[df['PID'] == pid, 'Dose'].values
		gen_ac_path = os.path.join(gen_ac_dir, '{}_ac_gen.nii'.format(pid))

		return SUV_ratio[0], nii_dir, gen_ac_path

	def percentage_error(self, real, pred):
		return abs(pred - real) / abs(real) * 100

	def suv_feature(self, img, mask):
		volume = np.count_nonzero(mask)
		crop_idx = np.nonzero(mask)
		suv_mean = np.mean(img[crop_idx])
		suv_max = np.max(img[crop_idx])
		tlg = volume * suv_mean
		return suv_mean, suv_max, tlg

	def compute_radiomics(self):
		extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(binCount=100)
		if filter:
			extractor.enableImageTypeByName('LoG', customArgs={'sigma': [3.0]})
		all_df = pd.DataFrame()
		for index, row in self.test_PID_df.iterrows():
			loca = row['Loca']
			pid = row['PID']
			SUV_ratio, nii_dir, gen_ac_path = self.get_data(loca, pid)
			ori_ac = np.array(nib.load(os.path.join(nii_dir,pid,'AC.nii')).dataobj) * SUV_ratio
			# ori_nac = np.array(nib.load(os.path.join(nii_dir,pid,'NAC.nii')).dataobj) * SUV_ratio
			sphere = np.array(nib.load(os.path.join(nii_dir,pid,'sphere.nii')).dataobj)
			gen_ac = np.array(nib.load(gen_ac_path).dataobj) * SUV_ratio
			if ori_ac.shape[2] != gen_ac.shape[2]:
				ori_ac = ori_ac[:,:,-gen_ac.shape[2]:]
				# ori_nac = ori_nac[:,:,-gen_ac.shape[2]:]
				sphere = sphere[:,:,-gen_ac.shape[2]:]
			ac_sitk = sitk.GetImageFromArray(ori_ac)
			# nac_sitk = sitk.GetImageFromArray(ori_nac)
			gen_ac_sitk = sitk.GetImageFromArray(gen_ac)
			for label, organ in enumerate(self.organ_list, start=1):
				print('{}-{}-{}---start'.format(loca,pid,organ))
				df = pd.DataFrame({'Location': [loca], 'PID': [pid], 'Organ': [organ]})
				column_list = ['Location','PID','Organ']
				organ_mask = np.where(sphere==label,sphere,0)
				organ_mask = np.where(organ_mask!=label,organ_mask,1)
				mask_sitk = sitk.GetImageFromArray(organ_mask)
				gen_ac_feature_vector = extractor.execute(gen_ac_sitk, mask_sitk)
				ac_feature_vector = extractor.execute(ac_sitk, mask_sitk)
				feature_list = []
				for f in self.selected_feature_list:
					feature_list.extend([s for s in ac_feature_vector.keys() if f in s])
				for feature in feature_list:
					feature_df = pd.DataFrame({feature: [self.percentage_error(ac_feature_vector[feature],gen_ac_feature_vector[feature])]})
					df = pd.concat([df, feature_df], axis=1)
					column_list.extend([feature])
				for i in range(len(self.SUV_feature_list)):
					feature_df = pd.DataFrame({self.SUV_feature_list[i]: [self.percentage_error(self.suv_feature(ori_ac, organ_mask)[i], self.suv_feature(gen_ac,organ_mask)[i])]})
					df = pd.concat([df, feature_df], axis=1)
					column_list.extend([self.SUV_feature_list[i]])
				all_df = all_df.append(df)
			all_df.to_csv(os.path.join(self.save_dir,'radiomics.csv'), index=False)



if __name__ == '__main__':
	eva = Evaluate()
	eva.compute_radiomics()
