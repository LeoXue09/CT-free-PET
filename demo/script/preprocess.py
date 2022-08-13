import os
import numpy as np
import dicom2nifti
import shutil
import h5py
import tables
import nibabel as nib
from sklearn.model_selection import KFold
import logging
import time
import pydicom as dicom
import pandas as pd
from datetime import datetime
import skimage.transform as skitran

def get_SUV_ratio(source_dir, pid, loca):
	if loca == 'Quadra':
		for sub in [x for x in os.listdir(os.path.join(source_dir,pid)) if not x.startswith('.')]:
			for moda in [x for x in os.listdir(os.path.join(source_dir,pid,sub)) if not x.startswith('.')]:
				if not 'Raw' in moda and not 'CT' in moda:
					dirs = os.path.join(source_dir,pid,sub,moda)
					try:
						ds = dicom.read_file(os.path.join(dirs,os.listdir(dirs)[10]))
					except:
						print('{}-{}---error'.format(pid,moda))
					else:
						weight = float(ds[0x0010, 0x1030].value) * 1000
						dose = float(ds[0x0054, 0x0016][0][0x0018, 0x1074].value)
						out_ratio = weight/dose
						if out_ratio:
							break	
	else:
		for moda in os.listdir(os.path.join(source_dir,pid)):
			if not 'CT' in moda:
				dirs = os.path.join(source_dir,pid,moda)
				try:
					ds = dicom.read_file(os.path.join(dirs,os.listdir(dirs)[10]))
				except:
					print('{}-{}---error'.format(pid,moda))
				else:
					weight = float(ds[0x0010, 0x1030].value) * 1000
					dose = float(ds[0x0054, 0x0016][0][0x0018, 0x1074].value)
					out_ratio = weight/dose
					if out_ratio:
						break
	return out_ratio

def compute_ratio(NAC, AC):
	input_x = NAC.copy()
	input_x[input_x < 1] = 1
	return AC/input_x

def cut_pad_downsample(image, downsample_order, scale, width, loca):
	img = image.copy()
	### Adjusting slice thickness to be 2, which is consistent to SH_Vision
	if 'Vision' in loca:
		if 'Bern_Vision' in loca:
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
	else:
		if 'GE' in loca:
			slice_num = int(img.shape[2] // (2/2.79))
		elif 'UI' in loca:
			slice_num = int(img.shape[2] // (2/2.68))
		img = skitran.resize(img, (img.shape[0], img.shape[1], slice_num), order=1)
		if img.shape[2] < 448:
			img = np.pad(img, ((0,0),(0,0),(0,448-img.shape[2])),'constant', constant_values=(0, 0))
		else:
			img = img[:,:,-448:]
		img = skitran.resize(img, (width-2, width-2, img.shape[2] // scale), order=downsample_order)
		out = np.pad(img, ((1,1),(1,1),(0,0)),'constant', constant_values=(0, 0))
	return out

def find_max(data_dir, loca, SUV, source_dir, downsample_order, scale, width):
	ac_max, nac_max,ratio_max = 0,0,0
	count = 1 
	for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
		print('{}/{}--{}--start'.format(count, len(os.listdir(data_dir)), pid))
		count += 1
		nac_temp, ac_temp = 0,0
		nac, ac = None, None
		if SUV == True:
			suv_ratio = get_SUV_ratio(source_dir,pid,loca)
		else:
			suv_ratio = 1
		nac = np.array(nib.load(os.path.join(data_dir,pid,'NAC.nii')).dataobj)
		nac_post = cut_pad_downsample(nac, downsample_order, scale, width, loca)
		nac_temp = np.max(nac_post) * suv_ratio
		if nac_temp > nac_max:
			nac_max = nac_temp
		ac = np.array(nib.load(os.path.join(data_dir,pid,'AC.nii')).dataobj)
		ac_post = cut_pad_downsample(ac, downsample_order, scale, width, loca)
		ac_temp = np.max(ac_post)
		if ac_temp > ac_max:
			ac_max = ac_temp
		ratio = compute_ratio(nac,ac)
		ratio_post = cut_pad_downsample(ratio, downsample_order, scale, width, loca)
		ratio_temp = np.max(ratio_post)
		orr = np.count_nonzero(ratio_post>100)
		print('ratio: ', ratio_post.shape, ratio_temp, orr)
		if ratio_temp > ratio_max:
			ratio_max = ratio_temp
	print('AC_PET--max: {}\tNAC_PET--max: {}\tratio--max: {}'.format(ac_max, nac_max, ratio_max))

def read_dicom_header(source_dir, loca):
	test_info = pd.read_csv('/media/uni/ST_2/Attenuation/Data/SH_Vision/info.csv', dtype=object)
	test_pid = test_info['B'].values
	FMT = '%H%M%S'
	df = pd.DataFrame()
	count = 0
	for pid in [x for x in os.listdir(source_dir) if not x.startswith('.')]:
		count += 1
		print('{}--{}---start'.format(loca,pid), count)
		if 'SH_Vision' in loca:
			dirs = os.path.join(source_dir,pid,'AC')
		elif 'Bern' in loca:
			dirs = os.path.join(source_dir,pid,'AC_PET')
		elif loca == 'SH_GE':
			dirs = os.path.join(source_dir,pid,'ST0','SE4')
		elif loca == 'SH_UI':
			tar = [x for x in os.listdir(os.path.join(source_dir,pid)) if 'AC-WB-PET' in x][0]
			dirs = os.path.join(source_dir,pid,tar)
		for file in [x for x in os.listdir(dirs) if not x.startswith('.')]:
			try:
				ds = dicom.read_file(os.path.join(dirs,file))
				ManufacturerModelName = ds[0x0008, 0x1090].value
				dose = int(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
				inj_start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
				inj_start_time = inj_start_time.split('.')[0]
				acquisition_time = ds[0x0008, 0x0032].value.split('.')[0]
				post_inj_time = datetime.strptime(acquisition_time, FMT) - datetime.strptime(inj_start_time, FMT)
				days, seconds = post_inj_time.days, post_inj_time.seconds
				post_inj_minute = seconds // 60
				radionuclide_name = ds[0x0054, 0x0016][0][0x0018, 0x0031].value
				gender = ds[0x0010, 0x0040].value
				try:
					age = int(ds[0x0010, 0x1010].value[1:-1])
				except:
					age = None
				weight = int(ds[0x0010, 0x1030].value)
				df = df.append({'PID': pid, 'ManufacturerModelName': ManufacturerModelName, 'Dose': dose,
								'post_inj_time': post_inj_minute, 'radionuclide_name': radionuclide_name,
								'gender': gender, 'age': age, 'weight': weight}, ignore_index=True)
				break
			except AttributeError:
				print('{}---error'.format(pid))
				continue
				
		df = df[['PID', 'ManufacturerModelName', 'Dose', 'post_inj_time', 'radionuclide_name',
				'gender', 'age', 'weight']]
		save_path = '/media/uni/ST_2/Attenuation/Data/dicom_header'
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		df.to_csv(os.path.join(save_path,'{}_dicom_header.csv'.format(loca)), index=False)


def minmax_normalization_center(img, center_max):
	center_min = 0.0
	return (img - center_min) / (center_max - center_min)

def minmax_normalization_individual(img):
	center_min = 0.0
	return (img - center_min) / (np.max(img) - center_min)

def generate_h5_3d(data_dir, save_path, normalization, loca, SUV, source_dir, center_max, cross_validation, fold, downsample_order,scale, width):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	def write_h5(target_data_dir,dataset,target_file,pid_idx):
		NAC_PET_pool = target_file.create_earray(target_file.root,dataset+'_NAC_PET',tables.Float32Atom(),
                                      			shape=(0,width,width,width),expectedrows=1000000)
		ratio_pool = target_file.create_earray(target_file.root,dataset+'_ratio',tables.Float32Atom(),
                                      			shape=(0,width,width,width),expectedrows=1000000)
		filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
											shape=(0,1),expectedrows=1000000)

		count = 0
		for pid in pid_idx:
			print('{}--start'.format(pid))
			nac = np.array(nib.load(os.path.join(data_dir,pid,'NAC.nii')).dataobj)
			ac = np.array(nib.load(os.path.join(data_dir,pid,'AC.nii')).dataobj)
			if SUV == True:
				suv_ratio = get_SUV_ratio(source_dir,pid,loca)
			else:
				suv_ratio = 1
			ac = ac * suv_ratio
			nac = nac * suv_ratio
			ratio = compute_ratio(nac,ac)
			ac = cut_pad_downsample(ac, downsample_order,scale, width,loca)
			nac = cut_pad_downsample(nac, downsample_order,scale, width,loca)
			ratio = cut_pad_downsample(ratio, downsample_order,scale, width,loca)
			if normalization == 'center':
				nac = minmax_normalization_center(nac, center_max[1])
				# ac = minmax_normalization_center(ac, center_max[0])
				ratio = minmax_normalization_center(ratio, center_max[2])
			elif normalization == 'individual':
				nac = minmax_normalization_individual(nac)
				# ac = minmax_normalization_individual(ac)
			elif normalization == 'NAC_only':
				nac = minmax_normalization_center(nac, center_max[1])
			elif normalization == 'None':
				pass

			NAC_PET_pool.append(np.expand_dims(nac, axis=0))
			ratio_pool.append(np.expand_dims(ratio, axis=0))
			filenames_pool.append(np.expand_dims([pid], axis=0))
			print(np.max(ac),np.max(nac),np.max(ratio))
			count += 1
			print(os.path.basename(target_data_dir)+'__'+dataset+': {}/{}'.format(count, len(pid_idx)))

	patient_list = [x for x in os.listdir(data_dir) if not x.startswith('.')]

	if cross_validation == True:
		pass

	elif cross_validation == False:
		target_file = tables.open_file(os.path.join(save_path, 'data.h5'), mode='w')
		if loca == 'SH_Vision':
			df = pd.read_csv('/media/uni/ST_2/Attenuation/Data/SH_Vision/info.csv', dtype=object)
			valid_pid = df['B'].values
			train_pid = list(set(patient_list)-set(valid_pid))
		else:
			train_pid = patient_list[:int(0.1*len(patient_list))]
			valid_pid = list(set(patient_list)-set(train_pid))
		write_h5(data_dir, 'train', target_file, train_pid)
		write_h5(data_dir, 'valid', target_file, valid_pid)
		target_file.close()

	elif cross_validation == None:
		target_file = tables.open_file(os.path.join(save_path, 'data.h5'), mode='w')
		test_pid = patient_list
		write_h5(data_dir, 'test', target_file, test_pid)
		target_file.close()

	else:
		pass

def shuffle_h5_3d(h5_dir, cross_validation, width):
	def shuffle(dataset, data, target_file):
		shape = data.get(dataset + '_NAC_PET').shape
		NAC_PET_pool = target_file.create_earray(target_file.root,dataset+'_NAC_PET',tables.Float32Atom(),
                                      			shape=(0,width,width,width),expectedrows=1000000)
		ratio_pool = target_file.create_earray(target_file.root,dataset+'_ratio',tables.Float32Atom(),
                                      			shape=(0,width,width,width),expectedrows=1000000)
		filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
											shape=(0,1),expectedrows=1000000)

		index = np.arange(0, shape[0])
		if dataset == 'train':
			np.random.seed(1314)
			np.random.shuffle(index)

		count = 0
		for idx in index:
			count += 1
			print('\t'+dataset+': {}/{}'.format(count, len(index)))
			# AC_PET_pool.append(np.expand_dims(data.get(dataset+'_AC_PET')[idx,:,:,:], axis=0))
			NAC_PET_pool.append(np.expand_dims(data.get(dataset+'_NAC_PET')[idx,:,:,:], axis=0))
			ratio_pool.append(np.expand_dims(data.get(dataset+'_ratio')[idx,:,:,:], axis=0))
			filenames_pool.append(np.expand_dims(data.get(dataset+'_filenames')[idx], axis=0))

	for (dirpath, dirnames, filenames) in os.walk(h5_dir):
		for name in filenames:
			if name.endswith('.h5'):
				h5_path = os.path.join(dirpath, name)
				data = h5py.File(h5_path, mode='r')
				save_path = os.path.join(dirpath, os.path.splitext(name)[0]+'_shuffled.h5')
				target_file = tables.open_file(save_path,mode='w')
				if cross_validation == True:
					shuffle('train',data,target_file)
					shuffle('valid',data,target_file)
				else:
					shuffle('train',data,target_file)
				data.close()
				target_file.close()

def SH_Vision_main():
	data_dir = '/media/uni/ST_2/Attenuation/Data/SH_Vision_converted_all'
	save_path = '/media/uni/ST_2/Attenuation/Data/h5_file/SH_Vision_ratio_downorder_4_112_prepad'
	SUV = False
	source_dir = '/media/uni/ST_2/Attenuation/Data/SH_Vision/SH_Vision_AC_NAC'
	loca = 'SH_Vision'
	downsample_order = 4
	scale = 4
	width = 112
	# read_dicom_header(source_dir,loca)
	# find_max(data_dir, loca=loca, SUV=SUV, source_dir=source_dir, downsample_order=downsample_order, scale=scale, width=width)
	# raise ValueError
	cross_validation = False
	normalization = 'None'
	### SH_Vision_ratio_downorder_4_112_prepad: AC_PET--max: 1400867.625 NAC_PET--max: 367249.78125 ratio--max: 33.927001953125
	center_max = (1400868, 367250, 34)
	generate_h5_3d(data_dir, save_path=save_path, normalization=normalization, loca=loca, SUV=SUV, source_dir=source_dir,
				   center_max=center_max, cross_validation=cross_validation, fold=None, downsample_order=downsample_order,scale=scale,width=width)
	shuffle_h5_3d(save_path, cross_validation=cross_validation,width=width)


if __name__ == '__main__':
	SH_Vision_main()