from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import re
import random
import json

from stop_words import ENGLISH_STOP_WORDS

class M2E2DataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			csv,
			sentences,
			we,
			we_dim=300,
			max_words=30
	):
		"""
		Args:
		"""
		self.csv = pd.read_csv(csv)
		self.sentences = json.load(open(sentences))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		
	def __len__(self):
		return len(self.csv)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, sentences):
		rint = random.randint(0,len(sentences)-1)
		return self._words_to_we(self._tokenize_text(sentences[rint]))

	def _get_video(self, feat_2d_path,feat_3d_path):
		feat_2d = np.load(feat_2d_path)
		feat_3d = np.load(feat_3d_path)
		
		feat_2d = th.from_numpy(feat_2d).float()
		feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)

		feat_3d = th.from_numpy(feat_3d).float()
		feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)

		return th.cat((feat_2d, feat_3d))


	def __getitem__(self, idx):
		video_id = self.csv['video_id'].values[idx]
		feat_2d_path = self.csv['2d'].values[idx]
		feat_3d_path = self.csv['3d'].values[idx]
		video = self._get_video(feat_2d_path,feat_3d_path)
		text = self._get_text(self.sentences[video_id])
		return {'video': video, 'text': text, 'video_id': video_id}

class M2E2ASRDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			csv,
			videoID2feature_path,
			we,
			we_dim=300,
			max_words=30
	):
		"""
		Args:
		"""
		self.csv = pd.read_csv(csv)
		self.videoID2feature = json.load(open(videoID2feature_path))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		
	def __len__(self):
		return len(self.csv)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))

	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced


	def _get_video(self, feat_2d_path,feat_3d_path,start,end):
		feat_2d = self._get_slice(feat_2d_path,start,end,'2d')
		feat_3d = self._get_slice(feat_3d_path,start,end,'3d')
		feat_2d = th.from_numpy(feat_2d).float()
		feat_3d = th.from_numpy(feat_3d).float()

		if len(feat_2d) < 1 or len(feat_3d) < 1: 
			print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
		else:
			feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
			feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)

		return th.cat((feat_2d, feat_3d))


	def __getitem__(self, idx):
		video_id = self.csv['video_id'].values[idx]
		text = self.csv['text'].values[idx]
		start = self.csv['start'].values[idx]
		end = self.csv['end'].values[idx]

		feat_2d_path = self.videoID2feature[video_id]['2d']
		feat_3d_path = self.videoID2feature[video_id]['3d']

		video = self._get_video(feat_2d_path,feat_3d_path,start,end)
		text = self._get_text(text)
		return {'video': video, 'text': text, 'video_id': video_id}



class M2E2MILODataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			videoID2obj_feature_path, # /kiwi-data/users/shoya/AIDA/obj_det_features/m2e2_obj_detections_merged.json.json
			videoID2vid_feature_path, # /kiwi-data/users/shoya/AIDA/videoID2feature_paths.json
			we,
			we_dim=300,
			max_words=30,
			num_candidates=5
	):
		"""
		Args:
		"""
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.videoID2vid_feature = json.load(open(videoID2vid_feature_path))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		
	def __len__(self):
		return len(self.videoID2obj_feature)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))

	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced


	def _get_video(self, feat_2d_path,feat_3d_path,start,end):
		feat_2d = self._get_slice(feat_2d_path,start,end,'2d')
		feat_3d = self._get_slice(feat_3d_path,start,end,'3d')
		feat_2d = th.from_numpy(feat_2d).float()
		feat_3d = th.from_numpy(feat_3d).float()

		if len(feat_2d) < 1 or len(feat_3d) < 1: 
			print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
		else:
			feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
			feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)
		return th.cat((feat_2d, feat_3d))

	def _get_obj_features(self,features_path):
		obj_features = np.load(features_path)
		return th.from_numpy(obj_features[:self.num_candidates]).float()

	def __getitem__(self, idx):
		video_id_raw = list(self.videoID2obj_feature.keys())[idx] # formatted {vid_name}_{midframe_sec}
		vid_id = video_id_raw.rsplit('_')[0] + '.mp4'

		text = self.videoID2obj_feature[video_id_raw]['text']
		start = self.videoID2obj_feature[video_id_raw]['start_sec']
		end = self.videoID2obj_feature[video_id_raw]['end_sec']
		obj_features = self._get_obj_features(self.videoID2obj_feature[video_id_raw]['features_path'])

		feat_2d_path = self.videoID2vid_feature[vid_id]['2d']
		feat_3d_path = self.videoID2vid_feature[vid_id]['3d']

		video = self._get_video(feat_2d_path,feat_3d_path,start,end)
		processed_text = self._get_text(text)
		return {'video': video, 'text': processed_text, 'obj':obj_features, 'video_id': vid_id}
		#return {'video': videos, 'text': texts, 'obj':obj_features, 'video_id': vid_name}


class M2E2MILOManualLabelsDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			videoID2obj_feature_path, # /kiwi-data/users/shoya/AIDA/obj_det_features/m2e2_manual_labels/ten_feats_per_frame/m2e2_obj_detections_manually_labeled_data_10_feats.json
			video_clip_feature_dir,
			we,
			we_dim=300,
			max_words=30,
			num_candidates=5
	):
		"""
		Args:
		"""
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.video_clip_feature_dir = video_clip_feature_dir
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		
	def __len__(self):
		return len(self.videoID2obj_feature)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))

	def _get_video(self, feat_2d_path,feat_3d_path):
		feat_2d = np.load(feat_2d_path)
		feat_3d = np.load(feat_3d_path)
		
		feat_2d = th.from_numpy(feat_2d).float()
		feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)

		feat_3d = th.from_numpy(feat_3d).float()
		feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)

		return th.cat((feat_2d, feat_3d))

	def _get_obj_features(self,features_path):
		# TODO: sample features from across different frames
		obj_features_all = np.load(features_path)

		if obj_features_all.shape[0] == 1:
			# only one frame sampled, so just return that 
			output = th.from_numpy(obj_features_all[0][:self.num_candidates]).float()
		else:
			# multiple frames were sampled, so we should just pick top 1 one from each possible frame 
			output = []
			i = 0 
			while len(output) < self.num_candidates:
				for j in range(obj_features_all.shape[0]):
					output.append(obj_features_all[j][i])
				i+=1 
			output = np.stack(output) 
			output = th.from_numpy(output[:self.num_candidates]).float()
		return output

	def __getitem__(self, idx):
		video_id_raw = list(self.videoID2obj_feature.keys())[idx] 
		vid_name = video_id_raw + '.mp4'

		text = self.videoID2obj_feature[video_id_raw]['text']
		obj_features = self._get_obj_features(self.videoID2obj_feature[video_id_raw]['features_path'])

		feat_2d_path = os.path.join(self.video_clip_feature_dir,'2d',video_id_raw+'.npy')
		feat_3d_path = os.path.join(self.video_clip_feature_dir,'3d',video_id_raw+'.npy')

		video = self._get_video(feat_2d_path,feat_3d_path)
		processed_text = self._get_text(text)
		return {'video': video, 'text': processed_text, 'obj':obj_features, 'video_id': video_id_raw}


class M2E2MILOManualLabelsWithArticleDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			videoID2obj_feature_path, 
			video_clip_feature_dir,
			we,
			we_dim=300,
			max_words=30,
			num_candidates=5
	):
		"""
		Args:
		"""
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.video_clip_feature_dir = video_clip_feature_dir
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		
	def __len__(self):
		return len(self.videoID2obj_feature)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))

	def _get_video(self, feat_2d_path,feat_3d_path):
		feat_2d = np.load(feat_2d_path)
		feat_3d = np.load(feat_3d_path)
		
		feat_2d = th.from_numpy(feat_2d).float()
		feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)

		feat_3d = th.from_numpy(feat_3d).float()
		feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)

		return th.cat((feat_2d, feat_3d))

	def _get_obj_features(self,features_path):
		# TODO: sample features from across different frames
		obj_features_all = np.load(features_path)

		if obj_features_all.shape[0] == 1:
			# only one frame sampled, so just return that 
			output = th.from_numpy(obj_features_all[0][:self.num_candidates]).float()
		else:
			# multiple frames were sampled, so we should just pick top 1 one from each possible frame 
			output = []
			i = 0 
			while len(output) < self.num_candidates:
				for j in range(obj_features_all.shape[0]):
					output.append(obj_features_all[j][i])
				i+=1 
			output = np.stack(output) 
			output = th.from_numpy(output[:self.num_candidates]).float()
		return output

	def __getitem__(self, idx):
		video_id_raw = list(self.videoID2obj_feature.keys())[idx] 
		vid_name = video_id_raw + '.mp4'
#		print(vid_name)
		sentences = self.videoID2obj_feature[video_id_raw]['article_sentences']
		texts = [] 
		is_event_coreferences = [] 
		for text,is_event_coreference in sentences:
			texts.append(self._get_text(text))
			if is_event_coreference == 'EVENT_COREFERENCE':
				is_event_coreferences.append(1)
			else:
				is_event_coreferences.append(0)
#		print(sentences)
#		print(len(texts))
		# below will change to start/end format 	
		obj_names = self.videoID2obj_feature[video_id_raw]['detection_class_names']

		obj_features = self._get_obj_features(self.videoID2obj_feature[video_id_raw]['features_path'])
		feat_2d_path = os.path.join(self.video_clip_feature_dir,'2d',video_id_raw+'.npy')
		feat_3d_path = os.path.join(self.video_clip_feature_dir,'3d',video_id_raw+'.npy')

		video = self._get_video(feat_2d_path,feat_3d_path)
		return {'video': video, 'text': texts, 'obj':obj_features, 'video_id': video_id_raw, 'is_event_coreferences':is_event_coreferences,'raw_texts':sentences,'obj_names':obj_names}

class M2E2MILOVidSegDiscriminatorDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			videoID2obj_feature_path, # /kiwi-data/users/shoya/AIDA/obj_det_features/m2e2_obj_detections_merged.json.json
			we,
			vid_feat_dir,
			we_dim=300,
			max_words=30,
			num_candidates=5,
			n_pair=36
	):
		"""
		Args:
		"""
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		self.n_pair=n_pair
		self.obj_embed_dim = 2048
		self.vid_embed_input_dim = 4096
		self.vid_feat_dir = vid_feat_dir
		
	def __len__(self):
		return len(self.videoID2obj_feature)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text_and_obj(self, vid_entry):
		starts = np.zeros(self.n_pair)
		ends = np.zeros(self.n_pair)
		texts = th.zeros(self.n_pair,self.max_words,self.we_dim)
		objs = th.zeros(self.n_pair,self.num_candidates,self.obj_embed_dim)

		r_ind = np.random.choice(range(len(vid_entry.keys())),self.n_pair,replace=False)

		for i in range(self.n_pair):
			mid_frame_sec = list(vid_entry.keys())[r_ind[i]]
			starts[i] = vid_entry[mid_frame_sec]['start_sec']
			ends[i] = vid_entry[mid_frame_sec]['end_sec']
			texts[i] = self._words_to_we(self._tokenize_text(vid_entry[mid_frame_sec]['text']))
			obj_features = np.load(vid_entry[mid_frame_sec]['features_path'])
			objs[i] = th.from_numpy(obj_features[:self.num_candidates]).float()

		return texts,starts,ends,objs

	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced

	def _get_video(self,vid_name,starts,ends):
		videos = th.zeros(self.n_pair,self.vid_embed_input_dim)

		feat_2d_path = os.path.join(self.vid_feat_dir,'2d',vid_name+'.npy')
		feat_3d_path = os.path.join(self.vid_feat_dir,'3d',vid_name+'.npy')

		for i, start in enumerate(starts):
			feat_2d = self._get_slice(feat_2d_path,start,ends[i],'2d')
			feat_3d = self._get_slice(feat_3d_path,start,ends[i],'3d')
			feat_2d = th.from_numpy(feat_2d).float()
			feat_3d = th.from_numpy(feat_3d).float()

			if len(feat_2d) < 1 or len(feat_3d) < 1: 
				print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
			else:
				feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
				feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)
			videos[i] = th.cat((feat_2d, feat_3d))
		return videos

	def _get_obj_features(self,features_path):
		obj_features = np.load(features_path)
		return th.from_numpy(obj_features[:self.num_candidates]).float()

	def __getitem__(self, idx):
		vid_name = list(self.videoID2obj_feature.keys())[idx] # 
		texts, starts, ends, obj_features = self._get_text_and_obj(self.videoID2obj_feature[vid_name])
		videos = self._get_video(vid_name,starts,ends)

		return {'video': videos, 'text': texts, 'obj':obj_features, 'video_id': vid_name}

class M2E2MILOVidSegDiscriminatorWithMultipleSamplingDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			videoID2obj_feature_path, # /kiwi-data/users/shoya/AIDA/obj_det_features/m2e2_obj_detections_merged.json.json
			we,
			vid_feat_dir,
			we_dim=300,
			max_words=30,
			num_candidates=5,
			n_pair=36,
			look_window=1
			
	):
		"""
		Args:
		"""
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		self.n_pair=n_pair
		self.obj_embed_dim = 2048
		self.vid_embed_input_dim = 4096
		self.vid_feat_dir = vid_feat_dir
		self.look_window = look_window
		
	def __len__(self):
		return len(self.videoID2obj_feature)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text_and_obj(self, vid_entry):
		starts = np.zeros(self.n_pair)
		ends = np.zeros(self.n_pair)
		texts = th.zeros(self.n_pair,self.max_words,self.we_dim)
		objs = th.zeros(self.n_pair,self.num_candidates,self.obj_embed_dim)

		num_ASRs = len(vid_entry.keys())
		r_ind = np.random.choice(range(num_ASRs),self.n_pair,replace=False)

		for i in range(self.n_pair):
			mid_frame_sec = list(vid_entry.keys())[r_ind[i]]
			# print("Picked middle frame for vid: {} at index {}".format(mid_frame_sec,r_ind[i]))

			starts[i] = vid_entry[mid_frame_sec]['start_sec']
			ends[i] = vid_entry[mid_frame_sec]['end_sec']

			
			start = max(0,r_ind[i]-self.look_window)
			end = min(num_ASRs, r_ind[i]+self.look_window+1)

			# print("Window from index {} to {}".format(start,end))

			# randomly choose text from nearby ASR 
			index_choices = [n+start for n in list(range(end-start))]
			# print("Index choices: {}".format(index_choices))
			text_ind = random.choice(index_choices)
			text_mid_frame_sec = list(vid_entry.keys())[text_ind]
			# print("For text, we chose index {} at sec {}".format(text_ind,text_mid_frame_sec))
			texts[i] = self._words_to_we(self._tokenize_text(vid_entry[text_mid_frame_sec]['text']))

			# get all the objects from nearby ASR pairs, and randomly sample ones above 0.6 confidence
			nearby_objects = [] 
			obj_detec_confidences = [] 
			for ind in index_choices:
				mid_fr = list(vid_entry.keys())[ind]
				obj = np.load(vid_entry[mid_fr]['features_path'])
				nearby_objects.append(obj)
				obj_detec_confidences.append(vid_entry[mid_fr]['obj_detec_confidence'])

			nearby_objects = np.stack(nearby_objects)
			obj_detec_confidences = np.stack(obj_detec_confidences) 
			above_threshold = obj_detec_confidences >= 0.6
			conf_nearby_objects = nearby_objects[above_threshold] 

			# print("conf_nearby_objects: {}".format(conf_nearby_objects.shape))

			if conf_nearby_objects.shape[0] >= self.num_candidates:
				chosen_objs_indices = np.random.choice(conf_nearby_objects.shape[0],self.num_candidates,replace=False)
				# print("Chosen object Indices: {}".format(chosen_objs_indices))
				# print("Chosen objects: {}".format(conf_nearby_objects[chosen_objs_indices].shape))
				objs[i] = th.from_numpy(conf_nearby_objects[chosen_objs_indices]).float()
			else:
				# print("Not enough confident object detections")
				obj_features = np.load(vid_entry[mid_frame_sec]['features_path'])
				objs[i] = th.from_numpy(obj_features[:self.num_candidates]).float()

		return texts,starts,ends,objs

	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced

	def _get_video(self,vid_name,starts,ends):
		videos = th.zeros(self.n_pair,self.vid_embed_input_dim)

		feat_2d_path = os.path.join(self.vid_feat_dir,'2d',vid_name+'.npy')
		feat_3d_path = os.path.join(self.vid_feat_dir,'3d',vid_name+'.npy')

		for i, start in enumerate(starts):
			feat_2d = self._get_slice(feat_2d_path,start,ends[i],'2d')
			feat_3d = self._get_slice(feat_3d_path,start,ends[i],'3d')
			feat_2d = th.from_numpy(feat_2d).float()
			feat_3d = th.from_numpy(feat_3d).float()

			if len(feat_2d) < 1 or len(feat_3d) < 1: 
				print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
			else:
				feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
				feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)
			videos[i] = th.cat((feat_2d, feat_3d))
		return videos

	def _get_obj_features(self,features_path):
		obj_features = np.load(features_path)
		return th.from_numpy(obj_features[:self.num_candidates]).float()

	def __getitem__(self, idx):
		vid_name = list(self.videoID2obj_feature.keys())[idx] # 
		texts, starts, ends, obj_features = self._get_text_and_obj(self.videoID2obj_feature[vid_name])
		videos = self._get_video(vid_name,starts,ends)

		return {'video': videos, 'text': texts, 'obj':obj_features, 'video_id': vid_name}

class M2E2MILOManualLabelsWithArticleAndProposalsDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			videoID2obj_feature_path, 
			video_clip_feature_dir,
			we,
			we_dim=300,
			max_words=30,
			num_candidates=5
	):
		"""
		Args:
		"""
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.video_clip_feature_dir = video_clip_feature_dir
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		self.obj_embed_dim = 2048
		self.vid_embed_input_dim = 4096
		self.fps = {'2d': 1, '3d': 1.5,'obj':1/3}
		
	def __len__(self):
		return len(self.videoID2obj_feature)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))


	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced

	def _get_video(self,feat_2d_path,feat_3d_path,proposals):
		videos = th.zeros(len(proposals),self.vid_embed_input_dim)

		for i, proposal in enumerate(proposals):
			start = proposal[0]
			end = proposal[1]
			feat_2d = self._get_slice(feat_2d_path,start,end,'2d')
			feat_3d = self._get_slice(feat_3d_path,start,end,'3d')
			feat_2d = th.from_numpy(feat_2d).float()
			feat_3d = th.from_numpy(feat_3d).float()

			if len(feat_2d) < 1 or len(feat_3d) < 1: 
				print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
			else:
				feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
				feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)
			videos[i] = th.cat((feat_2d, feat_3d))
		return videos


	def _get_obj_features_from_across_sections(self,obj_features_all, obj_names):
		used_obj_names = [] 
		if obj_features_all.shape[0] == 1:
			# only one frame sampled, so just return that 
			output = th.from_numpy(obj_features_all[0][:self.num_candidates]).float()
			used_obj_names = obj_names[0][:self.num_candidates]
		elif obj_features_all.shape[0] == 0:
			return th.zeros(size=(self.num_candidates,self.obj_embed_dim)), ['None'] * self.num_candidates 
		else:
			# multiple frames were sampled, so we should just pick top 1 one from each possible frame 
			output = []
			i = 0 
			while len(output) < self.num_candidates:
				for j in range(obj_features_all.shape[0]):
					output.append(obj_features_all[j][i])
					used_obj_names.append(obj_names[j][i])
				i+=1 
			output = np.stack(output) 
			output = th.from_numpy(output[:self.num_candidates]).float()
		return output, used_obj_names[:self.num_candidates]


	def _get_obj_features(self,features_path,proposals, obj_names):
		obj_features_all = np.load(features_path)
		objs = th.zeros(len(proposals),self.num_candidates,self.obj_embed_dim)
		obj_names_all = [] 

		for i, prop in enumerate(proposals):
			start =  round(prop[0]*self.fps['obj'])
			end = round(prop[1]*self.fps['obj'])+1

			obj_sliced = obj_features_all[start:end]
			obj_names_sliced = obj_names[start:end]
			used_objs, used_obj_names =self._get_obj_features_from_across_sections(obj_sliced,obj_names_sliced)
			objs[i] = used_objs
			obj_names_all.append(used_obj_names)

		return objs, obj_names_all


	def _get_proposals(self,proposals_raw,vid_dur):
		proposals = [[0,proposals_raw[0]]]
		for i in range(len(proposals_raw)):
			if i < len(proposals_raw)-1:
				proposals.append([proposals_raw[i],proposals_raw[i+1]])
			else:
				proposals.append([proposals_raw[i],vid_dur]) 
		return proposals 


	def __getitem__(self, idx):
		video_id_raw = list(self.videoID2obj_feature.keys())[idx] 
		vid_name = video_id_raw + '.mp4'

		texts = [] 
		proposals = self._get_proposals(self.videoID2obj_feature[video_id_raw]['temporal_proposals'],self.videoID2obj_feature[video_id_raw]['vid_duration'])

		for sent in self.videoID2obj_feature[video_id_raw]['article_sentences']:
			texts.append(self._get_text(sent))

		if np.load(self.videoID2obj_feature[video_id_raw]['features_path']).shape[0] == 0:
			print("No objects for {}".format(vid_id_raw))

		obj_features, used_obj_names = self._get_obj_features(self.videoID2obj_feature[video_id_raw]['features_path'],proposals,self.videoID2obj_feature[video_id_raw]['detection_class_names'])

		feat_2d_path = os.path.join(self.video_clip_feature_dir,'2d',video_id_raw+'.npy')
		feat_3d_path = os.path.join(self.video_clip_feature_dir,'3d',video_id_raw+'.npy')
		video_features = self._get_video(feat_2d_path,feat_3d_path,proposals)
		return {'video': video_features, 'text': texts, 'obj':obj_features, 'video_id': video_id_raw, 'event_coreference_pairs':self.videoID2obj_feature[video_id_raw]['event_coreference_text'],'raw_texts':self.videoID2obj_feature[video_id_raw]['article_sentences'], 'proposals':proposals, 'obj_names':used_obj_names}


class M2E2EventTypesZeroShotDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			m2e2_event_types_csv,
			videoID2obj_feature_path,
		    m2e2_event_types,
		    video_clip_feature_dir,
		    we,
		    we_dim=300,
		    max_words=30,
		    num_candidates=5
	):
		"""
		Args:
		"""
		self.csv = pd.read_csv(m2e2_event_types_csv)
		self.videoID2obj_feature = json.load(open(videoID2obj_feature_path))
		self.video_feature_dir = video_clip_feature_dir
		self.event_types = m2e2_event_types
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		self.num_candidates = num_candidates
		self.obj_embed_dim = 2048
		self.vid_embed_input_dim = 4096
		self.fps = {'2d': 1, '3d': 1.5,'obj':1/3}
		
	def __len__(self):
		return self.csv.shape[0]

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))

	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced

	def _get_video(self,feat_2d_path,feat_3d_path,start,end):
		feat_2d = self._get_slice(feat_2d_path,start,end,'2d')
		feat_3d = self._get_slice(feat_3d_path,start,end,'3d')
		feat_2d = th.from_numpy(feat_2d).float()
		feat_3d = th.from_numpy(feat_3d).float()

		if len(feat_2d) < 1 or len(feat_3d) < 1: 
			print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
		else:
			feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
			feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)
		videos = th.cat((feat_2d, feat_3d))
		return videos

	def _get_obj_features_from_across_sections(self,obj_features_all, obj_names):
		used_obj_names = [] 
		if obj_features_all.shape[0] == 1:
			# only one frame sampled, so just return that 
			output = th.from_numpy(obj_features_all[0][:self.num_candidates]).float()
			used_obj_names = obj_names[0][:self.num_candidates]
		elif obj_features_all.shape[0] == 0:
			return th.zeros(size=(self.num_candidates,self.obj_embed_dim)), ['None'] * self.num_candidates 
		else:
			# multiple frames were sampled, so we should just pick top 1 one from each possible frame 
			output = []
			i = 0 
			while len(output) < self.num_candidates:
				for j in range(obj_features_all.shape[0]):
					output.append(obj_features_all[j][i])
					used_obj_names.append(obj_names[j][i])
				i+=1 
			output = np.stack(output) 
			output = th.from_numpy(output[:self.num_candidates]).float()
		return output, used_obj_names[:self.num_candidates]

	def _get_obj_features(self,features_path,start,end, obj_names):
		obj_features_all = np.load(features_path)
		
		start =  round(start*self.fps['obj'])
		end = round(end*self.fps['obj'])+1

		obj_sliced = obj_features_all[start:end]
		obj_names_sliced = obj_names[start:end]
		used_objs, used_obj_names =self._get_obj_features_from_across_sections(obj_sliced,obj_names_sliced)
	
		return used_objs, used_obj_names

	def _get_proposals(self,proposals_raw,vid_dur):
		proposals = [[0,proposals_raw[0]]]
		for i in range(len(proposals_raw)):
			if i < len(proposals_raw)-1:
				proposals.append([proposals_raw[i],proposals_raw[i+1]])
			else:
				proposals.append([proposals_raw[i],vid_dur]) 
		return proposals 

	def _get_class_texts(self):
		class_texts = th.zeros(len(self.event_types),self.max_words,self.we_dim)
		for idx, text in self.event_types.items():
			class_texts[idx] = self._get_text(text)
		return class_texts

	def __getitem__(self, idx):
		vid_name = self.csv.iloc[idx]['vid_name']
		event_type = self.csv.iloc[idx]['event_type']
		texts = self._get_class_texts()
	
		if np.load(self.videoID2obj_feature[vid_name[:-4]]['features_path']).shape[0] == 0:
			print("No objects for {}".format(vid_name))

		start = self.csv.iloc[idx]['boundary_start']
		end = self.csv.iloc[idx]['boundary_end']

		obj_features, used_obj_names = self._get_obj_features(self.videoID2obj_feature[vid_name[:-4]]['features_path'],start,end,self.videoID2obj_feature[vid_name[:-4]]['detection_class_names'])

		feat_2d_path = os.path.join(self.video_feature_dir,'2d',vid_name[:-4]+'.npy')
		feat_3d_path = os.path.join(self.video_feature_dir,'3d',vid_name[:-4]+'.npy')

		video_features = self._get_video(feat_2d_path,feat_3d_path,start,end)

		return {'event_type':event_type,'video': video_features, 'text': texts, 'obj':obj_features, 'video_id': vid_name[:-4],'obj_names':used_obj_names,'boundary':(start,end)}
