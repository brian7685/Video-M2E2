# This script finetunes with ASR + object detection features and evaluates on the M2E2 Co-occurence data

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from args import get_args
import random
import os
from milo_model import Net
from metrics import compute_metrics, print_computed_metrics
from loss import MaxMarginRankingLoss
from gensim.models.keyedvectors import KeyedVectors
import pickle
from m2e2_dataloader import M2E2DataLoader,M2E2ASRDataLoader,M2E2MILOVidSegDiscriminatorDataLoader,M2E2MILOManualLabelsWithArticleDataLoader, M2E2MILOVidSegDiscriminatorWithMultipleSamplingDataLoader
from mil_loss_b import MILOLoss
from scipy.special import softmax
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from collections import defaultdict

args = get_args()
if args.verbose:
	print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
	os.mkdir(args.checkpoint_dir)


print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')


if args.m2e2:
	print("Initializing Dataset")

	dataset = M2E2MILOVidSegDiscriminatorWithMultipleSamplingDataLoader(
		videoID2obj_feature_path=args.videoID2obj_feature_path, # /kiwi-data/users/shoya/AIDA/obj_det_features/ten_feats_per_frame/merged_obj_features_10_per_frame.json
		we=we,
		max_words=args.max_words,
		we_dim=args.we_dim,
		num_candidates=args.num_candidates,
		n_pair=args.n_pair,
		vid_feat_dir=args.train_np_vid_feat_dir,
		look_window = args.look_window
	)


dataset_size = len(dataset)
print("Initializing dataloader")
dataloader = DataLoader(
	dataset,
	batch_size=args.batch_size,
	num_workers=args.num_thread_reader,
	shuffle=True,
	batch_sampler=None,
	drop_last=True,
	pin_memory=True
)

if args.eval_m2e2:
	dataset_val_m2e2 = M2E2MILOManualLabelsWithArticleDataLoader(
		videoID2obj_feature_path=args.m2e2_manual_labels_obj_features_json_val, # event_occurences_video_and_all_sentences_original_vids.json
		video_clip_feature_dir=args.video_clip_feature_dir, 
		we=we,
		we_dim=args.we_dim,
		max_words=args.max_words,
		num_candidates=args.num_candidates_eval
	)

	dataloader_val_m2e2 = DataLoader(
		dataset_val_m2e2,
		batch_size=1, # args.batch_size_val
		num_workers=args.num_thread_reader,
		shuffle=False,
		pin_memory=True
	)

	dataset_test_m2e2 = M2E2MILOManualLabelsWithArticleDataLoader(
		videoID2obj_feature_path=args.m2e2_manual_labels_obj_features_json_test, # event_occurences_video_and_all_sentences_original_vids.json
		video_clip_feature_dir=args.video_clip_feature_dir, 
		we=we,
		we_dim=args.we_dim,
		max_words=args.max_words,
		num_candidates=args.num_candidates_eval
	)

	dataloader_test_m2e2 = DataLoader(
		dataset_test_m2e2,
		batch_size=1, # args.batch_size_val
		num_workers=args.num_thread_reader,
		shuffle=False,
		pin_memory=True
	)
    
print("Initializing Model")
net = Net(
	video_dim=args.feature_dim,
	embd_dim=args.embd_dim,
	we_dim=args.we_dim,
	max_words=args.max_words,
	sentence_dim=args.sentence_dim,
	obj_feat_dim=args.obj_feat_dim
)
net.train()
# Optimizers + Loss
loss_op = MILOLoss(args)

net.cuda()
loss_op.cuda()

if args.pretrain_path != '':
	net.load_checkpoint(args.pretrain_path)
	print("pretrained weights loaded")

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.verbose:
	print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun):
	text = data['text'].cuda()
	video = data['video'].cuda()
	obj = data['obj'].cuda()
	video = video.view(-1, video.shape[-1])
	text = text.view(-1, text.shape[-2], text.shape[-1])
	obj = obj.view(-1,obj.shape[-2],obj.shape[-1])

	opt.zero_grad()
	with th.set_grad_enabled(True):
		video_embed, text_embed, obj_embed = model(video, text,obj)
		loss = loss_fun(video_embed,text_embed,obj_embed)
	loss.backward()
	opt.step()
	return loss.item()

# video is along the column
def get_sim_matrices(video_embed,text_embed,obj_embed):
	vid_feat_sim_matrix = th.matmul(video_embed, text_embed.t())
	vid_feat_sim_matrix  = vid_feat_sim_matrix.cpu().detach().numpy()
	vid_feat_sim_matrix_softmaxed = softmax(vid_feat_sim_matrix,axis=0)

	obj_embed = obj_embed.view(obj_embed.shape[0]*obj_embed.shape[1],-1)
	obj_feat_sim_matrix = th.matmul(video_embed, obj_embed.t())
	obj_feat_sim_matrix = obj_feat_sim_matrix.view(video_embed.shape[0], video_embed.shape[0], -1).cpu().detach().numpy()
	obj_feat_sim_matrix_averaged = np.mean(obj_feat_sim_matrix,axis=2)
	obj_feat_sim_matrix_averaged_softmaxed = softmax(obj_feat_sim_matrix_averaged,axis=0)

	avg_sim_matrix = (vid_feat_sim_matrix_softmaxed + obj_feat_sim_matrix_averaged_softmaxed)/2
	
	return avg_sim_matrix,vid_feat_sim_matrix,obj_feat_sim_matrix_averaged 

# sim matrices with text along the column 
def get_sim_matrices2(video_embed,text_embed,obj_embed):
	vid_feat_sim_matrix = softmax(th.matmul(text_embed,video_embed.t()).cpu().detach().numpy(),axis=0)

	obj_embed = obj_embed.view(obj_embed.shape[0]*obj_embed.shape[1],-1)
	obj_feat_sim_matrix = th.matmul(text_embed, obj_embed.t()).cpu().detach().numpy()
	obj_feat_sim_matrix_averaged = np.mean(obj_feat_sim_matrix,axis=2)
#	print('obj_feat_sim_matrix_averaged',obj_feat_sim_matrix_averaged.shape)
	obj_feat_sim_matrix_averaged_softmaxed = softmax(obj_feat_sim_matrix_averaged,axis=0)

	vid_feat_sim_matrix = vid_feat_sim_matrix.reshape((vid_feat_sim_matrix.shape[0],vid_feat_sim_matrix.shape[1]))
#	print('in get_sim_matrices2',vid_feat_sim_matrix.shape,obj_feat_sim_matrix_averaged_softmaxed.shape)
#	print(vid_feat_sim_matrix,obj_feat_sim_matrix_averaged_softmaxed)
	a=0.3
	avg_sim_matrix = a*vid_feat_sim_matrix + (1-a)*obj_feat_sim_matrix_averaged_softmaxed
	#(vid_feat_sim_matrix + 2*obj_feat_sim_matrix_averaged_softmaxed)/2
	
	return avg_sim_matrix,vid_feat_sim_matrix,obj_feat_sim_matrix_averaged_softmaxed 

def Eval_retrieval(model, eval_dataloader, dataset_name, epoch, best_f1,testset=False,best_threshold=None):
	model.eval()
	print('Evaluating Event Coreference Matching on {} data'.format(dataset_name))

	actual = [] 

	if testset:
		preds_by_threshold = {
            str(best_threshold):{'avg':[],'global':[],'regional':[]}
        }
	else:
		preds_by_threshold = {
			'0.08':{'avg':[],'global':[],'regional':[]},
			'0.09':{'avg':[],'global':[],'regional':[]},
			'0.10':{'avg':[],'global':[],'regional':[]},
			'0.11':{'avg':[],'global':[],'regional':[]},
			'0.12':{'avg':[],'global':[],'regional':[]},
			'0.13':{'avg':[],'global':[],'regional':[]},
			'0.14':{'avg':[],'global':[],'regional':[]},
			'0.15':{'avg':[],'global':[],'regional':[]},
			'0.16':{'avg':[],'global':[],'regional':[]}
		}

	with th.no_grad():
		# batch size is 1 
		for i_batch, data in enumerate(eval_dataloader):
			texts = data['text']
			video = data['video'].cuda()
			obj = data['obj'].cuda()
			is_event_coreferences = data['is_event_coreferences']

#			print('texts.shape',len(texts),video.shape,obj.shape)

			video_embed = model.GU_video(video)
			sentences = [] 
			for text in texts:
				sentences.append(model.GU_text(model.text_pooling(text.cuda())))
			obj_embed = model.GU_obj(obj)
			#video_embed, text_embed, obj_embed = model(video, texts, obj)

			text_embed = th.stack(sentences)
#			print('video_embed.shape',video_embed.shape,text_embed.shape,obj_embed.shape)

			avg_sim_matrix,vid_feat_sim_matrix,obj_feat_sim_matrix = get_sim_matrices2(video_embed,text_embed,obj_embed)
#			print('avg_sim_matricx.shape',avg_sim_matrix.shape,vid_feat_sim_matrix.shape,obj_feat_sim_matrix.shape)

#			print('avg_sim_matrix',avg_sim_matrix)

			for threshold in preds_by_threshold.keys():
				for sim_type, sim_matrix in [('avg',avg_sim_matrix),('global',vid_feat_sim_matrix),('regional',obj_feat_sim_matrix)]:
					for sim in sim_matrix:
						if sim>float(threshold):
							preds_by_threshold[threshold][sim_type].append(1)
						else:
							preds_by_threshold[threshold][sim_type].append(0)
			actual.extend(is_event_coreferences)

		best_epoch_f1 = float('-inf')
		best_threshold = float(list(preds_by_threshold.keys())[0])
		best_t = 0
		for threshold in preds_by_threshold.keys():
			for sim_type in preds_by_threshold[threshold].keys():
				accuracy = accuracy_score(actual,preds_by_threshold[threshold][sim_type])
				prec,recall,fscore,support = precision_recall_fscore_support(actual,preds_by_threshold[threshold][sim_type])
                
				print("Threshold: {}, Sim Type: {}, Acc: {}, Pre: {}, Recall: {}, F1: {}, Sup: {}".format(threshold,sim_type,accuracy,prec[1],recall[1],fscore[1],support))
				#if sim_type == 'avg':
				if fscore[1]>best_epoch_f1:
					best_epoch_f1 = fscore[1]#max(best_epoch_f1,fscore[1])
					best_t = threshold
				#random_baseline = np.random.randint(low=0,high=2,size=(len(actual)))
				#accuracy = accuracy_score(actual,random_baseline)
				#prec,recall,fscore,support = precision_recall_fscore_support(actual,random_baseline)
				#print("Random Baseline - Accuracy: {}, Precision: {}, Recall: {}, F-score: {}".format(accuracy,prec[1],recall[1],fscore[1]))

		if best_epoch_f1 > best_f1 and not testset:
			print("======= New Best F1 Score: {} | Saving Checkpoint =======".format(best_epoch_f1))
			best_f1 = best_epoch_f1
			best_threshold = best_t

			if args.checkpoint_dir != '':
				path = os.path.join(args.checkpoint_dir, 'milo_{}.pth'.format(epoch + 1))
				net.save_checkpoint(path)
                
                
	return best_f1,best_threshold

best_f1 = float('-inf')
best_epoch = 0
best_threshold = None
for epoch in range(args.epochs):
	running_loss = 0.0
	if args.eval_m2e2:
		epoch_f1,thres = Eval_retrieval(net, dataloader_val_m2e2, 'M2E2', epoch,best_f1)
		if best_f1 < epoch_f1:
			best_f1 = epoch_f1
			best_epoch = epoch+1
			best_threshold = thres
		print("=== Evaluation on Test Set from model epoch {} with threshold {}. ".format(best_epoch,
																								   best_threshold))
		Eval_retrieval(net, dataloader_test_m2e2, 'M2E2', -1, best_f1, testset=True, best_threshold=best_threshold)
	if args.verbose:
		print('Epoch: %d' % epoch)
	for i_batch, sample_batch in enumerate(dataloader):
		batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss_op)
		running_loss += batch_loss
		if (i_batch + 1) % args.n_display == 0 and args.verbose:
			print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
			(epoch + 1, args.batch_size * float(i_batch) / dataset_size,
			running_loss / args.n_display))
			running_loss = 0.0
	for param_group in optimizer.param_groups:
		param_group['lr'] *= args.lr_decay
	
if args.eval_m2e2:
	best_model_path = os.path.join(args.checkpoint_dir, 'milo_{}.pth'.format(best_epoch))   
	print("=== Evaluation on Test Set from model epoch {} with threshold {}. Model: {}".format(best_epoch,best_threshold,best_model_path))
	net.load_checkpoint(best_model_path)
	best_f1,_ = Eval_retrieval(net, dataloader_test_m2e2, 'M2E2', -1, best_f1,testset=True,best_threshold=best_threshold)

	print("=== Below output is to get HowTo100M unfinetuned baseline score on the test set: ===")   
	net = Net(
        video_dim=args.feature_dim,
        embd_dim=args.embd_dim,
        we_dim=args.we_dim,
        max_words=args.max_words,
        sentence_dim=args.sentence_dim,
        obj_feat_dim=args.obj_feat_dim
    )
	net.cuda()    
	net.load_checkpoint(args.pretrain_path)
	_,_ = Eval_retrieval(net, dataloader_test_m2e2, 'M2E2', -1, best_f1,testset=True,best_threshold=best_threshold)
