# MILNCELoss from https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/loss.py


import torch as th
from loss import MaxMarginRankingLoss
import base64


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)

class MILOLoss(th.nn.Module):
    def __init__(self,args):
        super(MILOLoss,self).__init__()
        self.milnce_loss = MILNCELoss()
        self.mmrloss = MaxMarginRankingLoss(
            margin=args.margin,
            negative_weighting=args.negative_weighting,
            batch_size=args.batch_size,
            n_pair=args.n_pair,
            hard_negative_rate=args.hard_negative_rate,
        )

    def forward(self,video_embed,text_embed,obj_embed):
        # text_embed 36x6144
        # video_embed 36x6144
        # obj_embed 36x5x6144
        vid_text_sim = th.matmul(text_embed, video_embed.t()) 
        loss_mmr = self.mmrloss(vid_text_sim)
        obj_embed = obj_embed.view(obj_embed.shape[0]*obj_embed.shape[1],-1)
        loss_mil_nce = self.milnce_loss(text_embed,obj_embed)
        #v_emb = th.unsqueeze(video_embed, 1)
        #print('v_shape',v_emb.shape)
        #print('o_shape', obj_embed.shape)
        all_emb = th.cat((video_embed, obj_embed), 0)
        loss_nce = self.milnce_loss(video_embed,text_embed)
        loss_mil_nce_all = self.milnce_loss(text_embed, all_emb)
        #print("MMRLoss: {} | MILNCE Loss: {}".format(loss_mmr,loss_mil_nce))
        return loss_mil_nce + loss_nce#+ #loss_mil_nce#loss_mmr #+ loss_mil_nce


