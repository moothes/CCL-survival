import os
import shutil
import torch
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
from sklearn.cluster import KMeans

import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

torch.set_num_threads(4)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        '''
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        '''
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler, subset):
        writer_dir = os.path.join(self.results_dir, subset + '_fold_' + str(self.fold))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        self.writer = SummaryWriter(writer_dir, flush_secs=15)
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.run_epoch(val_loader, model, criterion, phase='eval')
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.run_epoch(train_loader, model, criterion, phase='train', optimizer=optimizer)
            # evaluate on validation set
            c_index = self.run_epoch(val_loader, model, criterion, phase='eval')
            # remember best c-index and save checkpoint
            is_best = c_index >= self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score,
                    'subset': subset,})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            print('>')
        return self.best_score, self.best_epoch

    def run_epoch(self, data_loader, model, criterion, phase='train', optimizer=None):
        eval('model.{}()'.format(phase))
        sum_loss = 0.0
        
        all_loss_dict = {}
        for k in criterion.loss_collection.keys():
            all_loss_dict[k] = 0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='{} Epoch: {}'.format(phase, self.epoch), ncols=150)
        
        for batch_idx, (data_WSI, data_omic, label, event_time, c, idx) in enumerate(dataloader):
            data_WSI = data_WSI.float().cuda()
            data_omic = data_omic.float().cuda()
            label = label.float().cuda()
            event_time = event_time.float().cuda()
            c = c.float().cuda()
            
            if phase == 'train':
                out = model(x_path=data_WSI, x_omic=data_omic, phase=phase, label=label, c=c)
            else:
                with torch.no_grad():
                    out = model(x_path=data_WSI, x_omic=data_omic, phase=phase)
                    
            loss, loss_dict = criterion(out, {'label': label, 'event_time': event_time, 'c': c})
            risk = -torch.sum(out['S'][-1], dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            
            lr_str = 0
            sum_loss += loss.item()
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_str = optimizer.param_groups[-1]['lr']
            
            dataloader.set_postfix_str('LR: {:.1e}, loss: {:4f}'.format(lr_str, sum_loss/(batch_idx+1)))
        
        # calculate loss and error for epoch
        sum_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(sum_loss, c_index))

        if self.writer:
            self.writer.add_scalar('{}/loss'.format(phase), sum_loss, self.epoch)
            self.writer.add_scalar('{}/c_index'.format(phase), c_index, self.epoch)
        return c_index

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          state['subset'] + '_fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
