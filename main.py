import os
import sys
import csv
import time
import random
import torch
import numpy as np
import importlib

from shutil import copyfile

from torch.utils.data import DataLoader
from utils.options import parse_args
from utils.loss_factory import Loss_factory
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.engine import Engine
from utils.dataset_survival import Generic_MIL_Survival_Dataset

def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args):
    print(args)
    # create results directory
    results_dir = "./results/{model}_{extractor}_{loss}_{seed}_{lr}_{time}".format(
        model=args.model,
        extractor=args.extractor,
        loss=args.loss,
        seed=args.seed,
        lr=args.lr,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_path = os.path.join(results_dir, "results_level_{}.csv".format(args.level))
    header = ["name", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    print("############", csv_path)
    with open(csv_path, "a+", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
    
    datalist = args.sets.split(',')
    all_best_score = {}
    folds = list(map(int, args.fold.split(',')))
    for dataname in datalist:
        args.dataset = dataname
        # 5-fold cross validation
        best_epoch = ["best epoch"]
        best_score = [dataname]

        # start 5-fold CV evaluation.
        for fold in folds:
            set_seed(args.seed)
            dataset = Generic_MIL_Survival_Dataset(
                csv_path="./csv/tcga_%s_all_clean.csv" % (dataname),
                modal=args.modal,
                apply_sig=True,
                data_dir=args.data_root_dir,
                shuffle=False,
                seed=args.seed,
                patient_strat=False,
                n_bins=4,
                label_col="survival_months",
            )
            split_dir = "./splits/{}/tcga_{}".format(args.which_splits, dataname)
            train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path="{}/splits_{}.csv".format(split_dir, fold), set_name=args.dataset, extractor=args.extractor)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
            print("Dataset {} split_{}: train {}, val {}".format(dataname, fold, len(train_dataset), len(val_dataset)))
            
            try:
                my_module = importlib.import_module('models.{}.network'.format(args.model))
                model = getattr(my_module, args.model)(args)
            except:
                raise NotImplementedError("An error occur when loading \'{}\' model.".format(args.model))
                
            engine = importlib.import_module('utils.engine').Engine(args, results_dir, fold)
            criterion = Loss_factory(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            
            # start training
            score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler, dataname)
            
            # save best score and epoch for each fold
            best_epoch.append(epoch)
            best_score.append(score)

        # finish training
        best_epoch.append("~")
        best_epoch.append("~")
        best_score.append(np.mean(best_score[1:6]))
        best_score.append(np.std(best_score[1:6]))
        all_best_score[dataname] = best_score[1:6]

        print("############", csv_path)
        with open(csv_path, "a+", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(best_epoch)
            writer.writerow(best_score)
            
    print(all_best_score)
    return all_best_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args(sys.argv[1])
    main(args)
    print("Finished!")
