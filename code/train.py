import argparse
import collections
from parse_config import ConfigParser
import torch
from dataset.RealLowlightDataset import *
import loss.loss as module_loss
import loss.metric as module_metric
from model import *
from trainer import Trainer
from utils import prepare_device
from torch.utils.data import DataLoader, ConcatDataset
import loss.lpips.lpips as lpips__
import numpy as np
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def concatenate_subfolders(config, is_val=False):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """
    if config['dataset']['args']['is_cat']:
        if not is_val:
            base_folder = config['dataset']['args']['train_data_dir']
        else:
            base_folder = config['dataset']['args']['val_data_dir']

        subfolders = os.listdir(base_folder)
        print('Found {} samples in {}'.format(len(subfolders), base_folder))

        datasets = []
        for dataset_name in subfolders:
            sub_path = os.path.join(base_folder, dataset_name)
            datasets.append(eval(config['dataset']['type'])(sub_path, config['dataset']['args']))
        final_dataset = ConcatDataset(datasets)
    else:
        if not is_val:
            base_folder = config['dataset']['args']['train_data_dir']
        else:
            base_folder = config['dataset']['args']['val_data_dir']
        final_dataset = eval(config['dataset']['type'])(base_folder, config['dataset']['args'])

    return final_dataset


def main(config):
    logger = config.get_logger('train')

    # load dataset
    train_dataset = concatenate_subfolders(config, is_val=False)
    val_dataset = concatenate_subfolders(config, is_val=True)

    # load dataset
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config['data_loader']['batch_size'],
                                   shuffle=config['data_loader']['shuffle'],
                                   num_workers=config['data_loader']['num_workers'])
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=config['data_loader']['batch_size'],
                                 shuffle=config['data_loader']['shuffle_val'],
                                 num_workers=config['data_loader']['num_workers'])

    # build model architecture, then print to console
    # print(config['arch']['type'])
    args = config['arch']['args']
    if len(args) == 2:
        input_c = args['input_c']
        output_c = args['output_c']
    else:
        input_c_frame = args['input_c_frame']
        input_c_event = args['input_c_event']
        output_c = args['output_c']
    model = eval(config['arch']['type'])(input_c_frame, input_c_event, output_c)
    

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # loss_fn = lpips__.LPIPS_rewrite()
    loss_fn = lpips__.LPIPS(net='vgg')
    loss_fn = loss_fn.to(device)
    criterion = getattr(module_loss, config['loss'])  # return a function method
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # Filter parameters that do not need to be trained
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())  # choose useful param
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      train_data_loader=train_data_loader,
                      val_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler,
                      loss_fn=loss_fn)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='code')
    args.add_argument('-c', '--config', default='./config/RealLowlightDataset.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-i', '--initial_checkpoint', default=None, type=str,
                      help='path to the checkpoint with which to initialize the model weights (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # args = args.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataset;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

# nohup python train.py >./run_Log/906_1_Ours.log 2>&1 &
