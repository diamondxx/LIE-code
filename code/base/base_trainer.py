import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.event_rgb_separate_input = config['dataset']['args']['event_rgb_separate_input']
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # record current best result (val_ssim + val_psnr)
        self.mnt_best = 0.
        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _train_epoch_two_input(self, epoch):
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.event_rgb_separate_input:
                result = self._train_epoch_two_input(epoch)
            else:
                result = self._train_epoch(epoch)
            # torch.cuda.empty_cache()

            # save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            # check whether model performance improved or not, according to specified metric(mnt_metric)
            # log.keys() is dict_keys(['epoch', 'loss', 'ssim', 'psnr', 'val_loss', 'val_ssim', 'val_psnr'])
            if (log['val_ssim'] + log['val_psnr']) > self.mnt_best:
                self.mnt_best = log['val_ssim'] + log['val_psnr']
                not_improved_count = 0
                self._save_best_checkpoint(epoch, save_best=True)
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                break

            # model only with better result will be saved
            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            self.logger.info("Saving current best: model_best.pth ...")
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
