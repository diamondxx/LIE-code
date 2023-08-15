import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from PIL import Image
from loss.loss import mse_loss, l2_loss
from utils.util import *


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, val_data_loader=None, lr_scheduler=None, loss_fn=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.val_data_loader = val_data_loader
        self.len_epoch_val = len(self.val_data_loader)
        self.do_validation = self.val_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.img_channels = int(config['dataset']['args']['img_channels'])
        self.loss_fn = loss_fn

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.mode = 'red-blue'
        # self.mode = 'grayscale'

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target, self.loss_fn)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            batch_size, _, width, height = target.shape
            events_tmp = torch.ones((batch_size, 1, width, height))
            events_tmp = events_tmp.to(self.device)

            res_tensor = make_image(torch.cat((data[:, : self.img_channels, :, :], events_tmp), dim=1), output, target)
            self.writer.add_image('Training: events, image, output, gt.',
                                  make_grid(res_tensor.cpu(), nrow=8, normalize=False))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


    def _train_epoch_two_input(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (imgs, events, target, images, targets) in enumerate(self.train_data_loader):
            imgs, events, target = imgs.to(self.device), events.to(self.device), target.to(self.device)
            for i in range(len(images)):
                images[i], targets[i] = images[i].to(self.device), targets[i].to(self.device)
            self.optimizer.zero_grad()
            
            milti_output, output = self.model(imgs, events)
            loss = self.criterion(milti_output, targets, images, self.loss_fn)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), n=imgs.shape[0])

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), n=imgs.shape[0])

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))

            batch_size, _, width, height = target.shape
            if self.mode == 'red-blue':
                events_tmp = torch.zeros((batch_size, 3, width, height))
            else:
                events_tmp = torch.zeros((batch_size, 1, width, height))

            for i in range(batch_size):
                if self.mode == 'red-blue':
                    events_tmp[i] = torch.from_numpy(np.transpose(make_event_preview(events[i].unsqueeze(0)), (2, 0, 1)))
                else:
                    events_tmp[i] = torch.from_numpy(make_event_preview(events[i].unsqueeze(0)))
            events_tmp = events_tmp.to(self.device)

            # imgs B 3 H W ; events B 5 H W
            res_tensor = make_image(imgs, events_tmp, output, target)
            self.writer.add_image('Training: events, image, output, gt.',
                                  make_grid(res_tensor.cpu(), nrow=8, normalize=False))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch_two_input(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target, self.loss_fn)

                self.writer.set_step((epoch - 1) * len(self.val_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val-ing... Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress_val(batch_idx),
                        loss.item()))


                batch_size, _, width, height = target.shape
                events_tmp = torch.zeros((batch_size, 1, width, height))
                # for i in range(data.shape[0]):
                #     events_tmp[i] = torch.from_numpy(
                #         make_event_preview(data[i,  self.img_channels:, :, :].unsqueeze(0))).unsqueeze(0)
                events_tmp = events_tmp.to(self.device)

                res_tensor = make_image(torch.cat((data[:, : self.img_channels, :, :], events_tmp), dim=1), output, target)
                self.writer.add_image('Val: events, image, output, gt.',
                                      make_grid(res_tensor.cpu(), nrow=8, normalize=False))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _valid_epoch_two_input(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (imgs, events, target, images, targets) in enumerate(self.val_data_loader):
                imgs, events, target = imgs.to(self.device), events.to(self.device), target.to(self.device)
                for i in range(len(images)):
                    images[i], targets[i] = images[i].to(self.device), targets[i].to(self.device)

                milti_output, output = self.model(imgs, events)
                loss = self.criterion(milti_output, targets, images, self.loss_fn)

                # ########  without MCD
                # loss = self.criterion(output, target, imgs, self.loss_fn)

                self.writer.set_step((epoch - 1) * len(self.val_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), n=imgs.shape[0])
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), n=imgs.shape[0])

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Val-ing... Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress_val(batch_idx),
                        loss.item()))

                batch_size, _, width, height = target.shape
                if self.mode == 'red-blue':
                    events_tmp = torch.zeros((batch_size, 3, width, height))
                else:
                    events_tmp = torch.zeros((batch_size, 1, width, height))

                for i in range(imgs.shape[0]):
                    if self.mode == 'red-blue':
                        events_tmp[i] = torch.from_numpy(
                            np.transpose(make_event_preview(events[i].unsqueeze(0)), (2, 0, 1)))
                    else:
                        events_tmp[i] = torch.from_numpy(make_event_preview(events[i].unsqueeze(0)))
                events_tmp = events_tmp.to(self.device)

                res_tensor = make_image(imgs, events_tmp, output, target)
                self.writer.add_image('Val: events, image, output, gt.',
                                      make_grid(res_tensor.cpu(), nrow=8, normalize=False))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _progress_val(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.val_data_loader, 'n_samples'):
            current = batch_idx * self.val_data_loader.batch_size
            total = self.val_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch_val
        return base.format(current, total, 100.0 * current / total)

