#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Configer class for all hyper parameters.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

from models.protoseg_core.lib.utils.tools.logger import Logger as Log
from ast import literal_eval

import torch.backends.cudnn as cudnn

def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()

# ProtoSeg params
parser.add_argument('--configs', default="models/protoseg_core/configs/cityscapes/H_48_D_4_proto.json", type=str,dest='configs', help='The file of the hyper parameters.')
parser.add_argument('--phase', default='test', type=str,dest='phase', help='The phase of module.')
parser.add_argument('--gpu', default=[0], nargs='+', type=int,dest='gpu', help='The gpu list used.')

# ***********  Params for data.  **********
parser.add_argument('--data_dir', default='./models/protoseg_core/Cityscapes', type=str, nargs='+',dest='data:data_dir', help='The Directory of the data.')
parser.add_argument('--include_val', type=str2bool, nargs='?', default=False,dest='data:include_val', help='Include validation set for training.')
# include-coarse is only provided for Cityscapes.
parser.add_argument('--include_coarse', type=str2bool, nargs='?', default=False,dest='data:include_coarse', help='Include coarse-labeled set for training.')
parser.add_argument('--only_coarse', type=str2bool, nargs='?', default=False,dest='data:only_coarse', help='Only include coarse-labeled set for training.')
parser.add_argument('--only_mapillary', type=str2bool, nargs='?', default=False,dest='data:only_mapillary', help='Only include mapillary set for training.')
parser.add_argument('--only_small', type=str2bool, nargs='?', default=False,dest='data:only_small', help='Only include small val set for testing.')
# include-atr is used to choose ATR as extra training set for LIP dataset.
parser.add_argument('--include_atr', type=str2bool, nargs='?', default=False,dest='data:include_atr', help='Include atr set for LIP training.')
parser.add_argument('--include_cihp', type=str2bool, nargs='?', default=False,dest='data:include_cihp', help='Include cihp set for LIP training.')
parser.add_argument('--drop_last', type=str2bool, nargs='?', default='y',dest='data:drop_last', help='Fix bug for syncbn.')
parser.add_argument('--train_batch_size', default=None, type=int,dest='train:batch_size', help='The batch size of training.')
parser.add_argument('--val_batch_size', default=None, type=int,dest='val:batch_size', help='The batch size of validation.')

# ***********  Params for checkpoint.  **********
parser.add_argument('--checkpoints_root', default=None, type=str,dest='checkpoints:checkpoints_root', help='The root dir of model save path.')
parser.add_argument('--checkpoints_name', default='hrnet_w48_proto_lr1x_1', type=str,dest='checkpoints:checkpoints_name', help='The name of checkpoint model.')
parser.add_argument('--save_iters', default=None, type=int,dest='checkpoints:save_iters', help='The saving iters of checkpoint model.')
parser.add_argument('--save_epoch', default=None, type=int,dest='checkpoints:save_epoch', help='The saving epoch of checkpoint model.')

# ***********  Params for model.  **********
parser.add_argument('--model_name', default="hrnet_w48_proto", type=str,dest='network:model_name', help='The name of model.')
parser.add_argument('--backbone', default="hrnet48", type=str,dest='network:backbone', help='The base network of model.')
parser.add_argument('--bn_type', default=None, type=str,dest='network:bn_type', help='The BN type of the network.')
parser.add_argument('--multi_grid', default=None, nargs='+', type=int,dest='network:multi_grid', help='The multi_grid for resnet backbone.')
parser.add_argument('--pretrained', type=str, default=None,dest='network:pretrained', help='The path to pretrained model.')
parser.add_argument('--resume', default="./models/protoseg_core/checkpoints/hrnet_w48_proto_lr1x_hrnet_proto_80k_latest.pth", type=str,dest='network:resume', help='The path of checkpoints.')
parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,dest='network:resume_strict', help='Fully match keys or not.')
parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,dest='network:resume_continue', help='Whether to continue training.')
parser.add_argument('--resume_eval_train', type=str2bool, nargs='?', default=True,dest='network:resume_train', help='Whether to validate the training set  during resume.')
parser.add_argument('--resume_eval_val', type=str2bool, nargs='?', default=True,dest='network:resume_val', help='Whether to validate the val set during resume.')
parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,dest='network:gathered', help='Whether to gather the output of model.')
parser.add_argument('--loss_balance', type=str2bool, nargs='?', default=False,dest='network:loss_balance', help='Whether to balance GPU usage.')

# ***********  Params for solver.  **********
parser.add_argument('--optim_method', default=None, type=str,dest='optim:optim_method', help='The optim method that used.')
parser.add_argument('--group_method', default=None, type=str,dest='optim:group_method', help='The group method that used.')
parser.add_argument('--base_lr', default=None, type=float,dest='lr:base_lr', help='The learning rate.')
parser.add_argument('--nbb_mult', default=1.0, type=float,dest='lr:nbb_mult', help='The not backbone mult ratio of learning rate.')
parser.add_argument('--lr_policy', default=None, type=str,dest='lr:lr_policy', help='The policy of lr during training.')
parser.add_argument('--loss_type', default="pixel_prototype_ce_loss", type=str,dest='loss:loss_type', help='The loss type of the network.')
parser.add_argument('--is_warm', type=str2bool, nargs='?', default=False,dest='lr:is_warm', help='Whether to warm training.')

# ***********  Params for display.  **********
parser.add_argument('--max_epoch', default=None, type=int,dest='solver:max_epoch', help='The max epoch of training.')
parser.add_argument('--max_iters', default=None, type=int,dest='solver:max_iters', help='The max iters of training.')
parser.add_argument('--display_iter', default=None, type=int,dest='solver:display_iter', help='The display iteration of train logs.')
parser.add_argument('--test_interval', default=None, type=int,dest='solver:test_interval', help='The test interval of validation.')

# ***********  Params for logging.  **********
parser.add_argument('--logfile_level', default=None, type=str,dest='logging:logfile_level', help='To set the log level to files.')
parser.add_argument('--stdout_level', default=None, type=str,dest='logging:stdout_level', help='To set the level to print to screen.')
parser.add_argument('--log_file', default=None, type=str,dest='logging:log_file', help='The path of log files.')
parser.add_argument('--rewrite', type=str2bool, nargs='?', default=True,dest='logging:rewrite', help='Whether to rewrite files.')
parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,dest='logging:log_to_file', help='Whether to write logging into files.')

# ***********  Params for test or submission.  **********
parser.add_argument('--test_img', default=None, type=str,dest='test:test_img', help='The test path of image.')
parser.add_argument('--test_dir', default='./Cityscapes', type=str,dest='test:test_dir', help='The test directory of images.')
parser.add_argument('--out_dir', default='./result/hrnet_w48_proto_lr1x_1_val_ms', type=str,dest='test:out_dir', help='The test out directory of images.')
parser.add_argument('--save_prob', type=str2bool, nargs='?', default=False, dest='test:save_prob', help='Save the logits map during testing.')

# ***********  Params for env.  **********
parser.add_argument('--seed', default=304, type=int, help='manual seed')
parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')

# ***********  Params for distributed training.  **********
parser.add_argument('--local_rank', type=int, default=-1, dest='local_rank', help='local rank of current process')
parser.add_argument('--distributed', action='store_true', dest='distributed', help='Use multi-processing training.')
parser.add_argument('--use_ground_truth', action='store_true', dest='use_ground_truth', help='Use ground truth for training.')

parser.add_argument('REMAIN', nargs='*')

args_parser = parser.parse_args()

from models.protoseg_core.lib.utils.distributed import handle_distributed
handle_distributed(args_parser, os.path.expanduser(os.path.abspath(__file__)))

cudnn.enabled = True
cudnn.benchmark = args_parser.cudnn

# configer = Configer(args_parser=args_parser)

class Configer(object):

    def __init__(self, args_parser=args_parser, configs=None, config_dict=None):
        if config_dict is not None:
            self.params_root = config_dict

        elif configs is not None:
            if not os.path.exists(configs):
                Log.error('Json Path:{} not exists!'.format(configs))
                exit(0)

            json_stream = open(configs, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

        elif args_parser is not None:
            self.args_dict = args_parser.__dict__
            self.params_root = None

            if not os.path.exists(args_parser.configs):
                print('Json Path:{} not exists!'.format(args_parser.configs))
                exit(1)

            json_stream = open(args_parser.configs, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

            for key, value in self.args_dict.items():
                if not self.exists(*key.split(':')):
                    self.add(key.split(':'), value)
                elif value is not None:
                    self.update(key.split(':'), value)

            self._handle_remaining_args(args_parser.REMAIN)

        self.conditions = _ConditionHelper(self)


    def _handle_remaining_args(self, remain):

        def _parse_value(x: str):
            """
            We first try to parse `x` as python literal object.
            If failed, we regard x as string.
            """
            try:
                return literal_eval(x)
            except ValueError:
                return x

        def _set_value(key, value):
            """
            We directly operate on `params_root`.
            """
            remained_parts = key.split('.')
            consumed_parts = []

            parent_dict = self.params_root
            while len(remained_parts) > 1:
                cur_key = remained_parts.pop(0)
                consumed_parts.append(cur_key)

                if cur_key not in parent_dict:
                    parent_dict[cur_key] = dict()
                    Log.info('{} not exists, set as `dict()`.'.format('.'.join(consumed_parts)))
                elif not isinstance(parent_dict[cur_key], dict):
                    Log.error(
                        'Cannot set {child_name} on {root_name}, as {root_name} is `{root_type}`.'.format(
                            root_name='.'.join(consumed_parts),
                            child_name='.'.join(remained_parts),
                            root_type=type(parent_dict[cur_key])
                        )
                    )
                    sys.exit(1)
                
                parent_dict = parent_dict[cur_key]

            cur_key = remained_parts.pop(0)
            consumed_parts.append(cur_key)

            if cur_key.endswith('+'):
                cur_key = cur_key[:-1]
                target = parent_dict.get(cur_key)

                if not isinstance(target, list):
                    Log.error(
                        'Cannot append to {key}, as its type is {target_type}.'
                        .format(
                            key=key[:-1],
                            target_type=type(target)
                        )
                    )
                    sys.exit(1)

                target.append(value)
                Log.info(
                    'Append {value} to {key}. Current: {target}.'
                    .format(
                        key=key[:-1],
                        value=value,
                        target=target,
                    )
                )
                return

            existing_value = parent_dict.get(cur_key)
            if existing_value is not None:
                Log.warn(
                    'Override {key} using {value}. Previous value: {old_value}.'
                    .format(
                        key=key,
                        value=value,
                        old_value=existing_value
                    )
                )
            else:
                Log.info(
                    'Set {key} as {value}.'.format(key=key, value=value)
                )
            parent_dict[cur_key] = value

        assert len(remain) % 2 == 0, remain
        args = {}
        for i in range(len(remain) // 2):
            key, value = remain[2 * i: 2 * i + 2]
            _set_value(key, _parse_value(value))

    def clone(self):
        from copy import deepcopy
        return Configer(config_dict=deepcopy(self.params_root))

    def _get_caller(self):
        filename = os.path.basename(sys._getframe().f_back.f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_back.f_lineno
        prefix = '{}, {}'.format(filename, lineno)
        return prefix

    def get(self, *key):
        if len(key) == 0:
            return self.params_root

        elif len(key) == 1:
            if key[0] in self.params_root:
                return self.params_root[key[0]]
            else:
                Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        elif len(key) == 2:
            if key[0] in self.params_root and key[1] in self.params_root[key[0]]:
                return self.params_root[key[0]][key[1]]
            else:
                Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        else:
            Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
            exit(1)

    def exists(self, *key):
        if len(key) == 1 and key[0] in self.params_root:
            return True

        if len(key) == 2 and (key[0] in self.params_root and key[1] in self.params_root[key[0]]):
            return True

        return False

    def add(self, key_tuple, value):
        if self.exists(*key_tuple):
            Log.error('{} Key: {} existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        if len(key_tuple) == 1:
            self.params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            if key_tuple[0] not in self.params_root:
                self.params_root[key_tuple[0]] = dict()

            self.params_root[key_tuple[0]][key_tuple[1]] = value

        else:
            Log.error('{} KeyError: {}.'.format(self._get_caller(), key_tuple))
            exit(1)

    def update(self, key_tuple, value):
        if not self.exists(*key_tuple):
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        if len(key_tuple) == 1 and not isinstance(self.params_root[key_tuple[0]], dict):
            self.params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            self.params_root[key_tuple[0]][key_tuple[1]] = value

        else:
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

    def resume(self, config_dict):
        self.params_root = config_dict

    def plus_one(self, *key):
        if not self.exists(*key):
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key))
            exit(1)

        if len(key) == 1 and not isinstance(self.params_root[key[0]], dict):
            self.params_root[key[0]] += 1

        elif len(key) == 2:
            self.params_root[key[0]][key[1]] += 1

        else:
            Log.error('{} KeyError: {} !!!'.format(self._get_caller(), key))
            exit(1)

    def to_dict(self):
        return self.params_root


class _ConditionHelper:
    """Handy helper"""

    def __init__(self, configer):
        self.configer = configer

    @property
    def use_multi_dataset(self):
        root_dirs = self.configer.get('data', 'data_dir')
        return isinstance(root_dirs, (tuple, list)) and len(root_dirs) > 1

    @property
    def pred_sw_offset(self):
        return self.configer.exists('data', 'pred_sw_offset')

    @property
    def pred_dt_offset(self):
        return self.configer.exists('data', 'pred_dt_offset')

    @property
    def use_sw_offset(self):
        return self.configer.exists('data', 'use_sw_offset')

    @property
    def use_dt_offset(self):
        return self.configer.exists('data', 'use_dt_offset')

    @property
    def use_ground_truth(self):
        return self.config_equals(('use_ground_truth',), True)

    @property
    def pred_ml_dt_offset(self):
        return self.configer.exists('data', 'pred_ml_dt_offset')

    def loss_contains(self, name):
        return name in self.configer.get('loss', 'loss_type')

    def model_contains(self, name):
        return name in self.configer.get('network', 'model_name')

    def config_equals(self, key, value):
        if not self.configer.exists(*key):
            return False

        return self.configer.get(*key) == value

    def config_exists(self, key):
        return self.configer.exists(*key)

    def environ_exists(self, key):
        return os.environ.get(key) is not None

    @property
    def diverse_size(self):
        return self.configer.get('val', 'data_transformer')['size_mode'] == 'diverse_size'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='../../configs/cls/flower/fc_vgg19_flower_cls.json', type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of Pose Estimator.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of pretrained model.')
    parser.add_argument('--train_dir', default=None, type=str,
                        dest='data:train_dir', help='The path of train data.')

    args_parser = parser.parse_args()

    configer = Configer(args_parser=args_parser)

    configer.add(('project_dir',), 'root')
    configer.update(('project_dir',), 'root1')

    print (configer.get('project_dir'))
    print (configer.get('network', 'resume'))
    print (configer.get('logging', 'log_file'))
    print(configer.get('data', 'train_dir'))