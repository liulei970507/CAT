from collections import OrderedDict

pretrain_opts = OrderedDict()
pretrain_opts['use_gpu'] = True

pretrain_opts['init_model_path'] = './models/imagenet-vgg-m.mat'
pretrain_opts['model_path'] = './models'

pretrain_opts['batch_frames'] = 8
pretrain_opts['batch_pos'] = 64
pretrain_opts['batch_neg'] = 196

pretrain_opts['overlap_pos'] = [0.7, 1]
pretrain_opts['overlap_neg'] = [0, 0.5]

pretrain_opts['img_size'] = 107


pretrain_opts['lr'] = 0.0001
pretrain_opts['w_decay'] = 0.0005
pretrain_opts['momentum'] = 0.9
pretrain_opts['grad_clip'] = 10
#stage1 common
#pretrain_opts['ft_layers'] = ['parallel','fc']
#pretrain_opts['lr_mult'] = {'parallel':10,'fc':5}
#pretrain_opts['n_cycles'] = 1000
#stage2 guide
#pretrain_opts['ft_layers'] = ['guidance','fc']
#pretrain_opts['lr_mult'] = {'guidance':10,'fc':5}
#pretrain_opts['n_cycles'] = 1000
##stage3 ensemble
pretrain_opts['ft_layers'] = ['ensemble','fc','layer']
pretrain_opts['lr_mult'] = {'ensemble':10,'fc':5,'layer':1}
pretrain_opts['n_cycles'] = 1000

##################################### from RCNN #############################################
pretrain_opts['padding'] = 1.2
pretrain_opts['padding_ratio']=5.
pretrain_opts['padded_img_size'] = pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
pretrain_opts['frame_interval'] = 2
