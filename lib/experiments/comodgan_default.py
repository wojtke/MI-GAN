import copy
import json
import os
import os.path as osp
import pickle
import time

import dnnlib
import numpy as np
import PIL.Image
import psutil
import torch

from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix

from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.data_factory import get_dataset, collate, DistributedSampler
from lib.model_zoo import get_model
from lib.log_service import print_log

from .comodgan_loss import CoModGanLoss
from ..evaluator.stylegan_metrics import metric_main


def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')


def check_and_create_dir(path):
    while not osp.exists(path):
        if torch.distributed.get_rank() == 0:
            os.makedirs(path)
            break
        time.sleep(0.01)


class draw_functor:
    def __init__(self, RANK, grid_size, batch_gpu, log_dir):
        self.RANK = RANK
        self.grid_size = grid_size
        self.batch_gpu = batch_gpu
        self.log_dir = log_dir
        self.input = None
        self.subfolder = 'udemo'
        
    def __call__(self, **kwargs):
        RANK = self.RANK
        generator = kwargs['generator']
        filename = kwargs['filename'] if 'filename' in kwargs else 'demo.png'
        isinit = kwargs['isinit'] if 'isinit' in kwargs else False

        if 'input' in kwargs:
            images = [generator(**input_i).cpu() for input_i in kwargs['input']]
            # recover grid_real and grid_mask from overrided input
            grid_real = torch.cat([input_i['x'][:, 1:] for input_i in kwargs['input']])
            grid_mask = torch.cat([input_i['x'][:, 0:1] for input_i in kwargs['input']])+0.5
            grid_real, grid_mask = grid_real.detach().to('cpu'), grid_mask.detach().to('cpu')
        elif self.input is not None:
            images = [generator(**input_i).cpu() for input_i in self.input]
            # recover grid_real and grid_mask from saved input
            grid_real = torch.cat([input_i['x'][:, 1:] for input_i in self.input])
            grid_mask = torch.cat([input_i['x'][:, 0:1] for input_i in self.input])+0.5
            grid_real, grid_mask = grid_real.detach().to('cpu'), grid_mask.detach().to('cpu')
        else:
            evalset = kwargs['evalset']
            z_dim = kwargs['z_dim']
            grid_size_n = np.prod(self.grid_size)
            grid_z = torch.randn([grid_size_n, z_dim])
            grid_c = torch.randn([grid_size_n, 0])
            grid_real, grid_mask, _ = zip(*[evalset[i] for i in range(grid_size_n)])
            # To avoid a wierd bug in ZipFile VVVV
            if getattr(evalset.loader[0], 'zipfile_close', None) is not None:
                evalset.loader[0].zipfile_close()
            grid_real = torch.stack([torch.FloatTensor(i) for i in grid_real])
            grid_mask = torch.stack([torch.FloatTensor(i) for i in grid_mask]).unsqueeze(1)
            grid_real_erased = grid_real * grid_mask
            grid_x = torch.cat([grid_mask-0.5, grid_real_erased], axis=1)
            if RANK >= 0:
                grid_z, grid_c, grid_x = grid_z.to(RANK), grid_c.to(RANK), grid_x.to(RANK)
            grid_z = grid_z.split(self.batch_gpu)
            grid_c = grid_c.split(self.batch_gpu)
            grid_x = grid_x.split(self.batch_gpu)

            self.input = [{
                    'x': x,
                    'z': z, 
                    'c': c, 
                    'noise_mode': 'const',
                } for x, z, c in zip(grid_x, grid_z, grid_c)]
            images = [generator(**input_i).cpu() for input_i in self.input]

        images = torch.cat(images)
        self.output(images, os.path.join(
            self.log_dir, self.subfolder, filename),
            drange=[-1, 1], grid_size=self.grid_size)
        filename_combined = osp.splitext(filename)
        filename_combined = filename_combined[0]+'_combined'+filename_combined[1]
        images_combined = grid_real * grid_mask + images * (1-grid_mask)
        self.output(images_combined, os.path.join(
            self.log_dir, self.subfolder, filename_combined),
            drange=[-1, 1], grid_size=self.grid_size)

        if isinit:
            self.output(grid_mask, os.path.join(
                self.log_dir, self.subfolder, 'masks.png'),
                drange=[0, 1], grid_size=self.grid_size)
            self.output(grid_real, os.path.join(
                self.log_dir, self.subfolder, 'reals.png'),
                drange=[-1, 1], grid_size=self.grid_size)
            self.output(grid_real_erased, os.path.join(
                self.log_dir, self.subfolder, 'erased.png'),
                drange=[-1, 1], grid_size=self.grid_size)
            
    def output(self, img, fname, drange, grid_size):
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

        gw, gh = grid_size
        _N, C, H, W = img.shape
        img = img.reshape(gh, gw, C, H, W)
        img = img.transpose(0, 3, 1, 4, 2)
        img = img.reshape(gh * H, gw * W, C)

        assert C in [1, 3]
        check_and_create_dir(osp.split(fname)[0])
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)


class train_stage:
    def __init__(self):
        super().__init__()
        self.draw_function = draw_functor
        self.loss_function = CoModGanLoss
        self.stat_best = None
    
    def is_better(self, stat):
        if self.stat_best is not None:
            better = stat['results'][stat['metric']] < self.stat_best['results'][stat['metric']]
        else:
            better = True
        if better:
            self.stat_best = copy.deepcopy(stat)
        return better

    def resume(self, resume_pkl, models):
        print('Resuming from {}'.format(resume_pkl))
        G, D, G_ema = models
        with open(resume_pkl, 'rb') as f:
            resume_data = pickle.Unpickler(f).load()
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        return G, D, G_ema

    def get_ddp_modules(self, RANK, G, D, **kwargs):
        ddp_modules = {
            'G_mapping'  : G.mapping, 
            'G_encoder'  : G.encoder,
            'G_synthesis': G.synthesis, 
            'D'          : D,
        }
        ddp_modules.update(kwargs)
        for name in ddp_modules.keys():
            module = ddp_modules[name]
            if len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[RANK], broadcast_buffers=False)
                module.requires_grad_(False)
            ddp_modules[name] = module
        return ddp_modules

    def check_ddp_consistency(self, module):
        if cfguh().cfg.model_g.args.synthesis.type in [
                'comodgan_synthesis_ffc', 'comodgan_synthesis_shuv3', 'comodgan_synthesis_shuv4']:
            misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg|.*\.running_mean|.*\.running_var')
        else:
            misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')

    def main(self,
             batch,
             net,
             RANK,
             **kwargs):
        cuda = cfguh().cfg.env.cuda
        cfgt = cfguh().cfg.train
        batch_gpu = cfgt.batch_size_per_gpu
        effective_batch_gpu = batch_gpu 
        # used as a case when GPU is small and cannot maintain the original batch_gpu
        net_g, net_d = net
        phases = kwargs['phases']
        batch_idx = kwargs['batch_idx']
        loss = kwargs['loss']

        real, mask, fn = batch
        real = real.to(torch.float32)
        mask = mask[:, None].to(torch.float32)
        real_erased = real * mask
        real_c = torch.empty([batch_gpu, 0])

        all_gen_c = torch.zeros([len(phases) * batch_gpu, 0]).pin_memory()
        all_gen_z = torch.randn([len(phases) * batch_gpu, net_g.z_dim])

        if cuda:
            real, mask, real_erased, real_c, all_gen_c, all_gen_z = [
                i.to(RANK) for i in [real, mask, real_erased, real_c, all_gen_c, all_gen_z]]
        
        real, mask, real_erased, real_c = [
            i.split(effective_batch_gpu) for i in [real, mask, real_erased, real_c]]
        all_gen_z = [phase_gen_z.split(effective_batch_gpu) for phase_gen_z in all_gen_z.split(batch_gpu)]
        all_gen_c = [phase_gen_c.split(effective_batch_gpu) for phase_gen_c in all_gen_c.split(batch_gpu)]

        # Execute training phases.
        for phase, gen_z, gen_c in zip(phases, all_gen_z, all_gen_c):

            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(RANK))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (x, m, x0, c0, z, c1) in enumerate(zip(real, mask, real_erased, real_c, gen_z, gen_c)):
                sync = (round_idx == batch_gpu // effective_batch_gpu - 1)
                gain = phase.interval
                loss.accumulate_gradients(
                    phase=phase.name, real_img=x, mask=m, real_img_erased=x0, 
                    real_c=c0, gen_z=z, gen_c=c1, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(RANK))
        return {}

    def __call__(self, RANK, **paras):
        cfg = cfguh().cfg
        cfge   = cfg.env
        cfgt   = cfg.train
        cfgv   = cfg.eval
        cfgm_g = cfg.model_g
        cfgm_d = cfg.model_d
        isresume = getattr(cfge, 'resume_path', None) is not None

        # Initialize.
        start_time = time.time()
        device = torch.device('cuda', RANK)
        np.random.seed(cfge.rnd_seed * cfge.gpu_count + RANK)
        torch.manual_seed(cfge.rnd_seed * cfge.gpu_count + RANK)
        torch.backends.cudnn.benchmark = cfge.cudnn_benchmark    # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = cfge.allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = cfge.allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
        conv2d_gradfix.enabled = True                            # Improves training speed.
        grid_sample_gradfix.enabled = True                       # Avoids errors with the augmentation pipe.

        batch_size = cfgt.batch_size
        batch_gpu = cfgt.batch_size_per_gpu
        num_workers_gpu = cfgt.dataset_num_workers_per_gpu

        if RANK == 0:
            highlight_print('Setting')
            print_log('RNDSEED:', cfge.rnd_seed)
            print_log('BATCH_SIZE:', cfgt.batch_size)
            print_log('BATCH_SIZE_PER_GPU:', cfgt.batch_size_per_gpu)
            print_log('TRAIN_DATASET:', cfgt.dataset.name)
            print_log('VAL_DATASET:', cfgv.dataset.name)
            print_log('MODEL_G:', cfgm_g.name)
            print_log('MODEL_D:', cfgm_d.name)

        if cfge.debug and RANK==0:
            highlight_print('Debug')

        #####################
        # Load training set #
        #####################

        if RANK == 0:
            highlight_print('My dataset')

        trainset = get_dataset()(cfgt.dataset)
        sampler = DistributedSampler(dataset=trainset)
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size = batch_gpu,
            sampler = sampler, 
            num_workers = num_workers_gpu,
            drop_last = True, 
            pin_memory = True,
            collate_fn = collate(), 
        )

        batch_gpu_eval = cfgv.batch_size_per_gpu
        num_workers_gpu_eval = cfgv.dataset_num_workers_per_gpu
        evalset = get_dataset()(cfgv.dataset)
        sampler = DistributedSampler(evalset, shuffle=False, extend=True)
        evalloader = torch.utils.data.DataLoader(
            evalset, 
            batch_size = batch_gpu_eval,
            sampler = sampler,
            num_workers = num_workers_gpu_eval,
            drop_last = False, 
            pin_memory = False,
            collate_fn = collate(), 
        )

        ######################
        # Construct networks #
        ######################

        if RANK == 0:
            highlight_print('My model')

        G = get_model()(cfgm_g).train().requires_grad_(False).to(device)
        D = get_model()(cfgm_d).train().requires_grad_(False).to(device)
        G_ema = copy.deepcopy(G).eval()

        # Resume from existing pickle.
        if isresume and (RANK == 0):
            resume_path = getattr(cfge, 'resume_path', None)
            resume_pkl = osp.join(resume_path, 'weight', 'network-snapshot-{:06d}.pkl'.format(cfge.resume_itern))
            G, D, G_ema = self.resume(resume_pkl = resume_pkl, models=[G, D, G_ema])

        augment_pipe = None

        # Distribute across GPUs.
        if RANK == 0:
            highlight_print('Distributing across GPUs...')

        ddp_modules = self.get_ddp_modules(RANK, G, D, G_ema=G_ema)
        ddp_modules.pop('G_ema')
        ddp_modules['augment_pipe'] = augment_pipe

        # Setup training phases.
        if RANK == 0:
            highlight_print('Setting up training phases...')

        loss = self.loss_function(device=device, **ddp_modules, **cfgt.loss_kwargs)
        # subclass of training.loss.Loss
        phases = []
        for name, module, opt_kwargs, reg_interval in [
                ('G', G, cfgt.g_opt_kwargs, cfgt.g_reg_interval), 
                ('D', D, cfgt.d_opt_kwargs, cfgt.d_reg_interval)]:
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) 
                # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) 
                # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
        for phase in phases:
            phase.start_event = None
            phase.end_event = None
            if RANK == 0:
                phase.start_event = torch.cuda.Event(enable_timing=True)
                phase.end_event = torch.cuda.Event(enable_timing=True)

        # Export sample images.
        if RANK == 0:
            highlight_print('Exporting sample images...')
            demof = self.draw_function(RANK, [8, 6], batch_gpu, cfgt.log_dir)
            demof(z_dim=G.z_dim, generator=G_ema, evalset=evalset, 
                  filename='fakes_init.png', isinit=True)

        # Initialize logs.
        if RANK == 0:
            highlight_print('Initializing logs...')

        stats_collector = training_stats.Collector(regex='.*')
        stats_metrics = dict()
        stats_jsonl = None
        stats_tfevents = None
        if RANK == 0:
            stats_jsonl = open(os.path.join(cfgt.log_dir, 'stats.jsonl'), 'wt')
            try:
                import torch.utils.tensorboard as tensorboard
                stats_tfevents = tensorboard.SummaryWriter(osp.join(cfgt.log_dir, 'tensorboard'))
            except ImportError as err:
                print('Skipping tfevents export:', err)

        # Train.
        if RANK == 0:
            highlight_print('Training for {} kimg...'.format(cfgt.total_kimg))

        if isresume:
            cur_nimg = (cfge.resume_itern * 1000 // batch_size) * batch_size
            cur_tick = 0
            tick_start_nimg = cur_nimg
            batch_idx = cfge.resume_itern * 1000 // batch_size
        else:
            cur_nimg = 0
            cur_tick = 0
            tick_start_nimg = cur_nimg
            batch_idx = 0

        # Resume info
        if (RANK == 0) and isresume:
            highlight_print('Resume from {} kimg...'.format(cfge.resume_itern))
            print_log('Resume from {}'.format(cfge.resume_path))
            print_log('cur_nimg {}, batch_idx {}'.format(cur_nimg, batch_idx))


        tick_start_time = time.time()
        maintenance_time = tick_start_time - start_time

        done = False
        while not done:
            for batch in trainloader:
                rv = self.main(batch, [G, D], RANK, phases=phases, batch_idx=batch_idx, loss=loss)

                # Update G_ema.
                ema_nimg = cfgt.ema_kimg * 1000
                if cfgt.ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * cfgt.ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)

                # Update state.
                cur_nimg += batch_size
                batch_idx += 1

                # Perform maintenance tasks once per tick.
                done = (cur_nimg >= cfgt.total_kimg * 1000)

                if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + cfgt.kimg_per_tick * 1000):
                    continue

                # Print status line, accumulating the same information in stats_collector.
                tick_end_time = time.time()
                fields = []
                fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
                fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
                fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
                fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
                fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
                fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
                fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
                fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
                torch.cuda.reset_peak_memory_stats()
                fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
                training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
                training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
                if RANK == 0:
                    print_log(' '.join(fields))

                snapshot_data = None
                flag_better = False

                ####################
                # Evaluate metrics #
                ####################

                snapshot_ticks = getattr(cfgt.snapshot, 'evaluate', None)
                if (len(cfgt.metrics) > 0) and (snapshot_ticks is not None) \
                        and (done or cur_tick % snapshot_ticks == 0):
                    if RANK == 0:
                        print('Evaluating metrics...')

                    snapshot_data = {}
                    for name, module in [('G', G), 
                                         ('D', D), 
                                         ('G_ema', G_ema), 
                                         ('augment_pipe', augment_pipe)]:
                        if module is not None:
                            if cfge.gpu_count > 1:
                                self.check_ddp_consistency(module)
                            module = copy.deepcopy(module).eval().requires_grad_(False)
                        snapshot_data[name] = module

                    for metric in cfgt.metrics:
                        result_dict = metric_main.calc_metric(
                            metric=metric, 
                            G=snapshot_data['G_ema'],
                            num_gpus=cfge.gpu_count, 
                            rank=RANK, 
                            device=device, 
                            evalloader=evalloader,)
                        stats_metrics.update(result_dict.results)

                    if RANK == 0:
                        metric_main.report_metric(result_dict, run_dir=cfgt.log_dir, snapshot_pkl='{:06d}'.format(cur_nimg//1000))
                    flag_better = self.is_better(result_dict)

                #######################
                # Save image snapshot #
                #######################

                snapshot_ticks = getattr(cfgt.snapshot, 'image', None)
                snapshot_cond = (snapshot_ticks is not None) \
                    and (done or cur_tick % snapshot_ticks == 0)

                if (snapshot_cond or flag_better) and (snapshot_data is None):
                    snapshot_data = {}
                    for name, module in [('G', G), 
                                         ('D', D), 
                                         ('G_ema', G_ema), 
                                         ('augment_pipe', augment_pipe)]:
                        if module is not None:
                            if cfge.gpu_count > 1:
                                self.check_ddp_consistency(module)
                            module = copy.deepcopy(module).eval().requires_grad_(False)
                        snapshot_data[name] = module
                        del module # conserve memory

                if (RANK == 0) and snapshot_cond:
                    print_log('Save image snapshot...')
                    demof(generator=snapshot_data['G_ema'], filename='fakes{:06d}.png'.format(cur_nimg//1000))
                if (RANK == 0) and flag_better:
                    demof(generator=snapshot_data['G_ema'], filename='fakes_best.png')

                #########################
                # Save network snapshot #
                #########################
                
                snapshot_ticks = getattr(cfgt.snapshot, 'checkpoint', None)
                snapshot_cond = (snapshot_ticks is not None) \
                    and (done or cur_tick % snapshot_ticks == 0)

                if snapshot_cond and (snapshot_data is None):
                    snapshot_data = {}
                    for name, module in [('G', G), 
                                         ('D', D), 
                                         ('G_ema', G_ema), 
                                         ('augment_pipe', augment_pipe)]:
                        if module is not None:
                            if cfge.gpu_count > 1:
                                self.check_ddp_consistency(module)
                            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                        snapshot_data[name] = module
                        del module # conserve memory
                elif snapshot_cond:
                    for name in snapshot_data:
                        if snapshot_data[name] is not None:
                            snapshot_data[name] = snapshot_data[name].cpu()

                if (RANK == 0) and snapshot_cond:
                    check_and_create_dir(osp.join(cfgt.log_dir, 'weight'))
                    snapshot_pkl = os.path.join(cfgt.log_dir, 'weight', f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                    with open(snapshot_pkl, 'wb') as f:
                        pickle.dump(snapshot_data, f)                        
                if (RANK == 0) and flag_better:
                    check_and_create_dir(osp.join(cfgt.log_dir, 'weight'))
                    snapshot_pkl = os.path.join(cfgt.log_dir, 'weight', f'network-snapshot-best.pkl')
                    with open(snapshot_pkl, 'wb') as f:
                        pickle.dump(snapshot_data, f)

                del snapshot_data # conserve memory

                ######################
                # Collect statistics #
                ######################

                for phase in phases:
                    value = []
                    if (phase.start_event is not None) and (phase.end_event is not None):
                        phase.end_event.synchronize()
                        try:
                            # When resume from checkpoint, some event may be uninitalized.
                            value = phase.start_event.elapsed_time(phase.end_event)
                        except:
                            value = 0
                    training_stats.report0('Timing/' + phase.name, value)
                stats_collector.update()
                stats_dict = stats_collector.as_dict()

                # Update logs.
                timestamp = time.time()
                if stats_jsonl is not None:
                    fields = dict(stats_dict, timestamp=timestamp)
                    stats_jsonl.write(json.dumps(fields) + '\n')
                    stats_jsonl.flush()
                if stats_tfevents is not None:
                    global_step = int(cur_nimg / 1e3)
                    walltime = timestamp - start_time
                    for name, value in stats_dict.items():
                        stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                    for name, value in stats_metrics.items():
                        stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                    stats_tfevents.flush()

                # Update state.
                cur_tick += 1
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()
                maintenance_time = tick_start_time - tick_end_time
                if done:
                    break

        # Done.
        if RANK == 0:
            highlight_print('Exiting...')

##############
# eval stage #
##############

def legacy_load_G(G, G_tfmodel):
    from collections import OrderedDict

    def mapping_G(k):
        if k == 'lod':
            return None
        if k.find('noise') != -1:
            return None
        newk = k
        newk = newk.replace('/', '.')
        newk = newk.replace('dlatent_avg', 'mapping.w_avg')
        newk = newk.replace('Dense', 'fc')
        newk = newk.replace('synthesis.', '')
        for i in [1024, 512, 256, 128, 64, 32, 16, 8, 4]:
            newk = newk.replace('E_{0}x{0}'.format(i), 'encoder.b{}'.format(i))
            newk = newk.replace('G_{0}x{0}'.format(i), 'synthesis.b{}'.format(i))
        newk = newk.replace('b4.fc0'.format(i), 'b4.fc'.format(i))
        newk = newk.replace('b4.Conv'.format(i), 'b4.conv'.format(i))

        newk = newk.replace('Conv0_up'.format(i), 'conv0'.format(i))
        newk = newk.replace('Conv1_down'.format(i), 'conv1'.format(i))
        newk = newk.replace('Conv0'.format(i), 'conv0'.format(i))
        newk = newk.replace('Conv1'.format(i), 'conv1'.format(i))
        newk = newk.replace('ToRGB'.format(i), 'torgb'.format(i))
        newk = newk.replace('FromRGB'.format(i), 'fromrgb'.format(i))
        newk = newk.replace('mod_'.format(i), 'affine.'.format(i))
        return newk

    G_para = []
    for ki, vi in G_tfmodel.items():
        newki = mapping_G(ki)
        if newki is None:
            continue
        if ((newki.find('conv.weight') != -1)
                or (newki.find('conv0.weight') != -1)
                or (newki.find('conv1.weight') != -1)
                or (newki.find('fromrgb.weight') != -1)
                or (newki.find('torgb.weight') != -1) ):
            # Flip the convolution kernel from tf to torch
            G_para.append((newki, torch.FloatTensor(vi.transpose(3, 2, 0, 1)) ))
        elif ((newki.find('affine.weight') != -1)
                or ((newki.find('mapping.fc') != -1) and (newki.find('weight') != -1))
                or (newki.find('fc.weight') != -1) ):
            # Flip the fc kernel from tf to torch
            G_para.append((newki, torch.FloatTensor(vi.transpose(1, 0)) ))
        elif (newki.find('affine.bias') != -1):
            G_para.append((newki, torch.FloatTensor(vi)+1 ))
        else:
            G_para.append((newki, torch.FloatTensor(vi) ))

    G_para = OrderedDict(G_para)
    G_para_next = G.state_dict()
    G_para_next.update(G_para)
    G.load_state_dict(G_para_next)
    return G

def legacy_load_D(D, D_tfmodel):
    from collections import OrderedDict

    def mapping_D(k):
        if k == 'lod':
            return None
        if k.find('noise') != -1:
            return None
        newk = k
        newk = newk.replace('/', '.')
        for i in [1024, 512, 256, 128, 64, 32, 16, 8, 4]:
            newk = newk.replace('{0}x{0}'.format(i), 'b{}'.format(i))
        newk = newk.replace('b4.Dense0'.format(i), 'b4.fc'.format(i))
        newk = newk.replace('b4.Conv'.format(i), 'b4.conv'.format(i))
        newk = newk.replace('Output'.format(i), 'b4.out'.format(i))

        newk = newk.replace('Conv0'.format(i), 'conv0'.format(i))
        newk = newk.replace('Conv1_down'.format(i), 'conv1'.format(i))
        newk = newk.replace('Skip'.format(i), 'skip'.format(i))
        newk = newk.replace('FromRGB'.format(i), 'fromrgb'.format(i))
        return newk

    D_para = []
    for ki, vi in D_tfmodel.items():
        newki = mapping_D(ki)
        if newki is None:
            continue
        if ((newki.find('conv.weight') != -1)
                or (newki.find('conv0.weight') != -1)
                or (newki.find('conv1.weight') != -1)
                or (newki.find('fromrgb.weight') != -1)
                or (newki.find('skip.weight') != -1) ):
            # Flip the convolution kernel from tf to torch
            D_para.append((newki, torch.FloatTensor(vi.transpose(3, 2, 0, 1)) ))
        elif ((newki.find('fc.weight') != -1)
                or (newki.find('out.weight') != -1) ):
            # Flip the fc kernel from tf to torch
            D_para.append((newki, torch.FloatTensor(vi.transpose(1, 0)) ))
        else:
            D_para.append((newki, torch.FloatTensor(vi) ))

    D_para = OrderedDict(D_para)
    D_para_next = D.state_dict()

    D_para_next.update(D_para)
    D.load_state_dict(D_para_next)
    return D

class eval_stage:
    def __init__(self):
        self.draw_function = draw_functor
        
    def load_network_pkl(self, f):
        from easydict import EasyDict as edict

        class _TFNetworkStub(edict):
            pass

        class _LegacyUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'dnnlib.tflib.network' and name == 'Network':
                    return _TFNetworkStub
                return super().find_class(module, name)

        def collect_tf_params(tf_net):
            # pylint: disable=protected-access
            tf_params = dict()
            def recurse(prefix, tf_net):
                for name, value in tf_net.variables:
                    tf_params[prefix + name] = value
                for name, comp in tf_net.components.items():
                    recurse(prefix + name + '/', comp)
            recurse('', tf_net)
            return tf_params

        def convert_tf(tf_G):
            if tf_G.version < 4:
                raise ValueError('TensorFlow pickle version too low')
            return collect_tf_params(tf_G)

        data = _LegacyUnpickler(f).load()
        if isinstance(data, tuple) and all(isinstance(net, _TFNetworkStub) for net in data):
            data = [convert_tf(i) for i in data]
            isTensorFlow = True
        else:
            isTensorFlow = False

        return data, isTensorFlow

    def resume(self, resume_pkl, models):
        print('Resuming from {}'.format(resume_pkl))
        G, D, G_ema = models
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data, istfmodel = self.load_network_pkl(f)
        if istfmodel:
            G_tfmodel, D_tfmodel, G_ema_tfmodel = resume_data
            G = legacy_load_G(G, G_tfmodel)
            D = legacy_load_D(D, D_tfmodel)
            G_ema = legacy_load_G(G_ema, G_ema_tfmodel)
            # debug
            G_ema.mapping.load_state_dict(G.mapping.state_dict())
        else:
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        return G, D, G_ema

    def get_ddp_modules(self, RANK, G, D, **kwargs):
        ddp_modules = {
            'G_mapping'  : G.mapping, 
            'G_encoder'  : G.encoder,
            'G_synthesis': G.synthesis, 
            'D'          : D,
        }
        ddp_modules.update(kwargs)
        for name in ddp_modules.keys():
            module = ddp_modules[name]
            if len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[RANK], broadcast_buffers=False)
                module.requires_grad_(False)
            ddp_modules[name] = module
        return ddp_modules
    
    def check_ddp_consistency(self, module):
        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')

    def __call__(self, RANK, **paras):
        cfg = cfguh().cfg
        cfge   = cfg.env
        cfgv   = cfg.eval
        cfgm_g = cfg.model_g
        cfgm_d = cfg.model_d

        # Initialize.
        device = torch.device('cuda', RANK)
        np.random.seed(cfge.rnd_seed * cfge.gpu_count + RANK)
        torch.manual_seed(cfge.rnd_seed * cfge.gpu_count + RANK)
        torch.backends.cudnn.benchmark = cfge.cudnn_benchmark    # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = cfge.allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = cfge.allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
        conv2d_gradfix.enabled = True                            # Improves training speed.
        grid_sample_gradfix.enabled = True                       # Avoids errors with the augmentation pipe.

        if RANK == 0:
            highlight_print('Setting')
            print_log('RNDSEED:', cfge.rnd_seed)
            print_log('VAL_DATASET:', cfgv.dataset.name)
            print_log('MODEL_G:', cfgm_g.name)
            print_log('MODEL_D:', cfgm_d.name)

        if cfge.debug and RANK==0:
            highlight_print('Debug')

        #################
        # Load eval set #
        #################

        if RANK == 0:
            highlight_print('My dataset')

        batch_gpu_eval = cfgv.batch_size_per_gpu
        num_workers_gpu_eval = cfgv.dataset_num_workers_per_gpu
        evalset = get_dataset()(cfgv.dataset)
        sampler = DistributedSampler(evalset, shuffle=False, extend=True)
        evalloader = torch.utils.data.DataLoader(
            evalset, 
            batch_size = batch_gpu_eval,
            sampler = sampler,
            num_workers = num_workers_gpu_eval,
            drop_last = True, 
            pin_memory = False,
            collate_fn = collate(), 
        )

        ######################
        # Construct networks #
        ######################

        if RANK == 0:
            highlight_print('My model')

        G = get_model()(cfgm_g).eval().requires_grad_(False).to(device)
        D = get_model()(cfgm_d).eval().requires_grad_(False).to(device)
        G_ema = copy.deepcopy(G).eval()

        # Resume from existing pickle.
        resume_pkl = getattr(cfgv, 'pretrained_pkl', None)
        if (resume_pkl is not None) and (RANK == 0):
            resume_pkl = osp.abspath(osp.join(cfg.eval.log_dir, '..', resume_pkl))
            G, D, G_ema = self.resume(resume_pkl = resume_pkl, models=[G, D, G_ema])

        G = get_model()(cfgm_g).eval().requires_grad_(False).to(device)

        augment_pipe = None

        # Distribute across GPUs.
        if RANK == 0:
            highlight_print('Distributing across GPUs...')

        ddp_modules = self.get_ddp_modules(RANK, G, D, G_ema=G_ema)
        ddp_modules.pop('G_ema')
        ddp_modules['augment_pipe'] = augment_pipe

        # Export sample images.
        if RANK == 0:
            highlight_print('Exporting sample images...')
            demof = self.draw_function(RANK, [8, 6], batch_gpu_eval, cfgv.log_dir)
            demof(z_dim=G.z_dim, generator=G_ema, evalset=evalset, 
                  filename='fakes.png', isinit=True)

        if len(cfgv.metrics) <= 0:
            return

        if RANK == 0:
            highlight_print('Evaluating metrics...')

        stats_collector = training_stats.Collector(regex='.*')
        stats_metrics = dict()
        if RANK == 0:
            stats_jsonl = open(os.path.join(cfgv.log_dir, 'stats.jsonl'), 'wt')

        for metric in cfgv.metrics:
            # Control on mask random
            np.random.seed(cfge.rnd_seed * cfge.gpu_count + RANK)
            torch.manual_seed(cfge.rnd_seed * cfge.gpu_count + RANK)

            result_dict = metric_main.calc_metric(
                metric=metric, 
                G=G_ema,
                num_gpus=cfge.gpu_count, 
                rank=RANK, 
                device=device, 
                evalloader=evalloader,)
            if RANK == 0:
                metric_main.report_metric(
                    result_dict, run_dir=cfgv.log_dir, snapshot_pkl=None)
            stats_metrics.update(result_dict.results)

        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if RANK == 0:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
