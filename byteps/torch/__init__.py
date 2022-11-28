# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

from byteps.torch.compression import Compression
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import push_pull
from byteps.torch.ops import batched_fuse_, batched_unfuse_, batched_zero_
from byteps.torch.ops import byteps_torch_set_num_grads
from byteps.torch.ops import poll, synchronize, declare
from byteps.torch.ops import init, shutdown, suspend, resume
from byteps.torch.ops import size, local_size, rank, local_rank
from byteps.torch.ops import send_async, recv_async


import os
import torch
import collections
import io
try:
    import queue
except ImportError:
    import Queue as queue
import logging
import threading
import time
import numpy as np
#@yrchen: for debug
cache_time = []
cache_opt_time = []
copy_time = []
step_time = []
restore_time = []
total_time = []
class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression,
                 backward_passes_per_step=1, staleness=0,
                 pipesgd_warmup_iter=0,
                 pipesgd_dc_lambda=0,
                 pipesgd_dc_outerprod=False,
                 pipesgd_dc_adaptive=False,
                 pipesgd_dc_adaptive_eps=1e-7,
                 pipesgd_weight_prediction=None,
                 partial_stale_layer=0, model=None
):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        self._niter = 0
        self._staleness = staleness
        self._pipesgd_warmup_iter = pipesgd_warmup_iter
        assert staleness == 0 or staleness == 1, staleness

        # pipesgd staleness mitigation
        self._pipesgd_dc_lambda = pipesgd_dc_lambda
        self._pipesgd_dc_outerprod = pipesgd_dc_outerprod
        self._pipesgd_dc_adaptive = pipesgd_dc_adaptive
        self._pipesgd_dc_adaptive_eps = pipesgd_dc_adaptive_eps
        self._pipesgd_weight_prediction = pipesgd_weight_prediction
        self._pipesgd_num_cached_weight = 0
        if staleness == 1:
            assert pipesgd_dc_lambda >= 0, pipesgd_dc_lambda
            assert pipesgd_dc_adaptive_eps > 0, pipesgd_dc_adaptive_eps
            assert pipesgd_weight_prediction in [None, "local", "prev_sync", "local_prev_sync", "prev_sync_dc", "local_prev_sync_dc"], pipesgd_weight_prediction
            if pipesgd_weight_prediction is not None and self._pipesgd_warmup_iter <= 0:
                self._pipesgd_warmup_iter = 1
                print('BytePS: weight prediction requires pipesgd_warmup_iter, so that the optimizer states could be initialized first, pipesgd_warmup_iter is automatically set to 1')
            if pipesgd_weight_prediction is not None:
                self._pipesgd_num_cached_weight += 1
                if pipesgd_weight_prediction.endswith("_dc"):
                    self._pipesgd_num_cached_weight += 1
            print('BytePS: weight prediction requires caching the weights in the last %d steps' % (self._pipesgd_num_cached_weight))



        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER')) > 1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        if len(named_parameters) > 0:
            if isinstance(named_parameters[0][1], torch.Tensor):
                if any([not isinstance(p, torch.Tensor) for name, p in named_parameters]):
                    raise ValueError('named_parameters should consistently be a sequence of '
                                     'tuples (name, torch.Tensor)')
                self._is_tensor_instance = True
                # there is an issue when using torch.Tensor as key, so use its hash instead
                # https://github.com/pytorch/pytorch/issues/7733
                self._parameter_names = {v.__hash__(): k for k, v
                                         in sorted(named_parameters)}
                self._tensor_list = [tensor for name, tensor in named_parameters]
            else:
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v
                                         in sorted(named_parameters)}
        else:
            self._is_tensor_instance = False
            self._parameter_names = {v: 'push_pull.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self.backward_passes_per_step = backward_passes_per_step
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._should_sync = True

        # for pipesgd
        self._stale_handles = collections.defaultdict(dict)
        self._stale_push_pull_delay = collections.defaultdict(dict)
        self._stale_push_pull_delay = {self._get_param_name(v): self.backward_passes_per_step
                                       for _, v in sorted(named_parameters)}
        # whether it is the initial optimizer.step()
        self.state['byteps_skipped_init_step'] = False
        # a reference of self._niter stored in states
        self.state['byteps_niter'] = 0
        # store the scale factor from amp if fp16 dynamic scaling is present
        self.state['byteps_stale_scale'] = 1
        # checkpoint the materialized gradient before torch.save() is called.
        # the ckpt grad will be used in three cases: the iteration after checkpoint,
        # the 1-st iteration after resuming training from ckpt, and the first iteration
        # that pipesgd takes effect.
        self.state['byteps_stale_grad'] = {}

        # cache the weight in the previous step
        self.state['byteps_prev_weight'] = {}
        # cache the grad and weight for weight prediction
        self.state['byteps_weight_prediction_local_grad'] = {}
        self.state['byteps_weight_prediction_sync_grad'] = {}
        # cache the optimizer state in the previous step
        self.state['byteps_prev_opt_state'] = {}


        self._prev_weight_update_cnt = {}


        self._partial_stale_layer = partial_stale_layer if staleness == 1 else 0
        if self._partial_stale_layer:
            self._partial_stale_name = self._set_partial_stale_layer_name(partial_stale_layer)
        
        self._step = 0
        self._model = model
        self._logger = logging.getLogger("SAPipe")
        self._logger.debug("hvd size {}, rank {}".format(size(), rank()))
        self._desc = f"rank {rank()}"
        
        self._locks = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()
        
        if size() > 1:
            if self._partial_stale_layer:
                self._register_partial_stale_hooks()
                self._register_forward_hooks()
                # Poll whether the tensor push-pull is finished.
                self._event_queue = queue.Queue()
                #self._poller = threading.Thread(target=self._poll, args=())
                #self._poller.start()
            else:
                self._register_hooks()

        # declare tensors
        for name in sorted(self._parameter_names.values()):
            declare("Gradient."+name, staleness=staleness)
        # We use two loops for load-balancing
        for name in sorted(self._parameter_names.values()):
            declare("Parameter."+name)

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._push_pull_delay:
            self._push_pull_delay[p] = self.backward_passes_per_step
        for p in self._stale_push_pull_delay:
            self._stale_push_pull_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    
    def _register_partial_stale_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    # @yrchen: check if this grad is partial-stale
                    name = self._get_param_name(p)
                    partial_stale = 1 if self._partial_stale_name.get(name) else 0
                    if partial_stale:
                        print(f"Param {name} enables partial staleness!")
                    grad_acc.register_hook(self._make_hook(p, partial_stale))
                    self._grad_accs.append(grad_acc)  

    def _poll(self):
        """Separate thread to poll the completion of the tensor's synchronization 
        from a FIFA event queue. 
        TODO: The advantage of separate polling thread over the hook synchronization is 
        to avoid being blocked by non-prior layers/modules' params.
        TODO: leave out the stale-param
        """
        while True:
            p, handle, ctx = self._event_queue.get()
            if p is None:
                self._logger.debug("poller exists.")
                break
            # check whether the push/pull of this tensor is finished
            # if so, start param updating.
            if handle is not None and poll(handle):
                output = synchronize(handle)
                p.grad.set_(self._compression.decompress(output, ctx))
                self._logger.debug("{} {} finished push-pull".format(self._desc, self._get_param_name(p)))
                self._push_pull_delay[p] = self.backward_passes_per_step
                # So only support SGD, Adam and RMSprop optimizers in torch
                if isinstance(self.__class__, torch.optim.SGD):
                    self._sgd(p)
                else:
                    self._sgd(p)
                    #raise ValueError("Invalid optimizer! Only support SGD, Adam and RMSprop.")
                self._zero_one_grad(p)
                # notify update completion and parameter is ready for forward propagation
                if p in self._locks:
                    self._locks[p].release()
            else:
                self._event_queue.put((p, handle, ctx))

    def _register_forward_hooks(self):
        """@yrchen: Add hook before forward propagation of each layer to block forward computation
        until the gradient synchronization and parameter update are finished. For non-stale layer, do not
        synchronize the comm ops.
        """
        # Recursively find all submodules
        submodules = []
        q = queue.LifoQueue()
        for mod in self._model.children():
            q.put(mod)
        while not q.empty():
            mod = q.get()
            if len(list(mod.children())) == 0:
                submodules.append(mod)
            else:
                for m in mod.children():
                    q.put(m)
        
        def pre_forward_hook(mod, input):
            
            if mod.training is False:
                return
            for p in mod.parameters():
                name = self._get_param_name(p)
                #if p in self._handles:
                #    del self._handles[p]
                if name in self._partial_stale_name:
                    #print(f"param {name} no need for update, continue...")
                    self._sgd(p)
                    self._zero_one_grad(p)
                    continue
                if p not in self._locks:
                    continue
                #with self._locks[p]:
                #    self._logger.debug(f"{self._desc} param {name} is ready.")
                self._logger.debug(f"{self._desc} param {name} is ready.")
                
                self._partial_synchronize(p)
                self._sgd(p)
                self._zero_one_grad(p)
            self._logger.debug(f"{self._desc} starts forward {mod}.")


        def after_forward_hook(mod, input, ret):
            self._logger.debug("{} finished forward {}.".format(self._desc, mod))

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            self._logger.debug("{} registers forward hook on module {}".format(self._desc, mod))
            mod.register_forward_pre_hook(pre_forward_hook)
            mod.register_forward_hook(after_forward_hook)

    def _get_param_name(self, p):
        """Get the name of a parameter."""
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        return name

    def _push_pull_grad_async(self, p):
        name = self._get_param_name(p)
        if self._enable_async:
            # the real handle will be created in step()
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            # pipe-sgd: need to clone the gradient to avoid race condition
            version = 0
            niter = self.state['byteps_niter']
            if self._staleness and niter >= self._pipesgd_warmup_iter:
                tensor_compressed = tensor_compressed.clone()
            handle = byteps_push_pull(tensor_compressed, average=True, name="Gradient."+name,
                                      version=niter, staleness=self._staleness)

           
            if self._partial_stale_layer:
                if self._partial_stale_name.get(name) is None:
                    self._event_queue.put((p, handle, ctx))
        return handle, ctx, name

    
    def _make_hook(self, p, partial_stale=1):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._push_pull_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._push_pull_delay[p] > 0, f"push_pull_delay {self._get_param_name(p)}: {self._push_pull_delay[p]}"
            handle, ctx, name = None, None, None
            self._push_pull_delay[p] -= 1
            
            if self._push_pull_delay[p] == 0:
                if self._partial_stale_layer:
                    #self._locks[p].acquire()
                    pass
                handle, ctx, name = self._push_pull_grad_async(p)
            # FIXME: ``name`` may be undefined
            self._handles[p] = (handle, ctx, name)

        def stale_hook(*ignore):
            name = self._get_param_name(p)
            niter = self.state['byteps_niter']
            if name in self._stale_handles[niter] and self._stale_handles[niter][name][0] is not None:
                if self._stale_push_pull_delay[name] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            handle, ctx = None, None
            self._stale_push_pull_delay[name] -= 1
            if self._stale_push_pull_delay[name] == 0:
                if self.state['byteps_niter'] >= self._pipesgd_warmup_iter and self._pipesgd_weight_prediction is not None:
                    # pipe-sgd with weight prediction via latest sync gradient and local gradient replaced: 
                    # must be done before the local gradient cache is updated
                    if "local_prev_sync" in self._pipesgd_weight_prediction and name in self.state['byteps_weight_prediction_local_grad']:
                        if name not in self.state['byteps_weight_prediction_sync_grad']:
                            self.state['byteps_weight_prediction_sync_grad'][name] = self.state['byteps_weight_prediction_local_grad'][name].clone()
                        else:
                            self.state['byteps_weight_prediction_sync_grad'][name].copy_(self.state['byteps_weight_prediction_local_grad'][name])
                        # sync_grad = - prev_local_grad/n
                        self.state['byteps_weight_prediction_sync_grad'][name][:] *= (-1.0 / size())
                    # pipe-sgd with weight prediction via local gradient: cache the local gradient for weight prediction
                    if "local" in self._pipesgd_weight_prediction:
                        if name not in self.state['byteps_weight_prediction_local_grad']:
                            self.state['byteps_weight_prediction_local_grad'][name] = p.grad.clone()
                        else:
                            self.state['byteps_weight_prediction_local_grad'][name].copy_(p.grad)

                handle, ctx, name = self._push_pull_grad_async(p)
                self._stale_handles[niter][name] = (p, handle, ctx)

        # @yrchen: enable partial-stale hooks
        if self._staleness == 0:
            return hook
        else:
            return hook if partial_stale == 0 else stale_hook

        #return hook if self._staleness == 0 else stale_hook

    def synchronize(self):
        """Synchronizes the asynchronous pushpull(allreduce) operations for all
        gradients until they are completed.
        """
        if self._staleness:
            self._stale_synchronize()
        else:
            self._synchronize()


    def _synchronize(self):
        """Synchronize the pushpull operations"""
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            if type(p.grad) == type(None):
                continue
            handle, ctx, name = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx, name)

        for p, value in self._handles.items():
            handle, ctx, name = value
            if handle is None:
                handle, ctx, _ = self._push_pull_grad_async(p)
                self._handles[p] = (handle, ctx, name)
        for p, (handle, _, name) in self._handles.items():
            output = synchronize(handle)
            self._push_pull_delay[p] = self.backward_passes_per_step
            if not self._enable_async:
                tmp = self._compression.decompress(output, ctx)
                if self._compression == Compression.none:
                    p.grad.set_(tmp)
                else:
                    p.grad.copy_(tmp)
        self._handles.clear()

    def _stale_synchronize(self):
        """Synchronize the pushpull operations when pipesgd is enabled"""
        has_amp = hasattr(self, "_amp_stash")
        niter = self.state['byteps_niter']
        assert niter >= 0, niter
        loss_scale = 1

        if has_amp:
            import apex
            loss_scalers = apex.amp._amp_state.loss_scalers
            assert len(loss_scalers) == 1, f'Multiple amp loss is not supported: {loss_scalers}'
            loss_scale = loss_scalers[0].loss_scale()
        # if the loss scale increases at the current iteration
        # amp will rescale it back after synchronize(). so we need
        # to adjust the gradient from the previous step accordingly
        if niter > self._pipesgd_warmup_iter:
            prev_loss_scale = self.state['byteps_stale_scale']
        else:
            prev_loss_scale = loss_scale
        grad_ratio = loss_scale / prev_loss_scale

        # materialzed grad tensors are not available. obtain them from handles
        stale_grad_state = self.state['byteps_stale_grad']
        if not stale_grad_state:
            if niter <= self._pipesgd_warmup_iter:
                if niter == self._pipesgd_warmup_iter:
                    print(f'BytePS pipeSGD: started pipeline at iter {niter}', flush=True)
                for name, (p, handle, ctx) in self._stale_handles[niter].items():
                    assert handle is not None, name
                    assert not p.grad.is_sparse, "sparse gradient is not supported"
                    output = synchronize(handle)
                    tmp = self._compression.decompress(output, ctx)
                    # sync SGD duration warmup
                    if niter < self._pipesgd_warmup_iter:
                        p.grad.data = tmp
                    else:
                        stale_grad_state[name] = tmp
                        p.grad.copy_(tmp)
            else:
                for name, (p, handle, ctx) in self._stale_handles[niter-1].items():
                    assert handle is not None
                    assert not p.grad.is_sparse, "sparse gradient is not supported"
                    output = synchronize(handle)
                    prev_grad = self._compression.decompress(output, ctx)
                    with torch.no_grad():
                        # if the loss scale increases at the current iteration
                        # amp will rescale it back after synchronize(). so we need
                        # to adjust the gradient from the previous step accordingly
                        if grad_ratio != 1.0:
                            prev_grad.mul_(grad_ratio)
                        p.grad.copy_(prev_grad)
            if (niter - 1) in self._stale_handles:
                del self._stale_handles[niter - 1]
        else:
            # grad tensors alread materialized
            for p in self._requires_update:
                assert not p.grad.is_sparse, "sparse gradient is not supported"
                name = self._get_param_name(p)
                if name in stale_grad_state:
                    prev_grad = stale_grad_state[name]
                    with torch.no_grad():
                        if grad_ratio != 1.0:
                            prev_grad.mul_(grad_ratio)
                        p.grad.copy_(prev_grad)
            self.state['byteps_stale_grad'] = {}
        
        if niter > self._pipesgd_warmup_iter:
            # delay mitigation if pipesgd is active, first step skipped
            if self._pipesgd_weight_prediction is None and self._pipesgd_dc_lambda > 0:
                # delay compensation, if not using weight prediction, and pipesgd is active
                prev_weight_dict = self.state['byteps_prev_weight']
                for param_group in self.param_groups:
                    for p in param_group['params']:
                        if p.requires_grad:
                            name = self._get_param_name(p)
                            if name not in prev_weight_dict:
                                # in the first iteration where pipesgd is active, gradient has no delay, no need to use dc
                                prev_weight_dict[name] = [p.data.clone().detach()]
                            else:
                                self.delay_compensation(p.grad, p.data, prev_weight_dict[name][-1])
                                with torch.no_grad():
                                    # update cache
                                    prev_weight_dict[name][-1].copy_(p.data)
        
        if niter >= self._pipesgd_warmup_iter and self._pipesgd_weight_prediction is not None:
            # delay mitigation if pipesgd is active
            if "prev_sync" in self._pipesgd_weight_prediction:
                # pipe-sgd with weight prediction via latest sync gradient and local gradient replaced: 
                for group in self.param_groups:
                    for p in group['params']:
                        if p.requires_grad:
                            name = self._get_param_name(p)
                            prev_sync_grad = p.grad
                            if "local" in self._pipesgd_weight_prediction:
                                if name in self.state['byteps_weight_prediction_sync_grad']:
                                    # sync_grad = - prev_local_grad/n + prev_sync_grad 
                                    self.state['byteps_weight_prediction_sync_grad'][name][:] += prev_sync_grad
                                else:
                                    # sync_grad = prev_sync_grad 
                                    self.state['byteps_weight_prediction_sync_grad'][name] = prev_sync_grad.clone()
                            else:
                                # sync_grad = prev_sync_grad 
                                if name in self.state['byteps_weight_prediction_sync_grad']:
                                    self.state['byteps_weight_prediction_sync_grad'][name].copy_(prev_sync_grad)
                                else:
                                    self.state['byteps_weight_prediction_sync_grad'][name] = prev_sync_grad.clone()



        # update states
        for name in self._stale_push_pull_delay:
            self._stale_push_pull_delay[name] = self.backward_passes_per_step
        self.state['byteps_stale_scale'] = loss_scale
        self.state['byteps_niter'] += 1

    def prepare_stale_states(self):
        """
        This API is used to save _stale_grad and _stale_scale when both checkpointing
        and PipeSGD are enabled. The ckpt _stale_grad and _stale_scale will be used for
        update when resuming training from ckpt. Please Note: User must call this API intentionally
        before torch.save().
        """
        stale_grad_states = {}
        niter = self.state['byteps_niter']
        for name, (p, handle, ctx) in self._stale_handles[niter-1].items():
            assert handle is not None
            assert not p.grad.is_sparse, p
            output = synchronize(handle)
            prev_grad = self._compression.decompress(output, ctx)
            stale_grad_states[name] = prev_grad
        self.state['byteps_stale_grad'] = stale_grad_states
        del self._stale_handles[niter-1]
        for name in self._stale_push_pull_delay:
            self._stale_push_pull_delay[name] = self.backward_passes_per_step

    @contextmanager
    def skip_synchronize(self):
        if self._enable_async:
            raise AssertionError("skip_synchronize cannot be used in async training")
        self._should_sync = False
        try:
            yield
        finally:
            self._should_sync = True

    def step(self, closure=None):
        if self._enable_async:
            old_weight_map = {}
            # store the weights before update
            for p, _ in self._handles.items():
                old_weight_map[p] = p.data.clone().detach()
            # update
            loss = super(self.__class__, self).step(closure)

            for p, (h, _) in self._handles.items():
                # get the diff for each weight (in-place)
                p.data.sub_(old_weight_map.get(p))
                if h is None:
                    # create the handler now
                    if self._is_tensor_instance:
                        name = self._parameter_names.get(p.__hash__())
                    else:
                        name = self._parameter_names.get(p)
                    handle = byteps_push_pull(p, average=False, name="AsyncParam."+name)
                    _, ctx = self._compression.compress(p)
                    self._handles[p] = (handle, ctx)

            self.synchronize()
            return loss
        else:
            # skip sync if calling skip_synchronize
            if self._should_sync:
                self.synchronize()
            niter = self.state['byteps_niter']
            # synchronize() already incremented niter by 1
            pipesgd_active = self._staleness and niter > self._pipesgd_warmup_iter
            if not pipesgd_active:
                return super(self.__class__, self).step(closure)
            else:
                if not self.state['byteps_skipped_init_step']:
                    self.state['byteps_skipped_init_step'] = True
                else:
                    if self._pipesgd_weight_prediction is not None:
                        tic = time.time()
                        self.restore_weight()
                        if rank() == -1:
                            print(f"restore weight time: {time.time() - tic:.4f}")
                    # @yrchen: only apply update when partial-staleness is disabled
                    if self._partial_stale_layer == 0:
                        return super(self.__class__, self).step(closure)
                    else:
                        # call forward hook to apply update
                        if self._step == 0:
                            self._step += 1
                            self._synchronize()
                            return super(self.__class__, self).step(closure)
                        else:
                            loss = None
                            if closure is not None:
                                loss = closure()
                            self._step += 1
                            return loss            
    
    def partial_step(self, closure=None):
        """Override the default step function.
        """
        self._logger.debug(f"{self._desc} calls step function {self._step}.")

        # Step 0 is called for parameter initialization after parameter broadcast
        if size() > 1 and self._step > 0:
            # TODO: finish the partial synchronize
            self._partial_synchronize()
            # TODO: if this is the final training step, wait for the completion of all tensors
            loss = None
            if closure is not None:
                loss = closure()
            self._step += 1
            return loss
        else:
            # Optimizer.step() will be triggered when user calls byteps.broadcast_optimizer_sate()
            self._step += 1
            return super(self.__class__, self).step(closure)
            

    def delay_compensation(self, grad, cur_weight, prev_weight):
        with torch.no_grad():
            if self._pipesgd_dc_outerprod:
                if self._pipesgd_dc_adaptive:
                    # dc, outerprod, adaptive
                    grad[:] += (self._pipesgd_dc_lambda * ((torch.sum(grad * (cur_weight - prev_weight)) \
                        / (torch.sqrt(torch.sum(grad * grad) + self._pipesgd_dc_adaptive_eps))) * grad))
                else:
                    # dc, outerprod
                    grad[:] += self._pipesgd_dc_lambda * (torch.sum(grad * (cur_weight - prev_weight)) * grad)
            else:
                if self._pipesgd_dc_adaptive:
                    # dc, adaptive
                    grad_square = grad * grad
                    grad[:] += (self._pipesgd_dc_lambda * (grad_square * (cur_weight - prev_weight) \
                        / torch.sqrt(grad_square + self._pipesgd_dc_adaptive_eps)))
                else:
                    # dc
                    grad[:] += self._pipesgd_dc_lambda * (grad * grad * (cur_weight - prev_weight))
    
    def cache_weight(self):
        prev_weight_dict = self.state['byteps_prev_weight']
        src_tensor = []
        dst_tensor = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    name = self._get_param_name(p)
                    if name not in prev_weight_dict:
                        prev_weight_dict[name] = []
                        self._prev_weight_update_cnt[name] = -1
                    if len(prev_weight_dict[name]) < self._pipesgd_num_cached_weight:
                        prev_weight_dict[name].append(p.data.clone().detach())
                    else:
                        # discard the oldest weight
                        #prev_weight_dict[name].append(prev_weight_dict[name].pop(0))
                        # cache the latest weight
                        prev_weight_dict[name][-1].copy_(p.data)
                        self._prev_weight_update_cnt[name] += 1
                        update_idx = (self._prev_weight_update_cnt[name]) % self._pipesgd_num_cached_weight

        if len(dst_tensor) > 0 and False:
            torch._foreach_add_(dst_tensor, src_tensor)
            #pass
            #copy_tensor_(dst_tensor, src_tensor)
            return [], []
        else:
            pass
            #print(f"No cache tensor!")
        return dst_tensor, src_tensor
    
    def restore_weight(self):
        prev_weight_dict = self.state['byteps_prev_weight']
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    name = self._get_param_name(p)
                    p.data.copy_(prev_weight_dict[name][-1])

    def restore_weight_v2(self):
        prev_weight_dict = self.state['byteps_prev_weight']
        for group in self.param_groups:
            grads = []
            params_wigh_grad = []
            for p in group['params']:
                if p.requires_grad:
                    name = self._get_param_name(p)
                    update_idx = (self._prev_weight_update_cnt[name]) % self._pipesgd_num_cached_weight
                    grads.append(prev_weight_dict[name][update_idx])
                    params_wigh_grad.append(p.data)
            
            torch._foreach_add_(params_wigh_grad, grads)
 
    def cache_optimizer_state(self):
        src_tensor = []
        dst_tensor = []
        prev_opt_state_dict = self.state['byteps_prev_opt_state']
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    name = self._get_param_name(p)
                    state = self.state[p]
                    if name not in prev_opt_state_dict:
                        prev_opt_state_dict[name] = dict()
                    prev_opt_state = prev_opt_state_dict[name]
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            if k not in prev_opt_state:
                                prev_opt_state[k] = v.clone()
                            else:
                                prev_opt_state[k].copy_(v)
                                #src_tensor.append(v)
                                #dst_tensor.append(prev_opt_state[k])
                        else:
                            prev_opt_state[k] = state[k]
           
        if len(dst_tensor) > 0:
            torch._foreach_add_(dst_tensor, src_tensor)

    def restore_optimizer_state(self):
        src_tensor = []
        dst_tensor = []
        prev_opt_state_dict = self.state['byteps_prev_opt_state']
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    name = self._get_param_name(p)
                    state = self.state[p]
                    prev_opt_state = prev_opt_state_dict[name]
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            v.copy_(prev_opt_state[k])
                        else:
                            state[k] = prev_opt_state[k]
        # FIXME 
        if len(dst_tensor) > 0:
            torch._foreach_add_(dst_tensor, src_tensor)

    def shadow_step(self):
        if self._pipesgd_weight_prediction is not None and self.state['byteps_skipped_init_step']:
            # cache the weight and optimizer state
            step_tic = time.time()
            self.cache_weight()
            cache_tac = time.time()
            self.cache_optimizer_state()
            op_state_tac = time.time()
            # use the estimated gradient to execute 1 optimizer step ahead
            src_tensor = []
            dst_tensor = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        name = self._get_param_name(p)
                        # dc
                        if self._pipesgd_weight_prediction.endswith("_dc") \
                            and name in self.state['byteps_weight_prediction_sync_grad'] \
                            and len(self.state['byteps_prev_weight'][name]) >= 2:
                            self.delay_compensation(self.state['byteps_weight_prediction_sync_grad'][name], \
                                                    self.state['byteps_prev_weight'][name][-1], \
                                                    self.state['byteps_prev_weight'][name][-2])
                        if "local" in self._pipesgd_weight_prediction:
                            # weight prediction using local gradient
                            p.grad.copy_(self.state['byteps_weight_prediction_local_grad'][name])

                            if "prev_sync" in self._pipesgd_weight_prediction \
                                and name in self.state['byteps_weight_prediction_sync_grad'] \
                                and self.state['byteps_niter'] > self._pipesgd_warmup_iter + 1:
                                # grad = cur_local_grad/n - prev_local_grad/n + prev_avg_grad 
                                p.grad[:] /= size()
                                p.grad[:] += self.state['byteps_weight_prediction_sync_grad'][name]
                        else:
                            if name in self.state['byteps_weight_prediction_sync_grad']:
                                # weight prediction using prev_sync_grad
                                p.grad.copy_(self.state['byteps_weight_prediction_sync_grad'][name])
                            else:
                                # weight prediction using local gradient temporarily, if byteps_weight_prediction_sync_grad is not ready
                                p.grad.copy_(self.state['byteps_weight_prediction_local_grad'][name])
            copy_tac = time.time()
            ret = super(self.__class__, self).step()
            step_tac = time.time()
            self.restore_optimizer_state()
            restore_tac = time.time()
            return ret

    
    def _set_partial_stale_layer_name(self, partial_stale_layer=2):
        print(f"parameter_names: {self._parameter_names}")
        param_names = self._parameter_names
        partial_stale_name = {}
        for i in range(partial_stale_layer):
            # for VGG16 model
            layer_index = "." +  str(i) + "."
            for param_group in self.param_groups:
                for p in param_group['params']:
                    if p.requires_grad:
                        name = self._get_param_name(p)
                        if layer_index in name:
                            partial_stale_name[name] = 1
        print(f"Get partial stale layer name {partial_stale_layer}: {partial_stale_name}")
        return partial_stale_name

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as torch accumulates gradients by default.
        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    def _partial_synchronize(self, p):
        """synchronize function for partial staleness mode
        """
        # this may have been done in grad hooks
        if self._handles.get(p) is None:
            print(f"no handles for param {self._get_param_name(p)}")
            return None
        value = self._handles.pop(p)
        handle, ctx, name = value
        print(f"synchronizing for param {self._get_param_name(p)}")
        output = synchronize(handle)
        #if p in self._locks:
        #    self._locks[p].release()
        self._push_pull_delay[p] = self.backward_passes_per_step
        print(f"Reset param {self._get_param_name(p)} delay to 1.")
        if not self._enable_async:
            tmp = output
            if self._compression == Compression.none:
                p.grad.set_(tmp)
            else:
                p.grad.copy_(tmp)

        #del self._handles[p]

    @torch.no_grad()
    def _sgd(self, p):
        """Perform a single update step using SGD optimizer on a parameter.
        Argumetns:
            p: the parameter to be updated.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            #TODO
            for gp in group['params']:
                #if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                if self._get_param_name(p) != self._get_param_name(gp) or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._get_param_name(p)))
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                #p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p, alpha=-group['lr'])
                break


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1, staleness=0,
                         pipesgd_warmup_iter=0,
                         pipesgd_dc_lambda=0,
                         pipesgd_dc_outerprod=False,
                         pipesgd_dc_adaptive=False,
                         pipesgd_dc_adaptive_eps=1e-7,
                         pipesgd_weight_prediction=None,
                         partial_stale_layer=0, model=None
):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an push_pull to
    average gradient values before applying gradients to model weights.
    push_pull operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all push_pull operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces push_pull operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:

    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          push_pull operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
        staleness: Number of controlled gradients staleness if pipelined SGD is enabled. 
                   This allows optimizer using stale gradients to update parameters. Defaults 
                   to not using pipelined SGD, i.e., staleness=0. If set to 1, the parameter
                   update is delayed by 1 step. Reference: https://arxiv.org/abs/1811.03619
        pipesgd_warmup_iter: Number of warmup steps for pipesgd, during which pipesgd staleness
                   is fixed at 0.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression, backward_passes_per_step,
               staleness=staleness, pipesgd_warmup_iter=pipesgd_warmup_iter,
               pipesgd_dc_lambda=pipesgd_dc_lambda, pipesgd_dc_outerprod=pipesgd_dc_outerprod,
               pipesgd_dc_adaptive=pipesgd_dc_adaptive, pipesgd_dc_adaptive_eps=pipesgd_dc_adaptive_eps,
               pipesgd_weight_prediction=pipesgd_weight_prediction,
               partial_stale_layer=partial_stale_layer, model=model
)


def broadcast_parameters(params, root_rank, prefix="Parameter."):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.
    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run synchronous broadcasts.
    for name, p in params:
        # Broadcast is implemented as push + pull in BytePS
        # To make it a real broadcast, we set the non-root tensors all 0.
        if rank() != root_rank:
            p.fill_(0)
        # Remember to disable averaging because we are doing broadcast
        if name:
            handle = byteps_push_pull(p, average=False, name=prefix+name)
        else:
            handle = byteps_push_pull(p, average=False)
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank, prefix="Parameter."):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces push_pull on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    scalars = {}
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we place the scalars into a single dict,
    # then pickle and broadcast with broadcast_object (under the assumption
    # that there are not many scalars, and so the overhead of pickling will
    # be relatively low). Because broadcast_object is performed out-of-place,
    # we then use a callback to assign the new value to the correct element
    # of the optimizer state.
    def _create_state_callback(pid, name):
        def _assign_state(v):
            state_dict['state'][pid][name] = v
        return _assign_state

    def _create_option_callback(index, option_key):
        def _assign_option(v):
            optimizer.param_groups[index][option_key] = v
        return _assign_option

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be broadcast separately
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value]).cuda()
            scalars[key] = option_value
            callbacks[key] = _create_option_callback(index, option_key)

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            if pid not in state_dict['state']:
                # The param has not set requires_grad, so skip broadcast
                continue

            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if torch.is_tensor(p):
                    # Tensor -> use broadcast_parameters
                    params.append((key, p))
                else:
                    # Scalar -> use broadcast_object
                    scalars[key] = p
                    callbacks[key] = _create_state_callback(pid, name)

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank, prefix)

    # Broadcast and cleanup for non-tensor parameters
    scalars = broadcast_object(scalars, root_rank)
    for key, p in scalars.items():
        callbacks[key](p)

def broadcast_object(obj, root_rank=0, name=None):
    """
    Serializes and broadcasts an object from root rank to all other processes.
    Typical usage is to broadcast the `optimizer.state_dict()`, for example:

    .. code-block:: python

        state_dict = broadcast_object(optimizer.state_dict(), 0)
        if bps.rank() > 0:
            optimizer.load_state_dict(state_dict)

    Arguments:
        obj: An object capable of being serialized without losing any context.
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        name: Optional name to use during broadcast, will default to the class
              type.
    Returns:
        The object that was broadcast from the `root_rank`.
    """
    import cloudpickle

    if name is None:
        name = type(obj).__name__

    if rank() == root_rank:
        b = io.BytesIO()
        cloudpickle.dump(obj, b)
        t = torch.ByteTensor(bytearray(b.getvalue()))
        sz = torch.IntTensor([t.shape[0]])
        broadcast_parameters([(name + '.sz', sz)], root_rank, prefix="Size.")
    else:
        sz = torch.IntTensor([0])
        broadcast_parameters([(name + '.sz', sz)], root_rank, prefix="Size.")
        t = torch.ByteTensor(sz.tolist()[0])

    broadcast_parameters([(name + '.t', t)], root_rank, prefix="Parameter.")

    if rank() != root_rank:
        buf = io.BytesIO(t.numpy().tobytes())
        obj = cloudpickle.load(buf)

    return obj

