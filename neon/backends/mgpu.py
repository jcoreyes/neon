# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Neon backend wrapper for the NervanaGPU library. Most functions are thin
wrappers around functions from the NervanaGPU class, the GPUTensor is taken
directly from NervanaGPU as well.
NervanaGPU is available at `<https://github.com/NervanaSystems/nervanagpu>`
"""
import logging

from neon.backends.backend import Backend
from neon.backends.gpu import GPU
from nervanagpu import NervanaGPU, GPUTensor
from neon.diagnostics.timing_decorators import FlopsDecorator
import pycuda.driver as drv
import numpy as np

logger = logging.getLogger(__name__)


def replicate(method):
    def decorate(cls):
        def func(self, *args, **kwargs):
            tsrlist = []
            for idx, ctx in enumerate(getattr(self, 'ctxs')):
                ctx.push()
                myargs = [a._tensorlist[idx] if isinstance(
                    a, MGPUTensor) else a for a in args]
                mykwargs = {k: v._tensorlist[idx] if isinstance(
                    v, MGPUTensor) else v for k, v in kwargs.iteritems()}
                tsrlist.append(
                    getattr(super(cls, self), method)(*myargs, **mykwargs))
                ctx.pop()
            return MGPUTensor(tsrlist) if tsrlist[0] is not None else None
        setattr(cls, method, func)
        return cls
    return decorate


def passthru(method):
    def decorate(cls):
        def func(self, *args, **kwargs):
            tsrlist = []
            for idx, (tsr, ctx) in enumerate(zip(getattr(self, '_tensorlist'),
                                                 getattr(self, 'ctxs'))):
                ctx.push()
                myargs = [a._tensorlist[idx] if isinstance(
                    a, MGPUTensor) else a for a in args]
                mykwargs = {k: v._tensorlist[idx] if isinstance(
                    v, MGPUTensor) else v for k, v in kwargs.iteritems()}
                tsrlist.append(getattr(tsr, method)(*myargs, **mykwargs))
                ctx.pop()
            return MGPUTensor(tsrlist) if tsrlist[0] is not None else None
        setattr(cls, method, func)
        return cls
    return decorate


@passthru('_assign')
@passthru('fill')
@passthru('reshape')
@passthru('__getitem__')
class MGPUTensor(object):
    ctxs = None
    num_dev = 0

    def __init__(self, tensorlist, ptype='fragment'):
        self._tensorlist = tensorlist
        self.ptype = ptype

    @property
    def shape(self):
        return self._tensorlist[0].shape

    @property
    def dtype(self):
        return self._tensorlist[0].dtype

    @property
    def size(self):
        return self._tensorlist[0].size

    @property
    def ptr(self):
        return self._tensorlist[0].gpudata.__int__()

    def __setitem__(self, index, value):
        if self.ctxs is None:
            raise ValueError("Contexts not defined")
        for idx, (tsr, ctx) in enumerate(zip(getattr(self, '_tensorlist'),
                                             getattr(self, 'ctxs'))):
            ctx.push()
            if isinstance(value, MGPUTensor):
                tsr.__setitem__(index, value._tensorlist[idx])
            else:
                tsr.__setitem__(index, value)
            ctx.pop()

    def asnumpyarray(self):
        if self.ptype == 'replica':
            self.ctxs[0].push()
            rval = self._tensorlist[0].get()
            self.ctxs[0].pop()
            return rval
        else:
            rval = []
            for i, subtensor in enumerate(self._tensorlist):
                self.ctxs[i].push()
                npv = subtensor.get()
                rval.append(npv)
                self.ctxs[i].pop()
            return np.hstack(rval)

    @property
    def T(self):
        """
        return a transposed view
        """
        tsrlist = []
        for tsr in self._tensorlist:
            tsrlist.append(GPUTensor(backend=tsr.backend,
                                     shape=tsr.shape[::-1], dtype=tsr.dtype,
                                     allocator=tsr.allocator, base=tsr,
                                     gpudata=tsr.gpudata,
                                     strides=tsr.strides[::-1],
                                     is_trans=(not tsr.is_trans),
                                     name=tsr.name, rounding=tsr.rounding))
        return self.__class__(tsrlist)


@replicate('fprop_fc')
@replicate('bprop_fc')
@replicate('update_fc')
@replicate('fprop_conv')
@replicate('convolution')
@replicate('bprop_conv')
@replicate('update_conv')
@replicate('fprop_pool')
@replicate('bprop_pool')
@replicate('logistic')
@replicate('rectlin')
@replicate('sum')
@replicate('mean')
@replicate('min')
@replicate('max')
@replicate('variance')
@replicate('fabs')
@replicate('sqrt')
@replicate('zeros')
@replicate('ones')
@replicate('empty')
@replicate('array')
@replicate('add')
@replicate('subtract')
@replicate('multiply')
@replicate('divide')
@replicate('greater')
@replicate('equal')
@replicate('not_equal')
@replicate('clip')
@replicate('log')
@replicate('tanh')
@replicate('argmax')
@replicate('softmax')
@replicate('softmax_gradient')
@replicate('make_binary_mask')
@replicate('gdm_compound')
@replicate('gdmwd_compound')
@replicate('ada_update')
@replicate('fprop_bn_compound')
@replicate('bprop_bn_compound')
class MGPU(GPU):
    default_dtype = np.float32

    def __init__(self, rng_seed, stochastic_round=False, device_id=0,
                 num_dev=2):
        import pycuda.driver as drv
        drv.init()
        self.ctxs = []
        self.num_dev = num_dev
        self.dev_list = range(num_dev)
        import atexit
        for i in self.dev_list:
            self.ctxs.append(drv.Device(i).make_context())
            self.ctxs[i].pop()

        self.ctxs[0].push()
        self.strms = [drv.Stream() for i in self.dev_list]
        atexit.register(self.ctxs[0].pop)
        MGPUTensor.ctxs = self.ctxs
        MGPUTensor.num_dev = num_dev

        self.ng = NervanaGPU(stochastic_round=stochastic_round)
        logger.info("Initialized NervanaGPU with stochastic_round=%s",
                    stochastic_round)
        self.rng_seed = rng_seed
        self.rng_init()

        # Setup the pairwise contexts
        for i in range(len(self.dev_list)):
            d1 = self.dev_list[i]
            for j in range(i+1, len(self.dev_list)):
                d2 = self.dev_list[j]
                if drv.Device(d1).can_access_peer(drv.Device(d2)):
                    self.ctxs[d1].enable_peer_access(self.ctxs[d2])
                else:
                    print('Cannot enable peer access between '
                          '{:d} and {:d}'.format(d1, d2))

        self.device_id = device_id if device_id is not None else 0

    def uniform(self, low=0.0, high=1.0, shape=1, dtype=default_dtype,
                name=None, allocator=drv.mem_alloc):
        """
        generate numpy random number and convert to a GPUTensor.
        If called with dtype=None it will probably explode
        """
        ary = np.random.uniform(low, high, shape)
        return self.array(ary, dtype)

    def normal(self, loc=0.0, scale=1.0, size=1, dtype=default_dtype,
               name=None, allocator=drv.mem_alloc):
        """
        Gaussian/Normal random number sample generation
        """
        ary = np.random.normal(loc, scale, size)
        return self.array(ary, dtype)

    def synchronize(self):
        for s in self.strms:
            s.synchronize()

    def allocate_fragment(self, shape, dtype=default_dtype):
        return self.empty((shape[0], shape[1]/self.num_dev), dtype)

    def scatter(self, hbuf, dbuf, async=False):
        '''
        scatters the array data in hbuf to the mgpu tensor
        assumes that dbuf is a M x N and hbuf is M x (Nxk) where k is the
        number of replicas
        also assumes that dtype of hbuf and dbuf are the same
        '''
        assert hbuf.size == dbuf.size * dbuf.num_dev
        assert isinstance(dbuf, MGPUTensor)
        assert hbuf.dtype == dbuf.dtype
        ndata = dbuf.size
        for idx, dest_buf in enumerate(dbuf._tensorlist):
            src = hbuf.reshape((hbuf.size))[(idx * ndata):((idx+1) * ndata)]
            self.ctxs[idx].push()
            strm = self.strms[idx] if async else None
            drv.memcpy_htod_async(dest_buf.ptr, src, strm)
            self.ctxs[idx].pop()
        if async:
            self.synchronize()

    def reduce_tensor(self, ary, async=False):
        '''
        This is the case for the scalar tensor
        '''
        assert ary.size == 1
        nbytes = ary.dtype.itemsize
        result = np.zeros((self.num_dev, 1), ary.dtype)
        for i, src_buf in enumerate(ary._tensorlist):
            self.ctxs[i].push()
            strm = self.strms[i] if async else None
            drv.memcpy_dtoh_async(result[i], src_buf.ptr, strm)
            self.ctxs[i].pop()
        if async:
            self.synchronize()
        return result.sum()

    def reduce(self, ary, ubuf, async=False):
        numrep = self.num_dev
        totsz = ary.size
        subsz = (totsz + numrep - 1) / numrep
        dsz = ary.dtype.itemsize
        assert ubuf.size >= totsz

        for dest_idx, dest_buf in enumerate(ubuf._tensorlist):
            for src_idx, src_buf in enumerate(ary._tensorlist):
                dest = dest_buf.ptr + src_idx * subsz * dsz
                src = src_buf.ptr + dest_idx * subsz * dsz
                nbytes = dsz * subsz
                strm = self.strms[src_idx] if async else None
                myargs = [dest, src, nbytes]
                if src_idx == dest_idx:
                    cpfunc = drv.memcpy_dtod_async
                else:
                    cpfunc = drv.memcpy_peer_async
                    myargs.extend([self.ctxs[dest_idx], self.ctxs[src_idx]])
                if async:
                    myargs.append(self.strms[src_idx])
                self.ctxs[src_idx].push()
                cpfunc(*myargs)
                self.ctxs[src_idx].pop()

        if async:
            self.synchronize()

        for src_idx, (sbuf, dbuf) in enumerate(zip(ary._tensorlist,
                                                   ubuf._tensorlist)):
            if async:
                self.ng.stream = self.strms[src_idx]
            start = src_idx * subsz
            end = start + subsz
            sbuf = sbuf.reshape((totsz, 1))
            ubtmp = dbuf.reshape((numrep, dbuf.size/numrep))
            self.ng.sum(ubtmp, axis=0, out=sbuf[start:end])

        if async:
            self.ng.stream = None
            self.synchronize()

        for dest_idx, dest_buf in enumerate(ary._tensorlist):
            for src_idx, src_buf in enumerate(ary._tensorlist):
                if src_idx == dest_idx:
                    continue
                dest = dest_buf.ptr + src_idx * subsz * dsz
                src = src_buf.ptr + src_idx * subsz * dsz
                nbytes = dsz * subsz
                strm = self.strms[src_idx] if async else None
                drv.memcpy_peer_async(dest, src, nbytes,
                                      self.ctxs[dest_idx],
                                      self.ctxs[src_idx], strm)

        if async:
            self.synchronize()

        return ary
