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
import logging
import os
import numpy as np
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class NoPar(object):

    def __init__(self):
        self.backend = None
        self.device_id = None

    def init_model(self, model, backend):
        backend.actual_batch_size = model.batch_size

    def scatter(self, src, dest):
        dest.copy_from(src)

    def associate(self, backend):
        backend.par = self
        opt_param(self, ['par_mode'], None)
        opt_param(self, ['num_dev'], 1)

        self.backend = backend

    def distribute(self, batchdata):
        return self.backend.array(batchdata)

    def reduce_tensor(self, tensor):
        return tensor.asnumpyarray()

    def rank(self):
        return 0

    def size(self):
        return 1

    def allocate_fragment(self, buf_shape, dtype=None):
        return self.backend.empty(buf_shape, dtype=dtype)

    def reduce(self, ary, ubuf):
        pass

    def all_reduce(self, tensor):
        pass

    def is_distributed(self):
        return False


class DPar(NoPar):

    def init_model(self, model, backend):
        backend.actual_batch_size = model.batch_size

    def associate(self, backend):
        backend.par = self
        opt_param(self, ['par_mode'], None)
        opt_param(self, ['num_dev'], 1)

        self.backend = backend

    def allocate_fragment(self, buf_shape, dtype=None):
        return self.backend.empty(buf_shape, dtype=dtype)

    def is_distributed(self):
        return True


class BasePar(object):

    def __init__(self):
        self.backend = None
        self.device_id = None
        try:
            from mpi4py import MPI
            self.mpi = MPI
            self.comm = self.mpi.COMM_WORLD
            self.mpi_size = self.comm.size
            self.mpi_rank = self.comm.rank
        except ImportError:
            raise RuntimeError(
                "mpi4py not found, can't run in datapar or modelpar")

        try:
            # Determine local rank (assumes OpenMPI).
            self.mpi_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            self.mpi_local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        except:
            raise RuntimeError(
                "OpenMPI variable OMPI_COMM_WORLD_LOCAL_RANK or "
                "OMPI_COMM_WORLD_LOCAL_SIZE not found.\n"
                "Are you using: mpirun -n <#procs> neon <example.yaml>?")
        self.device_id = self.mpi_local_rank

    def init_model(self, model, backend):
        # save the original batch_size value that is specified in
        # the configuration file
        backend.actual_batch_size = model.batch_size

    def associate(self, backend):
        backend.par = self
        self.backend = backend

    def distribute(self, batchdata):
        raise NotImplementedError()

    def reduce_tensor(self, tensor):
        raise NotImplementedError()

    def distributable(self, layer):
        if hasattr(layer, 'distributable'):
            return layer.distributable
        return False

    def rank(self):
        return self.mpi_rank

    def size(self):
        return self.mpi_size

    def allocate_fragment(self, buf_shape, dtype=None):
        raise NotImplementedError()

    def dtype_to_mpi(self, t):
        if hasattr(self.mpi, '_typedict'):
            mpi_type = self.mpi._typedict[np.dtype(t).char]
        elif hasattr(self.mpi, '__TypeDict__'):
            mpi_type = self.mpi.__TypeDict__[np.dtype(t).char]
        else:
            raise ValueError('cannot convert type')
        return mpi_type

    def all_reduce(self, tensor):
        """
        This is for doing reduction
        """
        mdtype = self.dtype_to_mpi(tensor.dtype)
        if self.backend.__module__ == 'neon.backends.gpu':
            tmpbuf = tensor.asnumpyarray()
            self.comm.Allreduce(self.mpi.IN_PLACE,
                                [tmpbuf, mdtype],
                                op=self.mpi.SUM)
            tensor.copy_from(tmpbuf)
        else:
            self.comm.Allreduce(self.mpi.IN_PLACE,
                                [tensor.asbuffer(), mdtype],
                                op=self.mpi.SUM)

    def is_distributed(self):
        return True


class ModelPar(BasePar):

    class Config(object):
        pass

    def __init__(self):
        super(ModelPar, self).__init__()
        if self.mpi_rank == 0:
            logger.info('Model-parallel mode. Number of nodes = %d.',
                        self.mpi_size)

    def init_model(self, model, backend):
        super(ModelPar, self).init_model(model, backend)
        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = ModelPar.Config()
            nout = layer.nout
            realnin = layer.nin
            nin = realnin / self.mpi_size
            conf.start = self.mpi_rank * nin
            if self.mpi_rank == (self.mpi_size - 1):
                # If the weights cannot be evenly partitioned, let the last
                # MPI node handle the extra weights.
                conf.end = realnin
            else:
                conf.end = conf.start + nin
            bs = model.batch_size
            bufshape = (layer.nout, bs)
            conf.fpropbuf = np.empty(bufshape, dtype=np.float32)
            bufshape = (layer.nin, bs)
            conf.bpropbuf = np.empty(bufshape, dtype=np.float32)
            conf.rcount = np.empty(self.mpi_size, dtype=np.int32)
            conf.rcount.fill(nin)
            conf.scount = conf.end - conf.start
            conf.rcount[-1] = realnin - nin * (self.mpi_size - 1)
            conf.displ = np.arange(0, realnin - nin + 1, nin)
            conf.scount *= bs
            conf.rcount *= bs
            conf.displ *= bs
            layer.weight_shape = (nout, conf.end - conf.start)
            layer.parconf = conf

    def associate(self, backend):
        super(ModelPar, self).associate(backend)
        self.orig_fprop_fc = backend.fprop_fc
        self.orig_bprop_fc = backend.bprop_fc
        self.orig_update_fc = backend.update_fc
        backend.fprop_fc = self.fprop_fc
        backend.bprop_fc = self.bprop_fc
        backend.update_fc = self.update_fc

    def distribute(self, batchdata):
        return self.backend.array(batchdata)

    def reduce_tensor(self, tensor):
        return tensor.asnumpyarray()

    def fprop_fc(self, out, inputs, weights, layer):
        conf = layer.parconf
        self.orig_fprop_fc(out, inputs[conf.start:conf.end], weights)
        sendbuf = [out.asbuffer(), self.mpi.FLOAT]
        recvbuf = [conf.fpropbuf, self.mpi.FLOAT]
        self.comm.Reduce(sendbuf, recvbuf, op=self.mpi.SUM)
        self.comm.Bcast(buf=[conf.fpropbuf, self.mpi.FLOAT])
        out.copy_from(conf.fpropbuf)

    def bprop_fc(self, out, weights, deltas, layer):
        conf = layer.parconf
        self.orig_bprop_fc(out[conf.start:conf.end], weights, deltas)
        outbuf = out.asbuffer()[conf.start:conf.end]
        sendbuf = [outbuf, conf.scount, self.mpi.FLOAT]
        recvbuf = [conf.bpropbuf, conf.rcount,
                   conf.displ, self.mpi.FLOAT]
        self.comm.Allgatherv(sendbuf, recvbuf)
        out.copy_from(conf.bpropbuf)

    def update_fc(self, out, inputs, deltas, layer):
        conf = layer.parconf
        self.orig_update_fc(out, inputs[conf.start:conf.end], deltas)

    def scatter(self, src, dest):
        dest.copy_from(src)

    def allocate_fragment(self, buf_shape, dtype=None):
        return self.backend.empty(buf_shape, dtype=dtype)


class DataPar(BasePar):

    class Config(object):
        pass

    def __init__(self):
        super(DataPar, self).__init__()
        if self.mpi_rank == 0:
            logger.info('Data-parallel mode. Number of nodes = %d.',
                        self.mpi_size)
        self.reducebuf = np.empty((1, 1), dtype=np.float32)

    def init_model(self, model, backend):
        super(DataPar, self).init_model(model, backend)
        self.batch_size = backend.actual_batch_size / self.mpi_size
        self.start = self.mpi_rank * self.batch_size
        if self.mpi_rank == (self.mpi_size - 1):
            self.batch_size = backend.actual_batch_size - self.start
        self.end = self.start + self.batch_size
        model.batch_size = self.batch_size

        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = DataPar.Config()
            conf.updatebuf = backend.empty(layer.weight_shape, dtype=np.float32)
            # conf.updatesz = layer.weight_shape[0] * layer.weight_shape[1]
            # if self.mpi_rank == 0:
            #     conf.updatebuf = backend.empty((self.mpi_size, conf.updatesz),
            #                                    dtype=np.float32)
            layer.parconf = conf

    def associate(self, backend):
        super(DataPar, self).associate(backend)
        self.orig_update_fc = backend.update_fc
        self.orig_update_conv = backend.update_conv
        backend.update_fc = self.update_fc
        backend.update_conv = self.update_conv
        self.npreducebuf = np.empty((self.mpi_size, 1), dtype=np.float32)
        if self.backend.__class__.__name__ == 'GPU':
            self.backend.setup_local_contexts(self.comm)

    def distribute(self, batchdata):
        return self.backend.array(batchdata[:, self.start:self.end])

    def reduce_tensor(self, tensor):
        # This is the case where we have a 1x1 tensor
        mdtype = self.dtype_to_mpi(tensor.dtype)
        self.comm.Gather([tensor.asnumpyarray(), mdtype],
                         [self.npreducebuf, mdtype])
        if self.mpi_rank == 0:
            return self.npreducebuf.sum() / self.mpi_size
        return 0

    def update(self, out, conf):
        # NOTE: To make this faster, compute the weight updates
        # asynchronously. There is no need to wait for completion
        # until the updates are to be applied to the weights (the
        # weights are updated after the gradients are propagated
        # all the way back).

        # NOTE: We should be able to shard the updates and do summation in
        # parts across the different devices, but it seems to block in MPI

        self.backend.all_reduce(self.comm, out, conf.updatebuf)
        # mdtype = self.dtype_to_mpi(out.dtype)

        # gbuf = conf.updatebuf.asbuffer() if self.mpi_rank == 0 else None
        # self.comm.Gather([out.asbuffer(), mdtype], [gbuf, mdtype])
        # if self.mpi_rank == 0:
        #     orig_shape = out.shape
        #     out = out.reshape((1, conf.updatebuf.shape[1]))
        #     self.backend.sum(conf.updatebuf, axes=0, out=out)
        #     out = out.reshape(orig_shape)
        # self.comm.Bcast([out.asbuffer(), mdtype])

    def update_fc(self, out, inputs, deltas, layer):
        self.orig_update_fc(out, inputs, deltas)
        self.update(out, layer.parconf)

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride,
                    ngroups, fwidth, updatebuf, local=False, layer=None):
        self.orig_update_conv(out, inputs, weights, deltas, ofmshape, ofmsize,
                              ofmlocs, ifmshape, links, nifm, padding, stride,
                              ngroups, fwidth, updatebuf, local)
        self.update(out, layer.parconf)

    def scatter(self, src, dest):
        self.backend.scatter_host(self.comm, src, dest)

        # if self.mpi_rank == 0:
        #     sz = src.shape[0] / self.mpi_size
        #     dest.copy_from(src[:sz])
        #     for i in range(1, self.mpi_size):
        #         self.comm.Send([src[sz*i:sz*(i+1)], mdtype], dest=i)
        # else:
        #     self.comm.Recv([dest.asbuffer(), mdtype], source=0)
        # self.comm.Scatter([src, mdtype],
        #                   [dest.asbuffer(), mdtype], root=0)

    def scatter_old(self, src, dest):
        mdtype = self.dtype_to_mpi(dest.dtype)
        # print mdtype
        if self.mpi_rank == 0:
            sz = src.shape[0] / self.mpi_size
            dest.copy_from(src[:sz])
            for i in range(1, self.mpi_size):
                self.comm.Send([src[sz*i:sz*(i+1)], mdtype], dest=i)
        else:
            self.comm.Recv([dest.asbuffer(), mdtype], source=0)
        # self.comm.Scatter([src, mdtype],
        #                   [dest.asbuffer(), mdtype], root=0)


    def allocate_fragment(self, buf_shape, dtype=None):
        fragment_buf_shape = (self.batch_size, buf_shape[1])
        return self.backend.zeros(fragment_buf_shape, dtype=dtype)
