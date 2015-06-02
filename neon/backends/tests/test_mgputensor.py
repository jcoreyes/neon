#!/usr/bin/env python
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

import numpy as np
from nose.plugins.attrib import attr
from nose.tools import nottest

from neon.util.testing import assert_tensor_equal, assert_tensor_near_equal

def m_assert_tensor_equal(t1, t2):
    for _t1, _t2, ctx in zip(t1._tensorlist, t2._tensorlist, t1._ctxs):
        ctx.push()
        assert_tensor_equal(_t1, _t2)
        ctx.pop()

@attr('cuda')
class TestGPU(object):

    def setup(self):
        from neon.backends.mgpu import MGPU, MGPUTensor
        # this code gets called prior to each test
        self.be = MGPU(rng_seed=0, num_dev=2)
        self.gpt = MGPUTensor

    @attr('bbx')
    def reduction_test(self):
        # create a numpy array as the test-bed
        ASIZE = 10
        h_a = np.random.randn(ASIZE * self.be.num_dev).reshape(
                (self.be.num_dev, ASIZE)).astype(self.be.default_dtype)
        h_result = np.sum(h_a, axis=0)

        d_a = self.be.empty((1, ASIZE))
        u_a = self.be.empty((1, ASIZE))
        self.be.scatter(h_a, d_a)
        self.be.reduce(d_a, u_a)
        print h_result
        print d_a._tensorlist[0].asnumpyarray()
        # np.testing.assert_allclose(actual, h_result, atol=1e-6, rtol=0)
