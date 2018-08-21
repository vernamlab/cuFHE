################################################################################
# Copyright 2018 Gizem S. Cetin <gscetin@wpi.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import lib.fhe_cpu as fhe
import random
import operator

def CheckResult(m, op, result):
        return op(m[0], m[1]) == result


pubkey, prikey = fhe.KeyGen()
m = [random.randint(0,1), random.randint(0,1)]
c0, c1 = fhe.Encrypt(m[0], prikey), fhe.Encrypt(m[1], prikey)

# AND Gate
c2 = c0 & c1
result = c2.Decrypt(prikey)
print("AND gate : " + str(CheckResult(m, operator.__and__, result)))

# XOR Gate
c3 = c0 ^ c1
result = c3.Decrypt(prikey)
print("XOR gate : " + str(CheckResult(m, operator.__xor__, result)))

# OR Gate
c4 = c0 | c1
result = c4.Decrypt(prikey)
print("OR gate : " + str(CheckResult(m, operator.__or__, result)))

# NOT Complement
c5 = ~c0
result = c5.Decrypt(prikey)
print("NOT gate : " + str(result != m[0]))

# NAND Gate
c6 = c0 & c1
c7 = ~c6
result = c7.Decrypt(prikey)
print("NAND gate : " + str(not CheckResult(m, operator.__and__, result)))
