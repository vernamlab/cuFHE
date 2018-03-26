import pyfhe
import time
import random


random.seed()
pyfhe.SetSeed(int(time.time()))

pubkey = pyfhe.PubKey()
prikey = pyfhe.PriKey()
pyfhe.KeyGen(pubkey, prikey)

P1 = pyfhe.Ptxt()
P1.message = random.randint(0,1)
C1 = pyfhe.Ctxt()
pyfhe.Encrypt(C1, P1, prikey)

P2 = pyfhe.Ptxt()
P2.message = random.randint(0,1)
C2 = pyfhe.Ctxt()
pyfhe.Encrypt(C2, P2, prikey)


# AND
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.AND(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " AND " + str(P2.message) + " : " + str(P.message))


# NAND
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.NAND(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " NAND " + str(P2.message) + " : " + str(P.message))


# OR
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.OR(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " OR " + str(P2.message) + " : " + str(P.message))


# XOR
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.XOR(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " XOR " + str(P2.message) + " : " + str(P.message))
