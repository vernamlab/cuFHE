import pyfhe
import time
import random


# Rand Seeds
random.seed()
pyfhe.SetSeed(int(time.time()))


# If you want to see the default FHE parameters,
# you can use:
params = pyfhe.Param()
print(params.lwe_n_)
print(params.tlwe_n_)
# ...

# Keys
pubkey = pyfhe.PubKey()
prikey = pyfhe.PriKey()
pyfhe.KeyGen(pubkey, prikey)
# If you want to save the keys, you can store them
# in a file:
#pyfhe.WritePriKeyToFile(prikey, "prikey.txt")
#pyfhe.WritePubKeyToFile(pubkey, "pubkey.txt")
# You can also read them from a file
#pyfhe.ReadPubKeyFromFile(pubkey, "pubkey.txt")
#pyfhe.ReadPriKeyFromFile(prikey, "prikey.txt")


# Plaintext, default ptext space is 2
# If you want to change it, you can use:
# P.PtxtSpace = some_integer;
P = pyfhe.Ptxt()
P.message_ = random.randint(0,1)
print("Plaintext message : " + str(P.message_))


# Ciphertext
C = pyfhe.Ctxt()


# Encrypt - Decrypt
pyfhe.Encrypt(C, P, prikey)
print("Writing computed ciphertext to file ./ctxt.txt")
pyfhe.WriteCtxtToFile(C, "ctxt.txt")
P_dec = pyfhe.Ptxt()
pyfhe.Decrypt(P_dec, C, prikey)
print("Decrypted message : " + str(P_dec.message_))

