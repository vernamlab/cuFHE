try:
    import fhepy_gpu as fhe
except ImportError:
    try:
        import fhepy_cpu as fhe
    except ImportError:
        pass
    pass

def LoadPubKey(pubfile="pubkey.txt"):
    pubkey = fhe.PubKey()
    fhe.ReadPubKeyFromFile(pubkey, pubfile)
    return pubkey

def LoadPriKey(prifile="prikey.txt"):
    prikey = fhe.PriKey()
    fhe.ReadPriKeyFromFile(prikey, prifile)
    return prikey

def LoadKeys(pubfile="pubkey.txt", prifile="prikey.txt"):
    return LoadPubKey(), LoadPriKey()

def StorePubKey(pubkey, pubfile="pubkey.txt"):
    fhe.WritePubKeyToFile(pubkey, pubfile)
    print("Public key is stored in ./" + pubfile)

def StorePriKey(prikey, prifile="prikey.txt"):
    fhe.WritePriKeyToFile(prikey, prifile)
    print("Private key is stored in ./" + prifile)

def StoreKeys(pubkey, prikey, pubfile="pubkey.txt", prifile="prikey.txt"):
    StorePubKey(pubkey, pubfile)
    StorePriKey(prikey, prifile)

def PriKeyGen():
    prikey = fhe.PriKey()
    fhe.PriKeyGen(prikey)
    return prikey

def PubKeyGen(prikey):
    pubkey = fhe.PubKey()
    fhe.PubKeyGen(pubkey, prikey)
    return pubkey

def KeyGen():
    prikey = PriKeyGen()
    return PubKeyGen(prikey), prikey

def PtxtMod():
    return fhe.Ptxt().PtxtSpace

def Encrypt(ptxt, prikey, pubkey=None):
    if pubkey is None:
        pubkey = PubKeyGen(prikey)
    ctxt = Ctxt(pubkey)
    if isinstance(ptxt, (int, long)):
        msg = ptxt
        ptxt = fhe.Ptxt()
        ptxt.message = msg;
    fhe.Encrypt(ctxt.ctxt_, ptxt, prikey)
    return ctxt

def Decrypt(ctxt, prikey):
    ptxt = fhe.Ptxt()
    fhe.Decrypt(ptxt, ctxt.ctxt_, prikey)
    return ptxt.message

def AND(ctxt1, ctxt2, pubkey):
    ctxt = fhe.Ctxt()
    fhe.AND(ctxt, ctxt1, ctxt2, pubkey)
    return ctxt

def OR(ctxt1, ctxt2, pubkey):
    ctxt = fhe.Ctxt()
    fhe.OR(ctxt, ctxt1, ctxt2, pubkey)
    return ctxt

def XOR(ctxt1, ctxt2, pubkey):
    ctxt = fhe.Ctxt()
    fhe.XOR(ctxt, ctxt1, ctxt2, pubkey)
    return ctxt

def NAND(ctxt1, ctxt2, pubkey):
    ctxt = fhe.Ctxt()
    fhe.NAND(ctxt, ctxt1, ctxt2, pubkey)
    return ctxt

def NOT(ctxt1):
    ctxt = fhe.Ctxt()
    fhe.NOT(ctxt, ctxt1)
    return ctxt


class Ctxt:
    def __init__(self, pubkey=None):
        self.ctxt_ = fhe.Ctxt()
        self.pubkey_ = pubkey

    def __and__(self, other):
         result = Ctxt(self.pubkey_)
         result.ctxt_ = AND(self.ctxt_, other.ctxt_, self.pubkey_)
         return result

    def __or__(self, other):
         result = Ctxt(self.pubkey_)
         result.ctxt_ = OR(self.ctxt_, other.ctxt_, self.pubkey_)
         return result

    def __xor__(self, other):
         result = Ctxt(self.pubkey_)
         result.ctxt_ = XOR(self.ctxt_, other.ctxt_, self.pubkey_)
         return result

    def __invert__(self):
         result = Ctxt(self.pubkey_)
         result.ctxt_ = NOT(self.ctxt_)
         return result

    def Decrypt(self, prikey):
        return Decrypt(self, prikey)

    def Encrypt(self, msg, prikey):
        Encrypt(msg, prikey, self.pubkey_)
        return self
