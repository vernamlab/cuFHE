import lib.fhepy_gpu as fhe
import time
import timeit

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

def Init(pubkey):
    fhe.Initialize(pubkey)

def PtxtMod():
    return fhe.Ptxt().PtxtSpace

def Encrypt(ptxt, prikey, count=1, pubkey=None):
    if pubkey is None:
        pubkey = PubKeyGen(prikey)
    if isinstance(ptxt, (int, long)):
        msg = ptxt
        ptxt = fhe.Ptxt()
        if count == 1:
            ptxt.message = msg;
            ctxt = Ctxt(pubkey)
    	    fhe.Encrypt(ctxt.ctxt_, ptxt, prikey)
            return ctxt

        msg_bin = bin(msg)[2:].zfill(count)
        msg_list = []
        ct = CtxtList(count, pubkey)
        for i in range(count):
            ptxt.message = int(msg_bin[i], 2)
            fhe.Encrypt(ct.ctxts_[count - i - 1].ctxt_, ptxt, prikey)
        return ct

def Decrypt(ctxt, prikey):
    ptxt = fhe.Ptxt()
    if isinstance(ctxt, Ctxt):
        fhe.Decrypt(ptxt, ctxt.ctxt_, prikey)
        return ptxt.message

    if isinstance(ctxt, CtxtList):
        ptxt_list = ""
        for c in reversed(ctxt.ctxts_):
            fhe.Decrypt(ptxt, c.ctxt_, prikey)
            ptxt_list += str(ptxt.message)
        return int(ptxt_list, 2)

def SetSeed():
    fhe.SetSeed(int(time.time()))


class Ctxt:
    def __init__(self, pubkey=None):
        self.ctxt_ = fhe.Ctxt()
        self.pubkey_ = pubkey

    def Decrypt(self, prikey):
        return Decrypt(self, prikey)

    def Encrypt(self, msg, prikey):
        Encrypt(msg, prikey, self.pubkey_)
        return self

    def __and__(self, other):
        result = Ctxt(self.pubkey_)
        st = fhe.Stream()
        st.Create()
        fhe.Synchronize()
        fhe.AND(result.ctxt_, self.ctxt_, other.ctxt_, st)
        fhe.Synchronize()
        return result

    def __xor__(self, other):
        result = Ctxt(self.pubkey_)
        st = fhe.Stream()
        st.Create()
        fhe.Synchronize()
        fhe.XOR(result.ctxt_, self.ctxt_, other.ctxt_, st)
        fhe.Synchronize()
        return result

    def __or__(self, other):
        result = Ctxt(self.pubkey_)
        st = fhe.Stream()
        st.Create()
        fhe.Synchronize()
        fhe.OR(result.ctxt_, self.ctxt_, other.ctxt_, st)
        fhe.Synchronize()
        return result

    def __invert__(self):
        result = Ctxt(self.pubkey_)
        st = fhe.Stream()
        st.Create()
        fhe.Synchronize()
        fhe.NOT(result.ctxt_, self.ctxt_, st)
        fhe.Synchronize()
        return result



class CtxtList:
    def __init__(self, length=0, pubkey=None):
        self.ctxts_ = []
        for i in range(length):
            self.ctxts_.append(Ctxt(pubkey))
        self.pubkey_ = pubkey

    def Decrypt(self, prikey):
        return Decrypt(self, prikey)

    def __and__(self, other):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = []
        for i in range(len(self.ctxts_)):
            st.append(fhe.Stream())
            st[i].Create()
        fhe.Synchronize()
        for i in range(len(self.ctxts_)):
            fhe.AND(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[i])
        fhe.Synchronize()
        return result

    def __xor__(self, other):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = []
        for i in range(len(self.ctxts_)):
            st.append(fhe.Stream())
            st[i].Create()
        fhe.Synchronize()
        for i in range(len(self.ctxts_)):
            fhe.XOR(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[i])
        fhe.Synchronize()
        return result

    def __or__(self, other):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = []
        for i in range(len(self.ctxts_)):
            st.append(fhe.Stream())
            st[i].Create()
        fhe.Synchronize()
        for i in range(len(self.ctxts_)):
            fhe.OR(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[i])
        fhe.Synchronize()
        return result

    def __invert__(self):
        result = CtxtList(len(self.ctxts_), self.pubkey_)
        st = []
        for i in range(len(self.ctxts_)):
            st.append(fhe.Stream())
            st[i].Create()
        fhe.Synchronize()
        for i in range(len(self.ctxts_)):
            fhe.AND(result.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, st[i])
        fhe.Synchronize()
        return result


    def __add__(self, other):
        k = len(self.ctxts_)
        st = []
        for i in range(3*k):
            st.append(fhe.Stream())
            st[i].Create()
        fhe.Synchronize()

        ksa_p = CtxtList(k, self.pubkey_)
        ksa_g = CtxtList(k, self.pubkey_)
	ksa_c = CtxtList(k, self.pubkey_)
	ksa_s = CtxtList(k, self.pubkey_)

        for i in range(k):
            fhe.AND(ksa_g.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[3*i])
            fhe.XOR(ksa_p.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[3*i+1])
            fhe.XOR(ksa_s.ctxts_[i].ctxt_, self.ctxts_[i].ctxt_, other.ctxts_[i].ctxt_, st[3*i+2])
	fhe.Synchronize()

        begin = 0
        step = 1
	while begin+step < k:
	    for i in range(begin+step, k):
                id = i - begin - step
                ctxt = ksa_p.ctxts_[i].ctxt_
	        fhe.AND(ksa_p.ctxts_[i].ctxt_, ctxt, ksa_p.ctxts_[i-step].ctxt_, st[2*id])
	        fhe.AND(ksa_c.ctxts_[i].ctxt_, ctxt, ksa_g.ctxts_[i-step].ctxt_, st[2*id+1])
            fhe.Synchronize()

	    for i in range(begin+step, k):
                id = i - begin - step
	        fhe.OR(ksa_g.ctxts_[i].ctxt_, ksa_c.ctxts_[i].ctxt_, ksa_g.ctxts_[i].ctxt_, st[id])
            fhe.Synchronize()
            step += 1
            begin += 1

        for i in range(1,k):
             fhe.XOR(ksa_s.ctxts_[i].ctxt_, ksa_s.ctxts_[i].ctxt_, ksa_g.ctxts_[i-1].ctxt_, st[i])
        fhe.Synchronize()
        return ksa_s
