ALPHA=None
BETA = None
L1 = None
L2 = None
LOGLIKHD = 0
LOSS = []
Z = None
N = None
COEFF = {}
CNAME = None
T_RATIO = 0.
RATE = None
SUBSAMPLE = None
NEPOCHS = None
FITFLG = None


def initialize(a, b, l1, l2, s_sam, ep=1, rate=0):
    ALPHA = a
    BETA = b
    L1 = l1
    L2 = l2
    LOGLIKHD = 0
    LOSS = []
    Z = None
    N = None
    COEFF = {}
    CNAME = None
    T_RATIO = 0.
    RATE = rate
    SUBSAMPLE = s_sam
    NEPOCHS = ep
    FITFLG = False
    return ALPHA,BETA,L1,L2,LOGLIKHD,LOSS,Z,N,COEFF,CNAME,T_RATIO,RATE,SUBSAMPLE,NEPOCHS,FITFLG
    
    
def reInitVars(self):
    LOGLIKHD = 0
    LOSS = []
    Z = None
    N = None
    COEFF = {}
    CNAME = None
    return LOGLIKHD,LOSS,Z,N,COEFF,CNAME

def LogisticLoss(p_t,y_t):
    min_val=min(p_t, 1. - pow(10,-14))
    p_t = max(min_val, pow(10,-14))
    if y_t == 1:
        ret_val=-log(p_t)
    else:
        ret_val=-log(1. - p_t)
    return ret_val
    
    
