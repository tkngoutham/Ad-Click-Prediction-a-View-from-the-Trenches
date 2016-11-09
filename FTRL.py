
import math.exp as exp
import math.sqrt as sqrt
import math.log as log

class FTRL(object):

    TrPath=""
    TePath=""
    ALPHA=None
    BETA = None
    L1 = None
    L2 = None
    LOGLIKHD = 0
    LOSS = []
    Z = None
    N = None
    NEPOCHS = None
    N_features=0
    def __init__(self,trainPath,testPath,a, b, l1, l2,num_feats, n_epoches):
        self.TrPath=trainPath
        self.TePath=testPath
        self.ALPHA = a
        self.BETA = b
        self.L1 = l1
        self.L2 = l2
        self.N_features=num_feats
        self.LOGLIKHD = 0
        self.LOSS = []
        self.Z = [0] * self.N_features
        self.N = [0] * self.N_features
        self.NEPOCHS = n_epoches
        return

    def updateZN(self,p_t,x_t,y_t,w):
        x_len=len(x_t)
        for i in range(0,x_len):
            g_i = (p_t - y_t) * x_t[i]
            sigma=(sqrt(self.N[i] + g_i * g_i) - sqrt(self.N[i]))
            sigma=sigma/self.ALPHA
            self.Z[i]=self.Z[i]+ g_i - sigma * w[i]
            self.N[i]=self.N[i]+ g_i * g_i

    def LogisticLoss(self,p_t,y_t):
        min_val=min(p_t, 1. - pow(10,-19)),
        p_t = max(min_val, pow(10,-19))
        if y_t == 1:
            ret_val=-log(p_t)
        else:
            ret_val=-log(1. - p_t)
        return ret_val


    #assuming that features are entirly numerical
    #N_featurs is number of features assuming last column in file be output variable
    #assuming that training data and testing data is in separate files
    def prepareData(self,tr_flag):
        x=[]
        y=[]
        if tr_flag=='tr':
            filepath=self.TrPath
        else: filepath=self.TePath
        with open(filepath,'r') as fp:
            for line in fp:
                tokens=line.split(',')
                if(len(tokens)!=self.N_features+1):
                    print(" invalid dataset ")
                    return
                x.append(tokens[0:self.N_features])
                y.append(tokens[self.N_features])
        return x,y

    def trainPerEpoch(self):
        X,Y=self.prepareData(tr_flag='tr')
        num_samples=len(Y)
        for i in range(0,num_samples):
            w = []
            prod_wx=0
            for j in range(0,len(X[i])):
                if self.Z[j] <= self.L1:
                    w[j] = 0
                else:
                    if self.Z[j] >= 0:
                        sgn=1
                    else: sgn=-1
                    w[j] = - (self.Z[j] - sgn * self.L1)
                    abpart=(self.L2 + (self.BETA + sqrt(self.N[j])) / self.ALPHA)
                    w[j]=w[j]/abpart
                prod_wx =prod_wx + w[j] * X[i][j]
            p_t = 1/(1+exp(-prod_wx))
            self.LOGLIKHD += self.LogisticLoss(p_t, Y[i])
            self.LOSS.append(self.LOGLIKHD)
            print('current log liklihood'+str(self.LOGLIKHD))
            self.updateZN(p_t,X[i],Y[i],w)



    def train(self):
        for epoch in range(0,self.NEPOCHS):
            print('current epoch '+str(epoch))
            self.trainPerEpoch('tr')




    def calcSigmoid(self):
        X,Y=self.prepareData(tr_flag='te')
        num_samples=len(Y)
        Y_PRED=[]
        for i in range(0,num_samples):
            w = []
            prod_wx=0
            for j in range(0,len(X[i])):
                if self.Z[j] <= self.L1:
                    w[j] = 0
                else:
                    if self.Z[j] >= 0:
                        sgn=1
                    else: sgn=-1
                    w[j] = - (self.Z[j] - sgn * self.L1)
                    abpart=(self.L2 + (self.BETA + sqrt(self.N[j])) / self.ALPHA)
                    w[j]=w[j]/abpart
                prod_wx =prod_wx + w[j] * X[i][j]
            p_t = 1/(1+exp(-prod_wx))
            if(p_t<=0.5):
                Y_PRED.append(0)
            else:
                Y_PRED.append(1)
        return Y_PRED





trainPath='path to train data'
testPath='path to test data'
# some random values
a=121
b=11
l1=1
l2=1
num_feats=10000
n_epoches=1

ftrl=FTRL(trainPath,testPath,a, b, l1, l2,num_feats, n_epoches)
ftrl.train()
labels=ftrl.calcSigmoid()
