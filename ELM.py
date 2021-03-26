from numpy import *
from sklearn.preprocessing import StandardScaler      #数据预处理
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
class ELM():
    def __init__(self,n,d,hidden_num,activate='sigmoid',w=None,dropout=0):
        self.d = d
        self.out_w = None
        if activate == 'sigmoid':
            self.activate = self.sigmoid
        else:
            self.activate = self.relu
        self.hidden_num = hidden_num
        #随机生成权重
        if w is None:
            self.w = random.uniform(-1,1,(d,hidden_num))
        else:
            self.w = w
        if dropout > 0:
            choose = []
            for i in range(int(hidden_num*dropout)):
                choose.append(random.randint(0,hidden_num))
            self.w = delete(self.w,array(choose),axis=1)
            self.hidden_num -= len(set(choose))

        #随机生成偏置
        self.b = zeros([n,self.hidden_num],dtype=float)
        for i in range(self.hidden_num):
            rand_b = random.uniform(-0.8,0.8)
            for j in range(n):
                self.b[j,i] = rand_b

    def sigmoid(self,x):
        return 1.0/(1+exp(-x))

    def relu(self,x):
        return maximum(0,x)

    def softmax(self,y):
        return exp(y)/sum(exp(y))

    def fit(self,data,label):
        stdsc = StandardScaler()#标准化
        data = stdsc.fit_transform(data)
        onehot = OneHotEncoder(categories='auto')#onehot编码
        y = onehot.fit_transform(label.reshape(label.shape[0],1)).todense().A#将数组转onehot
        s = dot(data,self.w)+self.b
        h = self.activate(s)
        h_ = linalg.pinv(h)
        self.out_w = dot(h_,y)

        # I = identity(self.hidden_num)
        # self.out_w = dot(linalg.inv(dot(transpose(h),h) + I /self.c) , dot(transpose(h),label))


    def predict(self,data):
        stdsc = StandardScaler()
        data = stdsc.fit_transform(data)
        n = data.shape[0]
        s = dot(data,self.w)+self.b[:n,:]
        h = self.activate(s)
        y = self.softmax(dot(h,self.out_w))
        predictions = zeros(n)
        for i in range(n):
            predictions[i] = y[i].argmax()
        return predictions

    def mini(self):
        self.b=array([self.b[0]])

    def evaluate(self,data):
        # data[data>0]=1
        # print(data)
        data = data*data
        s = dot(data,self.w)+self.b[0]
        h = self.activate(s)
        y = self.softmax(dot(h,self.out_w))
        print(y)
        return y

    def train(self,data,label):
        stdsc = StandardScaler()#标准化
        data = stdsc.fit_transform(data)
        onehot = OneHotEncoder(categories='auto')#onehot编码
        y = onehot.fit_transform(label.reshape(label.shape[0],1)).todense().A#将数组转onehot
        self.w = self.PSO(data,label,y,k = 10,maxIter = 20)
        s = dot(data,self.w)+self.b
        h = self.activate(s)
        h_ = linalg.pinv(h)
        self.out_w = dot(h_,y)



    def PSO(self,data,label,y,k,maxIter=10):
        W = random.uniform(-0.5,0.5,(k,self.d,self.hidden_num));V = zeros((k,self.d,self.hidden_num))
        Vmax = 0.2;Vmin = -0.2#速度范围
        Xmax = 0.5;Xmin = -0.5#h范围
        w=1;hiter=0
        pBestv = [inf]*k;pBesti = W.copy()
        gBestv = inf;gBesti = random.uniform(-1,1,(self.d,self.hidden_num)); gBestved = gBestv;
        while(hiter<maxIter):
            print(hiter)
            for i in range(k):
                s = dot(data,W[i])+self.b
                h = self.activate(s)
                h_ = linalg.pinv(h)
                out_w = dot(h_,y)
                y_predict = self.softmax(dot(h,out_w))
                f = log_loss(y_true=label,y_pred=y_predict,labels=[0,1,2,3,4,5,6,7,8,9])

                print(f)
                if f <= pBestv[i]:#得到粒子历史最优
                    pBesti[i] = W[i].copy()
                    pBestv[i] = f
                    if f < gBestv:
                        gBestv = f#粒子群最优
                        gBesti = W[i].copy()

            for i in range(k):
                V[i] = w*V[i] + (1-hiter/maxIter)*(pBesti[i]-W[i]) + (gBesti-W[i])#更新速度
                V[i] = V[i].clip(max=Vmax,min=Vmin)#限制速度范围
                W[i] = W[i] + V[i]#更新h
                W[i] = W[i].clip(max=Xmax,min=Xmin)#限制范围

            print(gBestv)
            if abs(gBestv - gBestved)<1e-9:#判断是否收敛
                break
            else:
                gBestved = gBestv
            hiter+=1#迭代次数加一
        return gBesti

