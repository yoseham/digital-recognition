
from numpy import *
from sklearn.metrics import f1_score, precision_score, recall_score
class AODE():
    def __init__(self, d, class_num = 2,interval = 1,kind='raw'):
        #discrete features number
        self.d = d
        self.class_num = class_num
        self.interval = interval
        self.kind = kind



    def fit(self, X, y):
        if self.kind=='feature':
            kt = 17
        elif self.kind == 'raw':
            kt = 256
        else:
            kt = 3
        count_xj_c_xi = zeros((self.class_num,self.d,kt,self.d,kt))
        count_c_xi = zeros((self.class_num,self.d,kt))
        prob_xj_c_xi = zeros((self.class_num,self.d,kt,self.d,kt))
        prob_c_xi = zeros((self.class_num,self.d,kt))
        N = X.shape[0]
        attrs = []
        for k in range(self.d):
            attrs.append([i for i in range(kt)])
        inl = self.interval
        for n in range(N):
            for i in range(self.d):
                count_c_xi[y[n],i,X[n,i]] += 1
                strong_link = [i-inl-1,i-inl,i-inl+1,i-1,i+1,i+inl-1,i+inl,i+inl+1]
                for j in strong_link:
                    if j<0 or j>=self.d:
                        continue
                    count_xj_c_xi[y[n],i,X[n,i],j,X[n,j]] += 1

        for c in range(self.class_num):
            for i in range(self.d):
                v_i = len(attrs[i])
                for k1 in range(kt):
                    # for attr_i_value, N_c_xi in count_c_xi[c][i].items()
                    prob_c_xi[c,i,k1] = float(count_c_xi[c,i,k1] + 1) / (N + self.class_num *v_i)
                    strong_link = [i-inl-1,i-inl,i-inl+1,i-1,i+1,i+inl-1,i+inl,i+inl+1]
                    for j in strong_link:
                        if j<0 or j>=self.d:
                            continue
                        v_j = len(attrs[j])
                        for k2 in range(kt):
                            prob_xj_c_xi[c,i,k1,j,k2] = log(float(count_xj_c_xi[c,i,k1,j,k2] + 1) / (count_c_xi[c,i,k1] + v_j))
        self.count_xj_c_xi = count_xj_c_xi
        self.count_c_xi = count_c_xi
        self.prob_xj_c_xi = prob_xj_c_xi
        self.prob_c_xi = prob_c_xi
        print('train finish')


    def predict(self, X):
        result = []
        inl = self.interval
        for x in X:
            probs = []
            for c in range(self.class_num):
                prob_c = 0
                for i in range(self.d):

                    prob_j_c_i_product = 0
                    strong_link = [i-inl-1,i-inl,i-inl+1,i-1,i+1,i+inl-1,i+inl,i+inl+1]
                    for j in strong_link:
                        if j<0 or j>=self.d:
                            continue
                    # for j in range(self.d):
                        prob_j_c_i_product += self.prob_xj_c_xi[c][i][x[i]][j][x[j]]
                    prob_c_i_term = self.prob_c_xi[c][i][x[i]] + prob_j_c_i_product
                    prob_c += prob_c_i_term
                probs.append(prob_c)
            label = probs.index(max(probs))
            self.y = probs
            result.append(label)

        return result


    def incrementLearning(self,trainX,trainLabel,testX,testLabel,batch,epoch):

        Nt, M = trainX.shape
        # data = concatenate((trainX,trainLabel),axis=1)
        # random.shuffle(data)
        labelArr = array(trainLabel).flatten()
        labelSet = set(labelArr)
        self.labelSet = labelSet
        for epo in range(epoch):
            #随机抽取
            choose = random.randint(0,Nt-1,batch)
            X = trainX[choose]
            y = trainLabel[choose]
            # new_data = data[epo*batch:epo*batch+batch,:]
            # X = new_data[:,:-1]
            # y = new_data[:,-1:]
            ####### 训练 ###########
            if self.kind=='feature':
                kt = 17
            elif self.kind == 'raw':
                kt = 256
            else:
                kt = 3
            if epo == 0:
                self.count_xj_c_xi = zeros((self.class_num,self.d,kt,self.d,kt))
                self.count_c_xi = zeros((self.class_num,self.d,kt))
                self.prob_xj_c_xi = zeros((self.class_num,self.d,kt,self.d,kt))
                self.prob_c_xi = zeros((self.class_num,self.d,kt))
            N = X.shape[0]
            attrs = []
            for k in range(self.d):
                attrs.append([i for i in range(kt)])
            inl = self.interval
            for n in range(N):
                for i in range(self.d):
                    self.count_c_xi[y[n],i,X[n,i]] += 1
                    strong_link = [i-inl-1,i-inl,i-inl+1,i-1,i+1,i+inl-1,i+inl,i+inl+1]
                    for j in strong_link:
                        if j<0 or j>=self.d:
                            continue
                        self.count_xj_c_xi[y[n],i,X[n,i],j,X[n,j]] += 1

            for c in range(self.class_num):
                for i in range(self.d):
                    v_i = len(attrs[i])
                    for k1 in range(kt):
                    # for attr_i_value, N_c_xi in count_c_xi[c][i].items()
                        self.prob_c_xi[c,i,k1] = float(self.count_c_xi[c,i,k1] + 1) / (N + self.class_num *v_i)
                        strong_link = [i-inl-1,i-inl,i-inl+1,i-1,i+1,i+inl-1,i+inl,i+inl+1]
                        for j in strong_link:
                            if j<0 or j>=self.d:
                                continue
                            v_j = len(attrs[j])
                            for k2 in range(kt):
                                self.prob_xj_c_xi[c,i,k1,j,k2] = log(float(self.count_xj_c_xi[c,i,k1,j,k2] + 1) / (self.count_c_xi[c,i,k1] + v_j))

            if epo % 5 ==0:
            ####### 预测 ###########
                X = testX
                result = []
                inl = self.interval
                for x in X:
                    probs = []
                    for c in range(self.class_num):
                        prob_c = 0
                        for i in range(self.d):

                            prob_j_c_i_product = 0
                            strong_link = [i-inl-1,i-inl,i-inl+1,i-1,i+1,i+inl-1,i+inl,i+inl+1]
                            for j in strong_link:
                                if j<0 or j>=self.d:
                                    continue
                                prob_j_c_i_product += self.prob_xj_c_xi[c,i,x[i],j,x[j]]
                            prob_c_i_term = self.prob_c_xi[c,i,x[i]] + prob_j_c_i_product
                            prob_c += prob_c_i_term
                        probs.append(prob_c)
                    label = probs.index(max(probs))
                    result.append(label)

                ####### 准确率分析 ###########
                print("训练数据：{}".format(batch*(epo+1)))
                print("F-score: {}".format(f1_score(average = 'macro', y_pred = result,  y_true = testLabel)))#计算预测的F1
                print("Accuracy: {}".format((result == testLabel).sum()/len(testLabel)))#计算预测的正确率