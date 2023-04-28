#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:02:15 2022

@author: changruidi
"""


import pandas as pd
from sklearn import preprocessing  
import numpy as np
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
import math
import xgboost as xgb
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Reshape, Activation
from keras.models import Sequential
from keras import backend,activations
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import scipy

from jqdatasdk import *
auth('18665883365','Zszq#201')

q=6#过去几个点
n=50#50支股票
p=160#用过去160个点标准化
onemonth=1008#一个月的时间
oneday=48#一天的时间




class Data_Preparation:
    def get_data():
        stocklist=['睿创微纳','天准科技','华兴源创','昊海生科','晶丰明源','致远互联','嘉元科技','普门科技','容百科技','杭可科技',
           '光峰科技','澜起科技','中国通号','福光股份','中微公司','交控科技','心脉医疗','乐鑫科技','安集科技','方邦股份',
           '瀚川智能','安恒信息','沃尔德','南微医学','山石网科','天宜上佳','传音控股','宝兰德','航天宏图','虹软科技',
           '申联生物','晶晨股份','威胜信息','三达膜','金山办公','天奈科技','西部超导','清溢光电','海尔生物','优刻得',
           '博瑞医药','安博通','柏楚电子','卓越新能','久日新材','泽璟制药','特宝生物','长阳科技','微芯生物','华熙生物']
        arr = np.zeros((4,50))
        stock=pd.DataFrame(arr,columns=[stocklist[i] for i in range(50)],index=['code','name','startdate','enddate'])
        for i in range(len(stocklist)):
            for j in range(len(df)):
                if df.iloc[j,0]==stocklist[i]:
                    stock.iloc[0,i]=str(df.iloc[j,:].name)
                    stock.iloc[1,i]=str(df.iloc[j,1])
                    stock.iloc[2,i]=str(df.iloc[j,2])
                    stock.iloc[3,i]=str(df.iloc[j,3])
        KC50=get_price('000688.XSHG', start_date='2020-07-23', end_date='2022-06-14', frequency='minute', fields=None, skip_paused=False, fq='pre')
        KC50.to_csv('KC50')
        for i in range(50):
            temp=get_price(stock.iloc[0,i], start_date='2020-07-23', end_date='2022-06-14', frequency='minute', fields=None, skip_paused=False, fq='pre')
            temp.to_csv(stock.iloc[1,i])
    def cal_alpha():#计算alpha
        path = "/Users/changruidi/CMUS2/CMS/科创板"
        filelist = os.listdir(path)
        dfList=[]
        namelist=[]
        for i in filelist:
            filepath = os.path.join(path,i)
            if filepath=='/Users/changruidi/CMUS2/CMS/科创板/.DS_Store':
                continue
            if filepath=='/Users/changruidi/CMUS2/CMS/科创板/KC50':
                continue
            if filepath=='/Users/changruidi/CMUS2/CMS/科创板/.ipynb_checkpoints':
                continue
            if filepath=='/Users/changruidi/CMUS2/CMS/科创板/CMS_JUN14.ipynb':
                continue
            a = pd.read_csv(filepath)
            a = a.set_index(a.columns[0])
            a=a['close']
            namelist.append(i)
            dfList.append(a)
        X = pd.concat(dfList, axis=1)
        X.columns=[namelist[i] for i in range(50)]
        X.to_csv('X.csv')
        #交易频率 5min
        for i in range(X.shape[0]):
            if str(X.iloc[i].name).endswith('5:00') or str(X.iloc[i].name).endswith('0:00'):
                X_5min=X_5min.append(X.iloc[i])
        X_5min.to_csv('X_5min.csv')
        KC50_5min=pd.DataFrame()
        for i in range(KC50.shape[0]):
            if str(KC50.iloc[i].name).endswith('5:00') or str(KC50.iloc[i].name).endswith('0:00'):
                KC50_5min=KC50_5min.append(KC50.iloc[i])
        KC50_5min.to_csv('KC50_5min.csv')
        #股票价格/KC50
        for j in range(X_5min.shape[1]):
            for i in range(len(X_5min)):
                X_5min.iloc[i,j]=X_5min.iloc[i,j]/KC50_5min.iloc[i,1]
    def pchange():#价格变化
        X_5min_price_change=pd.DataFrame(columns=[namelist[i] for i in range(50)])
        for i in range(X_5min.shape[1]):
            X_5min_price_change[namelist[i]]=(X_5min[namelist[i]].shift(-1) - X_5min[namelist[i]]) / X_5min[namelist[i]]
        X_5min_price_change.to_csv('X_5min_price_change.csv')
    def stan():#标准化
        arr = np.zeros((len(X_5min_price_change),50))
        X_price_change_standardization=pd.DataFrame(arr,columns=[namelist[i] for i in range(50)])
        for j in range(50):
            for i in range(len(X_price_change_standardization)):
                X_price_change_standardization.iloc[i,j]=X_5min_price_change.iloc[i,j]/X_5min_price_change.iloc[i-p:i,j].std()
        X_price_change_standardization
        a = X.index.tolist()
        a=a[-len(X_price_change_standardization):]
        b=np.array(X_price_change_standardization)
        output = pd.DataFrame(b,columns=[namelist[i] for i in range(50)],index=a)
        output.to_csv('X_5min_price_change_standardization.csv')
    def cdf():#累计分布函数
        x=pd.read_csv('/Users/changruidi/CMUS2/CMS/科创板/X_5min_price_change_standardization.csv')
        x=x.set_index(x.columns[0])
        x.drop(x.head(p).index, inplace=True)
        x.drop(x.tail(1).index, inplace=True)
        st=scipy.stats.norm.cdf(X_price_change_standardization)
        time=x.index.tolist()
        st=pd.DataFrame(st,columns=[namelist[i] for i in range(50)],index=time)
        st=st*2-1
        st.to_csv('standardized_data.csv')
    def xy(x):#构造xy
        st=x.values
        xx=np.zeros(shape=(len(st)-p,n,q))
        for j in range(n):
            for i in range(p,len(st)):
                xx[i-p,j]=st[i-q:i,j]
        y=st[p:]
        return xx,y
    
    
class XGB:
    def XGBR(xx,y):
        # XGB Regressor
        final=pd.DataFrame(columns=[namelist[i] for i in range(50)])
        for i in range(int(y.shape[0]*0.4/onemonth)):
            print(i)
            Y_train=np.array(y[i*onemonth:i*onemonth+int(y.shape[0]*0.6)])
            Y_test=np.array(y[int(y.shape[0]*0.6)+i*onemonth:int(y.shape[0]*0.6)+(i+1)*onemonth])
            X_train=xx[i*onemonth:i*onemonth+int(y.shape[0]*0.6)]
            X_test=xx[int(y.shape[0]*0.6)+i*onemonth:int(y.shape[0]*0.6)+(i+1)*onemonth]
            train_x = X_train.reshape(-1,X_train.shape[2])
            train_y=Y_train.reshape(-1,1)
            train_x[np.isnan(train_x)] = 0
            train_y[np.isnan(train_y)] = 0
            test_x = X_test.reshape(-1,X_test.shape[2])
            test_y=Y_test.reshape(-1,1)
            test_x[np.isnan(test_x)] = 0
            test_y[np.isnan(test_y)] = 0
            xgbr = xgb.XGBRegressor(booster='gbtree',objective='reg:linear',subsample=1)
            xgbr.fit(train_x, train_y)
            result= xgbr.predict(test_x)
            result=result.reshape(-1,50)
            result=pd.DataFrame(result,columns=[namelist[i] for i in range(50)],index=time[p+int(y.shape[0]*0.6)+i*onemonth:p+int(y.shape[0]*0.6)+(i+1)*onemonth])
            final=final.append(result) # 预测结果
        final.to_csv('final_XGB_06.csv')
        
        
class SVM:
    def SVM(xx,y):
        final=pd.DataFrame(columns=[namelist[i] for i in range(50)])
        for i in range(int(y.shape[0]*0.4/onemonth)):
            print(i)
            Y_train=np.array(y[i*onemonth:i*onemonth+int(y.shape[0]*0.6)])
            Y_test=np.array(y[int(y.shape[0]*0.6)+i*onemonth:int(y.shape[0]*0.6)+(i+1)*onemonth])
            X_train=xx[i*onemonth:i*onemonth+int(y.shape[0]*0.6)]
            X_test=xx[int(y.shape[0]*0.6)+i*onemonth:int(y.shape[0]*0.6)+(i+1)*onemonth]
            train_x = X_train.reshape(-1,X_train.shape[2])
            train_y=Y_train.reshape(-1,1)
            train_x[np.isnan(train_x)] = 0
            train_y[np.isnan(train_y)] = 0
            test_x = X_test.reshape(-1,X_test.shape[2])
            test_y=Y_test.reshape(-1,1)
            test_x[np.isnan(test_x)] = 0
            test_y[np.isnan(test_y)] = 0
            svr_rbf = svm.SVR(kernel='rbf', max_iter=100)
            svr_rbf.fit(train_x, train_y)
            result=svr_rbf.predict(test_x)
            result=result.reshape(-1,50)
            result=pd.DataFrame(result,columns=[namelist[i] for i in range(50)],index=time[160+int(y.shape[0]*0.6)+i*onemonth:p+int(y.shape[0]*0.6)+(i+1)*onemonth])
            final=final.append(result) # 预测结果
        final.to_csv('final_SVM_06.csv')


class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
    def predict(self, X):   
        alpha=[(0.9**i) for i in range(q-1,-1,-1)]
        X = np.asarray(X)
        result = np.zeros((X.shape[0],X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x=X[i,j,:]
                # 计算距离(计算与训练集中每个X的距离)
                xcopy=np.full((self.X.shape[0],self.X.shape[1],self.X.shape[2]),list(x))
                distance=((((xcopy-self.X)**2)*alpha).sum(axis=-1))**0.5
                flat=distance.reshape(distance.shape[0]*distance.shape[1]) 
                # 取前k个距离最近的索引
                index=flat.argsort()
                index = index[:self.k]
                # 求权重
                s = np.sum(1/(flat[index]+0.001))  # 加上0.001，是为了避免距离为0的情况
                weight = (1/(flat[index]+0.001))/s
                yflat=self.y.ravel()
                #tmp=[]
                #for h in range(self.k):
                #    tmp.append([time[index[h]//50],namelist[index[h]%50]])
                result[i,j]=np.sum(yflat[index]*weight)
        return result
    def KNNR(xx,y):
        result=np.zeros((0,50))
        for i in range(int(y.shape[0]*0.4/onemonth)):
            print(i)
            train_y=np.array(y[i*onemonth:i*onemonth+int(y.shape[0]*0.6)])
            test_y=np.array(y[int(y.shape[0]*0.6)+i*onemonth:int(y.shape[0]*0.6)+(i+1)*onemonth])
            train_x=xx[i*onemonth:i*onemonth+int(y.shape[0]*0.6)]
            test_x=xx[int(y.shape[0]*0.6)+i*onemonth:int(y.shape[0]*0.6)+(i+1)*onemonth]
            train_x[np.isnan(train_x)] = 0
            train_y[np.isnan(train_y)] = 0
            test_x[np.isnan(test_x)] = 0
            test_y[np.isnan(test_y)] = 0
            knn = KNN(k=30)
            knn.fit(train_x, train_y)
            result2= knn.predict(test_x)
            result=np.concatenate((result, result2))
        final=pd.DataFrame(result,columns=[namelist[i] for i in range(50)],index=[time[p+int(y.shape[0]*0.6):p+int(y.shape[0]*0.6)+(i+1)*onemonth]])
        final.to_csv('final_knnr_30_06.csv')



class RNNCNN:
    def myLoss(y_true,y_pred):
        return backend.sum(backend.square((y_pred-y_true)*y_true),axis=-1)
    def myMetric(y_true,y_pred):
        return backend.sum(y_pred*y_true,axis=-1)
    def myActivation(x):
        return 'tanh'
    def RNN(xx,y):
        xxx=xx*10
        yyy=y*10
        final=pd.DataFrame(columns=[namelist[i] for i in range(50)])
        for i in range(int(y.shape[0]*0.4/onemonth)):
            Y_train=np.array(yyy[i*onemonth:int(y.shape[0]*0.6)-onemonth+i*onemonth])
            Y_validation=np.array(yyy[i*onemonth+int(y.shape[0]*0.6)-onemonth:int(y.shape[0]*0.6)+i*onemonth])
            Y_test=np.array(yyy[i*onemonth+int(y.shape[0]*0.6):int(y.shape[0]*0.6)+(i+1)*onemonth])
            X_train=xxx[i*onemonth:int(y.shape[0]*0.6)-onemonth+i*onemonth]
            X_validation=np.array(xxx[i*onemonth+int(y.shape[0]*0.6)-onemonth:int(y.shape[0]*0.6)+i*onemonth])
            X_test=xxx[i*onemonth+int(y.shape[0]*0.6):int(y.shape[0]*0.6)+(i+1)*onemonth]
            train_x = X_train.reshape(-1,X_train.shape[2],1)
            train_y=Y_train.reshape(-1,1)
            train_x[np.isnan(train_x)]=0
            train_y[np.isnan(train_y)]=0
            validation_x = X_validation.reshape(-1,X_validation.shape[2],1)
            validation_y=Y_validation.reshape(-1,1)
            validation_x[np.isnan(validation_x)]=0
            validation_y[np.isnan(validation_y)]=0
            test_x = X_test.reshape(-1,X_train.shape[2],1)
            test_y=Y_test.reshape(-1,1)
            test_x[np.isnan(test_x)]=0
            test_y[np.isnan(test_y)]=0
            regressor=Sequential()
            regressor.add(SimpleRNN(units=32,input_shape=(train_x.shape[1],1)))
            regressor.add(Dense(units=1,activation='tanh'))
            regressor.compile(optimizer='nadam',loss=RNNCNN.myLoss,metrics=[RNNCNN.myMetric])
            earlystop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
            savebestmodel=tf.keras.callbacks.ModelCheckpoint('./best_weights.hdf5',monitor='val_loss',verbose=1,save_best_only=True, mode='auto')
            history=regressor.fit(train_x, train_y, batch_size=100, validation_data=(validation_x, validation_y), callbacks=[earlystop,savebestmodel], verbose=1,shuffle=False)   
            regressor.summary()
            predict=regressor.predict(test_x)
            predict=predict.reshape(-1,50)
            result=pd.DataFrame(predict,columns=[namelist[i] for i in range(50)],index=time[p+int(y.shape[0]*0.6)+i*onemonth:p+int(y.shape[0]*0.6)+(i+1)*onemonth])
            final=final.append(result) # 预测结果
        final.to_csv('final_RNN_48.csv')
    def CNN(xx,y):
        final=pd.DataFrame(columns=[namelist[i] for i in range(50)])
        for i in range(int(y.shape[0]*0.4/onemonth)):
            Y_train=np.array(y[i*onemonth:int(y.shape[0]*0.6)-onemonth+i*onemonth])
            Y_validation=np.array(y[i*onemonth+int(y.shape[0]*0.6)-onemonth:int(y.shape[0]*0.6)+i*onemonth])
            Y_test=np.array(y[i*onemonth+int(y.shape[0]*0.6):int(y.shape[0]*0.6)+(i+1)*onemonth])
            X_train=xx[i*onemonth:int(y.shape[0]*0.6)-onemonth+i*onemonth]
            X_validation=np.array(xx[i*onemonth+int(y.shape[0]*0.6)-onemonth:int(y.shape[0]*0.6)+i*onemonth])
            X_test=xx[i*onemonth+int(y.shape[0]*0.6):int(y.shape[0]*0.6)+(i+1)*onemonth]
            train_x = X_train.reshape(-1,X_train.shape[2],1)
            train_y=Y_train.reshape(-1,1)
            train_x[np.isnan(train_x)]=0
            train_y[np.isnan(train_y)]=0
            validation_x = X_validation.reshape(-1,X_validation.shape[2],1)
            validation_y=Y_validation.reshape(-1,1)
            validation_x[np.isnan(validation_x)]=0
            validation_y[np.isnan(validation_y)]=0
            test_x = X_test.reshape(-1,X_train.shape[2],1)
            test_y=Y_test.reshape(-1,1)
            test_x[np.isnan(test_x)]=0
            test_y[np.isnan(test_y)]=0
            regressor=Sequential() 
            regressor.add(Conv1D(100,4,padding='same',input_shape=(train_x.shape[1],1)))
            regressor.add(MaxPooling1D(2))
            regressor.add(Flatten())
            regressor.add(Dense(units=1,activation='tanh'))
            regressor.compile(optimizer='nadam',loss=RNNCNN.myLoss,metrics=[RNNCNN.myMetric])
            earlystop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
            savebestmodel=tf.keras.callbacks.ModelCheckpoint('./best_weights.hdf5',monitor='val_loss',verbose=1,save_best_only=True, mode='auto')
            history=regressor.fit(train_x, train_y, batch_size=100, validation_data=(validation_x, validation_y), callbacks=[earlystop,savebestmodel], verbose=1,shuffle=False)   
            regressor.summary()
            predict=regressor.predict(test_x)
            predict=predict.reshape(-1,50)
            result=pd.DataFrame(predict,columns=[namelist[i] for i in range(50)],index=time[p+int(y.shape[0]*0.6)+i*onemonth:p+int(y.shape[0]*0.6)+(i+1)*onemonth])
            final=final.append(result) # 预测结果
        final.to_csv('final_CNN_48.csv')

    
class Analysis:    
    def profit(final):
        # 平均单次投资收益
        arr= np.zeros((len(final),3))
        yprofit=pd.DataFrame(arr,columns=['prediction','price_change','profit'])
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        for i in range(50):#50支股票
            yprofit['prediction']=pd.to_numeric(final[namelist[i]]).tolist()
            yprofit['price_change']=pd.to_numeric(price_change[namelist[i]]).tolist()
            yprofit['profit']=yprofit['prediction']*yprofit['price_change']
            income=yprofit['profit'].sum(axis=0)
            put=abs(yprofit['prediction']).sum()
            print(namelist[i],income/put)
    
    def extreme(final):
        # 前后10%的
        arr= np.zeros((len(final),3))
        yprofit=pd.DataFrame(arr,columns=['prediction','price_change','profit'],index=final.index.tolist())
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        for i in range(50):#50支股票
            yprofit['prediction']=pd.to_numeric(final[namelist[i]]).tolist()
            yprofit['price_change']=pd.to_numeric(price_change[namelist[i]]).tolist()
            yprofit['profit']=yprofit['prediction']*yprofit['price_change']
            #income1=0
            #income2=0
            income=0
            #print(yprofit.iloc[:,0].quantile(0.9),yprofit.iloc[:,0].quantile(0.1))
            for j in range(len(final)):
                if yprofit.iloc[j,0]>yprofit.iloc[:,0].quantile(0.9):
                    #income1+=yprofit.iloc[j,1]
                    income+=yprofit.iloc[j,1]
                elif yprofit.iloc[j,0]<yprofit.iloc[:,0].quantile(0.1):
                    #income2-=yprofit.iloc[j,1]
                    income-=yprofit.iloc[j,1]
            print(namelist[i],income)
            #yprofit.to_csv('SVM_12_'str(namelist[i])+'.csv')  
    
    def timeanalysis(final):
        for i in range(1,8):#7个月
            pc=price_change.values
            fn=final.values
            fno=fn[i*onemonth:(i+1)*onemonth]
            pco=pc[i*onemonth:(i+1)*onemonth]
            fnpre=fn[(i-1)*onemonth:i*onemonth]
            pcpre=pc[(i-1)*onemonth:i*onemonth]
            fn9=fno[fno>np.quantile(fnpre.reshape(-1,1),0.9)]
            fn1=fno[fno<np.quantile(fnpre.reshape(-1,1),0.1)]
            profit9=pco[fno>np.quantile(fnpre.reshape(-1,1),0.9)]
            profit1=pco[fno<np.quantile(fnpre.reshape(-1,1),0.1)]
            income=profit9.sum()-profit1.sum()
            print(income/50)
    
    def yplot():
        XGB=pd.read_csv('final_XGB_06.csv')
        XGB=XGB.set_index(XGB.columns[0])
        KNN=pd.read_csv('final_knnr_30_06.csv')
        KNN=KNN.set_index(KNN.columns[0])
        RNN=pd.read_csv('final_RNN_48.csv')
        RNN=RNN.set_index(RNN.columns[0])
        arr= np.zeros((len(XGB),11))
        yprofit=pd.DataFrame(arr,columns=['price_change','XGB_06','KNN30_06','RNN_48','XGB_06_01','KNN30_06_01','RNN_48_01','price_change_cump','XGB_06_cump','KNN30_06_cump','RNN_48_01_cump'],index=price_change.index.tolist())
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        for i in range(49,50):#这里改第几支股票
            yprofit['price_change']=pd.to_numeric(price_change[namelist[i]]).tolist()
            yprofit['XGB_06']=pd.to_numeric(XGB[namelist[i]]).tolist()
            yprofit['KNN30_06']=pd.to_numeric(KNN[namelist[i]]).tolist()
            yprofit['RNN_48']=pd.to_numeric(RNN[namelist[i]]).tolist()
            for k in range(1,4):
                print(k)
                for j in range(onemonth,len(price_change)):
                    if yprofit.iloc[j,k]>yprofit.iloc[j-onemonth:j,k].quantile(0.9):
                        yprofit.iloc[j,k+3]=1
                    elif yprofit.iloc[j,k]<yprofit.iloc[j-onemonth:j,k].quantile(0.1):
                        yprofit.iloc[j,k+3]=-1
                    else:
                        yprofit.iloc[j,k+3]=0
            yprofit.drop(yprofit.head(onemonth).index,inplace=True)
            yprofit['price_change_cump']=(yprofit['price_change']+1).cumprod()
            for k in range(4,7):
                for j in range(len(price_change)-onemonth):
                    countsell=0
                    countbuy=0
                    if j<oneday:
                        for l in range(j+oneday):
                            if (yprofit.iloc[l,:].name[:10]==yprofit.iloc[j,:].name[:10]):
                                if yprofit.iloc[l,k]==1:
                                    countbuy+=1
                                if yprofit.iloc[l,k]==-1:
                                    countsell+=1
                        if yprofit.iloc[j,k]==1:
                            yprofit.iloc[j,k]=yprofit.iloc[j,k]/countbuy
                        if yprofit.iloc[j,k]==-1:
                            yprofit.iloc[j,k]=yprofit.iloc[j,k]/countsell
                    elif j>(len(price_change)-onemonth-oneday):
                        for l in range(j-oneday,len(price_change)-onemonth):
                            if (yprofit.iloc[l,:].name[10]==yprofit.iloc[j,:].name[10]):
                                if yprofit.iloc[l,k]==1:
                                    countbuy+=1
                                if yprofit.iloc[l,k]==-1:
                                    countsell+=1
                        if yprofit.iloc[j,k]==1:
                            yprofit.iloc[j,k]=yprofit.iloc[j,k]/countbuy
                        if yprofit.iloc[j,k]==-1:
                            yprofit.iloc[j,k]=yprofit.iloc[j,k]/countsell
                    else:
                        for l in range(j-oneday,j+oneday):
                            if (yprofit.iloc[l,:].name[10]==yprofit.iloc[j,:].name[10]):
                                if yprofit.iloc[l,k]==1:
                                    countbuy+=1
                                if yprofit.iloc[l,k]==-1:
                                    countsell+=1
                        if yprofit.iloc[j,k]==1:
                            yprofit.iloc[j,k]=yprofit.iloc[j,k]/countbuy
                        if yprofit.iloc[j,k]==-1:
                            yprofit.iloc[j,k]=yprofit.iloc[j,k]/countsell
            yprofit['XGB_06_cump']=(yprofit['XGB_06_01']*yprofit['price_change']+1).cumprod()
            yprofit['KNN30_06_cump']=(yprofit['KNN30_06_01']*yprofit['price_change']+1).cumprod()
            yprofit['RNN_48_cump']=(yprofit['RNN_48_01']*yprofit['price_change']+1).cumprod()
            #yprofit['XGB_30_cump']=(yprofit['XGB_30_cump']+1).cumprod()
            #yprofit['KNN30_30_cump']=(yprofit['KNN30_30_cump']+1).cumprod()
            #yprofit['RNN_240_cump']=(yprofit['RNN_240_cump']+1).cumprod()
            yprofit[['price_change_cump','XGB_06_cump','KNN30_06_cump','RNN_48_cump']].dropna().plot()
            plt.title(str(namelist[i]))
            print(yprofit)
            plt.savefig(str('5min_'+namelist[i]),bbox_inches='tight')
            plt.show()
            yprofit.to_csv(str('5min_'+namelist[i]+'.csv'))
            
    
    
if __name__ == '__main__':
    namelist=['SSWK','RBKJ','SLSW','PMKJ','RCWN','BRYY','QYGD','LQKJ','ZJZY','YKD','BLD','BCDZ','CYKG','HXYC',
              'JKKJ','HXSW','NWYX','TNKJ','FGGF','WED','JRXC','HTHT','LXKJ','JYKJ','HRKJ','WSXX','ZGTH','FBGF',
              'TYSJ','HKKJ','JFMY','ABT','GFKJ','HESW','XBCD','WXSW','XMYL','ZYXN','SDM','HCZN','AHXX','CYKJ',
              'JCGF','ZWGS','TZKJ','TBSW','ZYHL','AJKJ','JSBG','HHSK']

    x=pd.read_csv('standardized_data.csv')
    x=x.set_index(x.columns[0])
    time=x.index.tolist()
    

    price_change=pd.read_csv('X_5min_price_change.csv')
    price_change=price_change.set_index(price_change.columns[0])
    price_change=price_change['2021-09-07 14:30:00':'2022-05-25 14:25:00']
    
    #for i in range(8):
        #print(price_change.iloc[i*onemonth,:].name,price_change.iloc[(i+1)*onemonth-1,:].name)
        
    #xx,y=xy(x)
    #XGB.XGBR(xx,y) 
    #SVM.SVM(xx,y) 
    #KNN.KNNR(xx,y)
    #RNNCNN.RNN(xx,y)
    #RNNCNN.CNN(xx,y)

    #final=pd.read_csv('test805.csv')
    #final=final.set_index(final.columns[0])
    #Analysis.extreme(final)
    #Analysis.profit(final)
    #Analysis.timeanalysis(final)
    Analysis.yplot()
    