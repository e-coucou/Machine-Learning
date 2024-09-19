import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd
import os

from model.utils import Normalize

class getGasData:
    def __init__(self, seq_len, batch_len=32, global_size=0, **kwargs):
        super(getGasData,self).__init__(**kwargs)
        self.seq_len = seq_len
        self.batch_len = batch_len
        self.global_size = global_size
        root_path = "../../data/"
        data_path = "bloc_ttf_prices.csv"
        df_raw = pd.read_csv(os.path.join(root_path,data_path))
        df_raw = df_raw.drop(columns=['Quarter-ahead','Summer 24','Winter 24'],axis=1)
        df_raw = df_raw.dropna(axis=0)

        # On ajoute les infos temporelles : mois/jour/joursemain/heure
        df_raw['mois'] = df_raw.date_m.apply(lambda row:int(row[5:7])).astype(np.int64)
        df_raw['jour'] = df_raw.date_m.apply(lambda row:int(row[8:10])).astype(np.int64)
        # df_raw['heure'] = df_raw.date_m.apply(lambda row:int(row[11:13])).astype(np.int64)
        df_raw['sJour'] = df_raw['date_m'].astype('datetime64[s]').dt.dayofweek.astype(np.int64)
        #formatte la date ...
        df_raw['date_m'] = pd.to_datetime(df_raw['date_m'])
        df_raw = df_raw.set_index('date_m')

        self.df_Data = df_raw.copy()
        # filtre les années Ukraine
        self.df_Data = self.df_Data[self.df_Data['Day-ahead']<50]

        size = self.df_Data.shape[0] if self.global_size==0 else self.global_size
        feature = self.df_Data.shape[1]
        self.data = self.df_Data.iloc[:size,:].to_numpy().reshape((size,-1))
        dataN_, self.scalerData = Normalize(self.data)
        self.dataN = np.reshape(dataN_,(size,feature))

    def get(self):
        featuresData = self.df_Data.iloc[:,1:5].astype(np.float32)
        featuresDate = self.df_Data[['mois','jour','sJour']].astype(np.int64)

        # Convert Features to numpy array
        featuresData = featuresData.to_numpy()
        featuresDate = featuresDate.to_numpy()
        # Create a dataset Feature Data
        X = np.array([featuresData[i:i+self.seq_len] for i in range(0, featuresData.shape[0]-self.seq_len)], dtype=np.float32)
        Y = np.array([featuresData[i+self.seq_len] for i in range(0, featuresData.shape[0]-self.seq_len)], dtype=np.float32)
        # Create a dataset : Features Date
        Xd = np.array([featuresDate[i:i+self.seq_len] for i in range(0, featuresDate.shape[0]-self.seq_len)], dtype=np.float32)
        Yd = np.array([featuresDate[i+self.seq_len] for i in range(0, featuresDate.shape[0]-self.seq_len)], dtype=np.float32)
        size = featuresData.shape[0] if self.global_size==0 else self.global_size
        self.Xt = tf.convert_to_tensor(X[:size,:,:], dtype=tf.float32)
        self.Yt = tf.convert_to_tensor(Y[:size,:], dtype=tf.float32)
        # Create a dataset : Features Date
        self.XtDate = tf.convert_to_tensor(Xd[:size,:,:], dtype=tf.float32)
        self.YtDate = tf.convert_to_tensor(Yd[:size,:], dtype=tf.float32)
        self.df = self.df_Data.iloc[:size,:]

        return (self.Xt,self.XtDate,self.Yt,self.YtDate,self.df)
    
    def display(self):
        fig = plt.figure(figsize=(18,4))
        plt.plot(self.df_Data.index, self.df_Data['Day-ahead'], label='Day aHead');
        plt.legend()

    def displayNorm(self):
        fig = plt.figure(figsize=(18,8))
        plt.plot(self.df_Data.index[:self.dataN.shape[0]], self.dataN[:,0], label='Day aHead');
        plt.plot(self.df_Data.index[:self.dataN.shape[0]], self.dataN[:,1], linestyle=(0,(2,1)), label='Month aHead');
        plt.plot(self.df_Data.index[:self.dataN.shape[0]], self.dataN[:,2], linestyle=(0,(1,3)), label='Year aHead');
        plt.plot(self.df_Data.index[:self.dataN.shape[0]], self.dataN[:,3], linestyle=(0,(1,2)),color='tab:orange', gapcolor='black' ,label='2 Years aHead');
        plt.legend()

    def buildDataset(self, split=0.2):
        dec_len = self.seq_len//2
        pred_len = dec_len//2
        full_len = self.seq_len+pred_len
        tmp = np.array([self.dataN[i:i+full_len] for i in range(0, self.dataN.shape[0]-full_len)], dtype=np.float32)
        # self.y = np.array([self.dataN[i+self.seq_len] for i in range(0, self.dataN.shape[0]-self.seq_len)], dtype=np.float32)
        self.X = tmp[:,:self.seq_len,:]
        self.y = tmp[:,dec_len:,:]

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=split, random_state=1965, shuffle=True) #, stratify=Y)

        self.dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        self.dataset = self.dataset.batch(self.batch_len)
        self.dataset = self.dataset.shuffle(self.batch_len)
        self.datasetVal = tf.data.Dataset.from_tensor_slices((X_val,y_val))
        self.datasetVal = self.datasetVal.batch(self.batch_len)
        self.datasetVal = self.datasetVal.shuffle(self.batch_len)

        return self.dataset, self.datasetVal
    
class getMyData :
    def __init__(self, data_size, seq_len, batch_len, **kwargs):
        super(getMyData, self).__init__(**kwargs)
        self.data_size=data_size
        self.seq_len = seq_len
        self.batch_len = batch_len
        # Sinusoide de TEST
        x = np.sin(np.linspace(0, 0.1*self.data_size, self.data_size))
        x_random = np.random.normal(0, 0.05, self.data_size) # add noise
        x_rampe = np.linspace(0,0.1*self.data_size,self.data_size)/self.data_size*10 # add rampe
        x = x+x_random + x_rampe
        x,scaler = Normalize(x)
        # x,scaler,xOrg = conver2diff(x, self.data_size, False)
        # x = Norm(x)
        # Create a dataset
        X = np.array([x[ii:ii+self.seq_len] for ii in range(0, x.shape[0]-self.seq_len)], dtype=np.float32)
        Y = np.array([x[ii+self.seq_len] for ii in range(0, x.shape[0]-self.seq_len)], dtype=np.float32)
        self.X = X.astype('float32')
        self.Y = Y.astype('float32')

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=369, shuffle=True)#, stratify=Y)

        self.dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        self.dataset = self.dataset.batch(self.batch_len)
        self.dataset = self.dataset.shuffle(self.batch_len)
        self.datasetVal = tf.data.Dataset.from_tensor_slices((X_val,y_val))
        self.datasetVal = self.datasetVal.batch(self.batch_len)
        self.datasetVal = self.datasetVal.shuffle(self.batch_len)

    def get(self):
        return self.dataset, self.datasetVal        

    def display(self):
        fig = plt.figure(figsize=(18,3))
        plt.plot(self.Y);

class getDataETTh1:
    def __init__(self, seq_len, batch_len=32, global_size=0):
        self.batch_len = batch_len
        self.seq_len = seq_len
        self.global_size = global_size

        # Get Features from ETTh1
        root_path = "../../data/"
        data_path = "ETTh1.csv"
        df_raw = pd.read_csv(os.path.join(root_path,data_path))
    
        # On ajoute les infos temporelles : mois/jour/joursemain/heure
        df_raw['mois'] = df_raw.date.apply(lambda row:int(row[5:7])).astype(np.int64)
        df_raw['jour'] = df_raw.date.apply(lambda row:int(row[8:10])).astype(np.int64)
        df_raw['heure'] = df_raw.date.apply(lambda row:int(row[11:13])).astype(np.int64)
        df_raw['sJour'] = df_raw['date'].astype('datetime64[s]').dt.dayofweek.astype(np.int64)

        self.df_Data = df_raw.copy()

    def get(self):
        featuresData = self.df_Data.iloc[:,1:8].astype(np.float32)
        featuresDate = self.df_Data[['mois','jour','sJour','heure']].astype(np.int64)

        # Convert Features to numpy array
        featuresData = featuresData.to_numpy()
        featuresDate = featuresDate.to_numpy()

        # Create a dataset Feature Data
        X = np.array([featuresData[i:i+self.seq_len] for i in range(0, featuresData.shape[0]-self.seq_len)], dtype=np.float32)
        Y = np.array([featuresData[i+self.seq_len] for i in range(0, featuresData.shape[0]-self.seq_len)], dtype=np.float32)
        # Create a dataset : Features Date
        Xd = np.array([featuresDate[i:i+self.seq_len] for i in range(0, featuresDate.shape[0]-self.seq_len)], dtype=np.float32)
        Yd = np.array([featuresDate[i+self.seq_len] for i in range(0, featuresDate.shape[0]-self.seq_len)], dtype=np.float32)

        if (self.global_size==0):
            self.Xt = tf.convert_to_tensor(X[:,:,:], dtype=tf.float32)
            self.Yt = tf.convert_to_tensor(Y[:,:], dtype=tf.float32)
            # XT = torch.from_numpy(X[:,:,:])
            self.XtDate = tf.convert_to_tensor(Xd[:,:,:], dtype=tf.float32)
            self.YtDate = tf.convert_to_tensor(Yd[:,:], dtype=tf.float32)
            # XTDate = torch.from_numpy(X[:,:])
            self.df = self.df_Data.iloc[0:,:]       
        else:
            self.Xt = tf.convert_to_tensor(X[:self.global_size,:,:], dtype=tf.float32)
            self.Yt = tf.convert_to_tensor(Y[:self.global_size,:], dtype=tf.float32)
            # XT = torch.from_numpy(X[:global_size,:,:])
            # Create a dataset : Features Date
            self.XtDate = tf.convert_to_tensor(Xd[:self.global_size,:,:], dtype=tf.float32)
            self.YtDate = tf.convert_to_tensor(Yd[:self.global_size,:], dtype=tf.float32)
            # XTDate = torch.from_numpy(X[:global_size,:])
            self.df = self.df_Data.iloc[0:self.global_size,:]
        
        return (self.Xt,self.XtDate,self.Yt,self.YtDate,self.df)
    
    def buildDataset(self, split=0.2):
        N,F = self.df_Data.shape # global_size, 12
        if (self.global_size>0):
            N=self.global_size
        data_ = self.df_Data.iloc[:self.global_size,1:].to_numpy().reshape((N,-1))
        self.dataN,self.scalerData = Normalize(data_)
        self.data = np.reshape(self.dataN,(N,11))
        self.X = np.array([self.data[i:i+self.seq_len] for i in range(0, self.data.shape[0]-self.seq_len)], dtype=np.float32)
        self.y = np.array([self.data[i+self.seq_len] for i in range(0, self.data.shape[0]-self.seq_len)], dtype=np.float32)

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=split, random_state=1965, shuffle=True) #, stratify=Y)

        self.dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        self.dataset = self.dataset.batch(self.batch_len)
        self.dataset = self.dataset.shuffle(self.batch_len)
        self.datasetVal = tf.data.Dataset.from_tensor_slices((X_val,y_val))
        self.datasetVal = self.datasetVal.batch(self.batch_len)
        self.datasetVal = self.datasetVal.shuffle(self.batch_len)

        return self.dataset, self.datasetVal
