import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os   
import cv2
import tqdm
from    tensorflow import keras
from    tensorflow.keras import layers,Sequential,optimizers,losses,metrics
from    tqdm import tqdm
from    sklearn import preprocessing
from    sklearn.preprocessing import OneHotEncoder
from    pandas import DataFrame

def muil_to_bin(x):
    if x>0:
        return 1
    else:
        return 0

def preprocess(in_file,out_file,ifsample=0,sample_num=0,sample_frac=0.):
    col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count_","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
    protocol_type={'tcp': 0, 'udp': 1, 'icmp': 2}
    service={'aol': 0, 'auth': 1, 'bgp': 2, 'courier': 3, 'csnet_ns': 4, 'ctf': 5, 'daytime': 6, 'discard': 7, 'domain': 8,
         'domain_u': 9, 'echo': 10, 'eco_i': 11, 'ecr_i': 12, 'efs': 13, 'exec': 14, 'finger': 15, 'ftp': 16,
         'ftp_data': 17, 'gopher': 18, 'harvest': 19, 'hostnames': 20, 'http': 21, 'http_2784': 22, 'http_443': 23,
         'http_8001': 24, 'imap4': 25, 'IRC': 26, 'iso_tsap': 27, 'klogin': 28, 'kshell': 29, 'ldap': 30, 'link': 31,
         'login': 32, 'mtp': 33, 'name': 34, 'netbios_dgm': 35, 'netbios_ns': 36, 'netbios_ssn': 37, 'netstat': 38,
         'nnsp': 39, 'nntp': 40, 'ntp_u': 41, 'other': 42, 'pm_dump': 43, 'pop_2': 44, 'pop_3': 45, 'printer': 46,
         'private': 47, 'red_i': 48, 'remote_job': 49, 'rje': 50, 'shell': 51, 'smtp': 52, 'sql_net': 53, 'ssh': 54,
         'sunrpc': 55, 'supdup': 56, 'systat': 57, 'telnet': 58, 'tftp_u': 59, 'tim_i': 60, 'time': 61, 'urh_i': 62,
         'urp_i': 63, 'uucp': 64, 'uucp_path': 65, 'vmnet': 66, 'whois': 67, 'X11': 68, 'Z39_50': 69}
    flag={'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5, 'S1': 6, 'S2': 7, 'S3': 8, 'SF': 9, 'SH': 10}
    label={'normal.': 0, 'ipsweep.': 1, 'mscan.': 2, 'nmap.': 3, 'portsweep.': 4, 'saint.': 5, 'satan.': 6, 'apache2.': 7,
         'back.': 8, 'land.': 9, 'mailbomb.': 10, 'neptune.': 11, 'pod.': 12, 'processtable.': 13, 'smurf.': 14,
         'teardrop.': 15, 'udpstorm.': 16, 'buffer_overflow.': 17, 'httptunnel.': 18, 'loadmodule.': 19, 'perl.': 20,
         'ps.': 21, 'rootkit.': 22, 'sqlattack.': 23, 'xterm.': 24, 'ftp_write.': 25, 'guess_passwd.': 26, 'imap.': 27,
         'multihop.': 28, 'named.': 29, 'phf.': 30, 'sendmail.': 31, 'snmpgetattack.': 32, 'snmpguess.': 33, 'spy.': 34,
         'warezclient.': 35, 'warezmaster.': 36, 'worm.': 37, 'xlock.': 38, 'xsnoop.': 39}
    
    source_data=pd.read_csv(in_file,names=col_names,header=None,index_col=False)
    if ifsample == 1:
        if sample_num == 0 :
            source_data = source_data.sample(frac=sample_frac)
        if sample_frac == 0.:
            source_data = source_data.sample(n=sample_num)
    #先转化字符型数据为数值型
    #随机抽取数据之后一定要重置行索引，不然之后可能会使concat增加多余数据
    source_data=source_data.reset_index(drop=True)
    source_data.protocol_type=source_data.protocol_type.map(protocol_type)
    source_data.service=source_data.service.map(service)
    source_data.flag=source_data.flag.map(flag)
    source_data.label=source_data.label.map(label)
    label_temp=source_data.label
    #label_temp=label_temp.reset_index(drop=True)
    #label_temp.to_csv('VAL_LABEL.csv',header=None,index=None)
    #print('label:',label_temp)
    label_temp=label_temp.map(muil_to_bin)#转化为2分类
    #对字符型数据转化的数值编码进行独热编码
    enc=OneHotEncoder(sparse=False)
    #3种协议类型
    enc.fit( [ [0], [1], [2] ])
    enc_1=DataFrame(enc.transform(DataFrame(source_data.protocol_type)))#需要把列数据转换为DataFrame()类型
    #70中服务类型
    enc.fit([[0],[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19],
            [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37],
            [38], [39],[40],[41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],
            [60], [61], [62], [63], [64], [65], [66], [67], [68], [69],])
    enc_2=DataFrame(enc.transform(DataFrame(source_data.service)))
    #11种网络连接转态
    enc.fit([[0],[1], [2], [3], [4], [5], [6], [7], [8], [9],[10]])
    enc_3=DataFrame(enc.transform(DataFrame(source_data.flag)))
    #合并表
    after_onehot=pd.concat([source_data,enc_1,enc_2,enc_3],axis=1)
    print('afr_oh:',after_onehot.shape)
    #删去原本字符列
    after_onehot=after_onehot.drop(columns='protocol_type')
    after_onehot=after_onehot.drop(columns='service')
    after_onehot=after_onehot.drop(columns='flag')
    after_onehot=after_onehot.drop(columns='label')

    #标准化 z-score
    after_standardize=after_onehot.apply(preprocessing.scale)

    #量化 (x-min)/(max-min)*255
    after_quantize=after_standardize.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x))*255 )
    handled_data=after_quantize.fillna(0)
    handled_data=pd.concat([handled_data,label_temp],axis=1)#将label列保存在最后一列
    print(handled_data.shape)
    handled_data.to_csv(out_file,header=None,index=None)
    

def zizagcode(arrary,imgsize=16):
    #锯齿形编码填充
    zigzag=np.zeros([imgsize,imgsize])
    row=zigzag.shape[0]
    colum=zigzag.shape[1]
    i,j,k=0,0,0
    while i < row and j < colum and k < len(arrary):
        zigzag[i,j]=arrary[k]
        k+=1
        #i+j为偶数，向右上方向移动
        if (i+j)%2==0 :
            #如果右边界超出，则向下
            if (i-1) in range(row) and (j+1) not in range(colum):
                i+=1
            #如果上边界超出，则向下
            elif (i-1) not in range(row) and (j+1) in range(colum):
                j+=1
            #如果右上边界都超出，则向下
            elif (i-1) not in range(row) and (j+1) not in range(colum):
                i+=1
            else:
                i-=1
                j+=1
        #i+j为奇数，则向左下移动
        elif (i+j)%2==1:
            #如果左边界超出，则向下
            if (i+1) in range(row) and (j-1) not in range(colum):
                i+=1
            #如果上边界超出，则向下
            elif (i+1) not in range(row) and (j-1) in range(colum):
                j+=1
            #如果右上边界都超出，则向下
            elif (i+1) not in range(row) and (j-1) not in range(colum):
                j+=1
            else:
                i+=1
                j-=1
    
    #zigza填充结束，进行IDCT
    idct=cv2.idct(zigzag.flatten())
    img=idct.reshape([imgsize,imgsize])
    
    return img