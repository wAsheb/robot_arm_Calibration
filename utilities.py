# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:42:57 2018

@author: e10509
"""
# In[1]
import numpy as np
import matplotlib.pyplot as plt
#from scipy.special import expit
import math as math
from sympy import*
PI = np.pi
# In[2]
D2R = 0.01745329252    #PI/180
R2D = 57.2957795130    #180/PI
#define dHtable dictionary
# {0,0.105,0.2800,0.065,0.0,0.0,0.0}
Tool = [0]
q   = [Symbol('q%i' % ii)for ii in range(7)]
a   = [Symbol('a%i' % ii)for ii in range(7)]
d   = [Symbol('d%i' % ii)for ii in range(7)]
alp = [Symbol('al%i' %ii)for ii in range(7)]
#a,al,d,q = symbols('a al d q')
#print(q[1])  
# In[3]
DH_table = {'a0': 0.0,         'al0': PI,        'd1': -423.0,    'q1': 0.0, 
            'a1': 104.613,     'al1': PI/2,      'd2': 0.0,       'q2': -PI/2,   
            'a2': 280.129,     'al2': 0.0,       'd3': 0.0,       'q3': 0.0, 
            'a3': 65.0,        'al3': PI/2,      'd4': -284.0,    'q4': 0.0,
            'a4': 0.0,         'al4': -PI/2,     'd5': 0.0,       'q5': 0.0,
            'a5': 0.0,         'al5': -PI/2,     'd6': 90.5,      'q6': 0.0 }

nParameters = {'a0': 0.0,      'al0': PI,        'd1': -423.0,    'q1': 0.0,   
            'a1': 105.0,       'al1': PI/2,      'd2': 0.0,       'q2': 0.0, 
            'a2': 280.0,       'al2': 0.0,       'd3': 0.0,       'q3': 0.0,
            'a3': 65.0,        'al3': PI/2,      'd4': -284.0,    'q4': 0.0,
            'a4': 0.0,         'al4': -PI/2,     'd5': 0.0,       'q5': 0.0,
            'a5': 0.0,         'al5': -PI/2,     'd6': 90.5,      'q6': 0.0}

nTestParam = {'a0': 0.0,      'al0': PI,         'd1': -423.0,       
            'a1': 105.0,       'al1': PI/2,      'd2': 0.0,        
            'a2': 280.0,       'al2': 0.0,       'd3': 0.0,       
            'a3': 65.0,        'al3': PI/2,      'd4': -284.0,   
            'a4': 0.0,         'al4': -PI/2,     'd5': 0.0,       
            'a5': 0.0,         'al5': -PI/2,     'd6': 90.5 }

jointConfig = {'q1':9.05880481e-02,
               'q2':-1.37911820e+00,
               'q3':-5.84525391e-02,
               'q4':1.92422551e-04,
               'q5':1.57268214e+00,
               'q6':1.29712690e-02}
jointdq      = {'q1':0.0,
               'q2':0.0,
               'q3':0.0,
               'q4':0.0,
               'q5':0.0,
               'q6':0.0}
#sDH_table = {'a0': a[0],         'al0': alp[0],           'd1':  d[1],    'q1': q[1],
#             'a1': a[1],         'al1': alp[1],           'd2':  d[2],    'q2': q[2],
#             'a2': a[2],         'al2': alp[2],           'd3':  d[3],    'q3': q[3],
#             'a3': a[3],         'al3': alp[3],           'd4':  d[4],    'q4': q[4],
#             'a4': a[4],         'al4': alp[4],           'd5':  d[5],    'q5': q[5],
#             'a5': a[5],         'al5': alp[5],           'd6':  d[6],    'q6': q[6],
#             'a6': a[6],         'al6': alp[6],           'd7':  d[6],    'q7': 0}
# In[4]
def hTransformation(a,alpha,d,q):
    Ti_12i = Matrix([[cos(q), -1*sin(q),  0.0, a],
               [ sin(q)*cos(alpha), cos(q)*cos(alpha), -1*sin(alpha), -1*sin(alpha)*d],
               [ sin(q)*sin(alpha), cos(q)*sin(alpha), cos(alpha),   cos(alpha)*d],
               [      0.0,    0.0,    0.0,    1.0]])
    
    return Ti_12i
# In[5]
#print (hTransformation(0,(np.pi)/2,-423.205,0))
#for idx, val in enumerate(obIn):
#    print(idx,val)
# In[6]
def npHT(a,al,d,q):
    Tnp = np.matrix([[np.cos(q), -1*np.sin(q),  0.0, a],
               [ np.sin(q)*np.cos(al), np.cos(q)*np.cos(al), -1*np.sin(al), -1*np.sin(al)*d],
               [ np.sin(q)*np.sin(al), np.cos(q)*np.sin(al), np.cos(al),   np.cos(al)*d],
               [      0.0,    0.0,    0.0,    1.0]])
    return Tnp
# In[7]  
#def diffHT(a,al,d,q):
    Ti2 = hTransformation(a,al,d,q)
#    dTi2a  = Ti2.diff(a)
    print(Ti2.diff(al))
#    dTi2d  = Ti2.diff(d)
#    dTi2q  = Ti2.diff(q)
#    return dTi2a,dTi2al,dTi2d,dTi2q
# In[8]
def diffTa():
    Ta = np.matrix([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    return Ta
# In[9]
def diffT_al(a,al,d,q):
    Tal = np.array([[0.0, 0.0, 0.0,0.0],
                     [-1*np.sin(al)*np.sin(q), -1*np.sin(al)*np.cos(q), -1*np.cos(al), -d*np.cos(al)],
                     [np.sin(q)*np.cos(al), np.cos(al)*np.cos(q), -1*np.sin(al), -d*np.sin(al)],
                     [0.0, 0.0, 0.0, 0.0]], dtype = np.float32)
    return Tal
# In[10]
def diffT_d(a,al,d,q):
    Td = np.matrix([[0.0, 0.0, 0.0, 0.0], 
                    [0.0, 0.0, 0.0, -1*np.sin(al)], 
                    [0.0, 0.0, 0.0, np.cos(al)], 
                    [0.0, 0.0, 0.0, 0.0]])
    return Td
# In[11]
def diffT_q(a,al,d,q):
    Tq = np.matrix([[-1*np.sin(q), -1*np.cos(q), 0.0, 0.0], 
                     [np.cos(al)*np.cos(q), -1*np.sin(q)*np.cos(al), 0.0, 0.0],
                     [np.sin(al)*np.cos(q), -1*np.sin(al)*np.sin(q), 0.0, 0.0], 
                     [0.0, 0.0, 0.0, 0.0]])
    return Tq
# In[12]
def all_Diff(param1,param2):
    
    d_all = [] #np.zeros((24,4), dtype = np.float32)
#    print(d_all[0,:])
    k=0
    for i in range(0,6):
       
        d_all.append(diffTa())
        d_all.append(diffT_al(param1["a" + str(i)],param1["al" +str(i)],param1["d"+ str(i+1)],param2["q"+str(i+1)]))
        d_all.append(diffT_d(param1["a" + str(i)],param1["al" +str(i)],param1["d"+ str(i+1)],param2["q"+str(i+1)]))
        d_all.append(diffT_q(param1["a" + str(i)],param1["al" +str(i)],param1["d"+ str(i+1)],param2["q"+str(i+1)]))

    return d_all

# In[14] 
def FKI(param1,param2,Tool):
    tool = np.identity(4)
    
    if Tool == "Tool_1":
#       tool[0:3,-1] = [-1.63,1.43,60.0] 
       tool[0,3] = -1.63 
       tool[1,3] = 1.43  
       tool[2,3] = 60.0   
    elif Tool == "Tool_2":
#        tool[0:3,-1] = [-0.23,0.84,54.92]
       tool[0,3] = -0.23
       tool[1,3] = 0.84  
       tool[2,3] = 54.92  

    npX = np.matrix([0.0,0.0,0.0])

    T0_1 = npHT(param1["a0"],param1["al0"],param1["d1"],param2["q1"])
    T1_2 = npHT(param1["a1"],param1["al1"],param1["d2"],param2["q2"])
    T2_3 = npHT(param1["a2"],param1["al2"],param1["d3"],param2["q3"])
    T3_4 = npHT(param1["a3"],param1["al3"],param1["d4"],param2["q4"])
    T4_5 = npHT(param1["a4"],param1["al4"],param1["d5"],param2["q5"])
    T5_6 = npHT(param1["a5"],param1["al5"],param1["d6"],param2["q6"])
    
    all_T=[T0_1,T1_2,T2_3,T3_4,T4_5,T5_6,tool]

    npT06 = T0_1*T1_2*T2_3*T3_4*T4_5*T5_6*tool

    
    npX =npT06[0:3,-1]

    return npX,all_T
# In[15]
def Jacobian(param1,param2,Tool):
    jac = np.zeros((3,24), dtype = np.float32)
#    ls = [24]
    
    X,allTrans = FKI(param1,param2,Tool)
   
    df_all = all_Diff(param1,param2)

    
    for i in range(0,4):
        for k in range(0,3):
            jac[k,i]    = (df_all[i]*allTrans[1]*allTrans[2]*allTrans[3]*allTrans[4]*allTrans[5]*allTrans[6])[k,3]
            jac[k,i+4]  = (allTrans[0]*df_all[i+4]*allTrans[2]*allTrans[3]*allTrans[4]*allTrans[5]*allTrans[6])[k,3]
            jac[k,i+8]  = (allTrans[0]*allTrans[1]*df_all[i+8]*allTrans[3]*allTrans[4]*allTrans[5]*allTrans[6])[k,3]
            jac[k,i+12] = (allTrans[0]*allTrans[1]*allTrans[2]*df_all[i+12]*allTrans[4]*allTrans[5]*allTrans[6])[k,3]
            jac[k,i+16] = (allTrans[0]*allTrans[1]*allTrans[2]*allTrans[3]*df_all[i+16]*allTrans[5]*allTrans[6])[k,3]
            jac[k,i+20] = (allTrans[0]*allTrans[1]*allTrans[2]*allTrans[3]*allTrans[4]*df_all[i+20]*allTrans[6])[k,3]
    
    return X,jac      

# In[17]
def changeConfiguration(param, refq,deltaq):
#    k=0

    for i in range(0,6):
         param["q" + str(i+1)]  =  refq[i] + deltaq["q" + str(i+1)]
#    print(param) 
    return

# In[18]
def updateParameters(param1,param2, dp,lr,model):# update paremeter after each iteration p += dp
    
    k=0 
#    print(dp.shape)
    dpk = np.zeros((24,1), dtype = np.float32)
    dpk[4:7]   = dp[0:3]
    dpk[8:10]   = dp[3:5]
    dpk[12:15]  = dp[5:8]
    dpk[16:19] = dp[8:11]
    dpk[20:23] = dp[11:14]
    
#    dpk[20:23] = dp[15:18]
#    dpk[22:24] = dp[21:23] 
    if model=="M1":
        
        for i in range(0,6):

            param1["a"+str(i)]    = param1["a"+str(i)]    + lr*dp[k,0]
            param1["al"+str(i)]   = param1["al"+str(i)]   + lr*dp[k+1,0]
            param1["d"+str(i+1)]  = param1["d"+str(i+1)]  + lr*dp[k+2,0]

            param2["q"+str(i+1)]  = param2["q"+str(i+1)]  + lr*dp[k+3,0]
            k+=4
    elif model=="M2":
        for i in range(0,6):
            param1["a"+str(i)]    = param1["a"+str(i)]    + lr*dpk[k,0]
            param1["al"+str(i)]   = param1["al"+str(i)]   + lr*dpk[k+1,0]
            param1["d"+str(i+1)]  = param1["d"+str(i+1)]  + lr*dpk[k+2,0]

            param2["q"+str(i+1)]  = param2["q"+str(i+1)]  + lr*dpk[k+3,0]
#        print(dpk[21])
            k+=4
 
    return    

# In[18]
def copyJacobian(inJ,Jc):
    Jc = np.zeros((3,24), dtype = np.float32)
    for i in range(0,3):
        for j in range(0,24):
            Jc[i,j] = inJ[i,j]
            if(abs(Jc[i,j]) < 1e-6):
                Jc[i,j] = 0.0
    return 

# In[19]
def observIndex(J):
    
    n = J.shape[0]/3
    obndx = 0
#    _,J = Xr_and_Jacobian(param1,param2)
    l = J.shape[1]
    u,sg,vh = np.linalg.svd(J)
    
    if ondexType == "O_1":
        obndx = np.power(np.prod(sg),(1/l))/np.sqrt(n)
    elif ondexType == "O_2":
        obndx = np.amax(sg)/np.amin(sg)  # 1/ condition number of J
    elif ondexType == "O_3":
        obndx = np.amin(sg)
    elif ondexType == "O_4":
        obndx = np.amin(sg)*np.amin(sg)/np.max(sg)
    return obndx
    
#    return np.power(np.prod(sg),1/l)/np.sqrt(n)
# In[20]
def compute_cost(AL, Y):
    """
    Implement the cost function defined as J = e*e^T.

    Arguments:
    AL -- Computed TCP Position(vector [x,y,z])
    Y --  Measured TCP position

    Returns:
    cost -- cross-entropy cost
    """ 
 
    m = Y.shape[0]/3
#    print(m)
    # Compute loss from aL and y.
#    cost = (1./m) * np.sum((-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)))
    cost = (1./m)*np.sum(np.dot((AL-Y).T,(AL-Y)))#(np.square((AL-Y)))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
   
    assert(cost.shape == ())
    
    return cost
 
# In[21]
"""
this routine is to select a calibiration configuration from avialable pools of measured data
compute Jacobian of each configuration
agrigate the avvalible jocobian and check observability of the aggrigate
if Observability index increase keep the point of decrease remove it 

"""
#def calibConfigSelect(nominalParameters,measuredConfig,Tool):
#    nominal_parameters ={}
#    temp_agr_J = np.zeros((1,24), dtype = np.float32)
#    prev_obIndx = 0.0
#    Agr_J = np.zeros((1,24), dtype = np.float32)
   
#    configPool = []  # keep track of selected measuring point 
#    pooL = [k for k in range(measuredConfig.shape[1])]
    
#    for ii in pooL:
        
#        changeConfiguration(jointConfig,measuredConfig[:,ii],jointdq)
        
#        _, J         = Jacobian(nominalParameters,jointConfig,Tool)
        
#        temp_agr_J   = np.vstack((Agr_J,J))
#        print(temp_agr_J.shape)
#        s          = temp_agr_J.shape[0]
#        temp_agr_J = temp_agr_J[1:s,:]
#        print(temp_agr_J.shape)
#        obIndx       = observIndex(temp_agr_J)
        
#        if(obIndx > prev_obIndx):
#            configPool.append(ii)
#           Agr_J = np.vstack((Agr_J,J))
#            pooL.remove(ii)
#       prev_obIndx = obIndx
#        print(obIndx)

#    return configPool,pooL

def calibConfigSelect(nominalParameters,measuredConfig,initialPool,al_Pool,Tool,num_Itration):

    prev_obIndx = 0.0
    obIndx = 0.0
 
    pooL = [k for k in range(measuredConfig.shape[1])]
    
    _,Jr = Xr_and_Jacobian(nominalParameters,jointConfig,measuredConfig,jointdq,Tool,initialPool)
    
    prev_obIndx = observIndex(Jr,"O_1")
    print(prev_obIndx)
    
    
    for x in initialPool:
        if x in al_Pool:
            al_Pool.remove(x)
   
# iterate of measured configuration to selct optimal calibration poses 
    for i in range(num_Itration):
#        initialPool.sort()
        for jj in initialPool:
            tempL = initialPool.copy()
            tempL.remove(jj)
            _,J = Xr_and_Jacobian(nominalParameters,jointConfig,measuredConfig,jointdq,Tool,tempL)
        
            obIndx = observIndex(J,"O_1")
            if (obIndx > prev_obIndx):
                prev_obIndx = obIndx
                initialPool.remove(jj)
#                for kk in al_Pool:
#                    tempL = initialPool.copy()
#                    tempL.append(kk)
#                    _,J    = Xr_and_Jacobian(nominalParameters,jointConfig,measuredConfig,jointdq,Tool,tempL)
#                    obIndx = observIndex(J,"O_1")
#                    if (obIndx > prev_obIndx):
#                        prev_obIndx = obIndx
                print(obIndx)
                print("list Updated")
#            else:
#                print("not Good choice")
#            print(obIndx)
#        initialPool.sort()#            print(initialPool)
        for kk in al_Pool:
            tempL = initialPool.copy()
            tempL.append(kk)
            _,J = Xr_and_Jacobian(nominalParameters,jointConfig,measuredConfig,jointdq,Tool,tempL)
            obIndx =  observIndex(J,"O_1")
            if (obIndx > prev_obIndx):
                prev_obIndx = obIndx
                print("updateList")
                initialPool.append(kk)
#                al_Pool.remove(kk)

#            else:
#            print(obIndx)
        print('For number of Itration {0} observability index is{1}.'.format(i,prev_obIndx))
        
    return prev_obIndx,initialPool,al_Pool
# In[22]

#        
#    obInd = observIndex(J)
    
#    configPool = [xyzVec[3],qVec[6]]

