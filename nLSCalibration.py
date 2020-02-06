# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:29:40 2018
Version: 0.0.01
@author: e10509
"""

# In[1]
import timeit
import matplotlib.pyplot as plt
import numpy as np
#import tkinter
import pandas as pn

from utilities import *

#import xlsxwriter 
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#plt.style.use('presentation')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)
# In[1]
trainData = pn.read_excel('sample.xlsx')
testData  = pn.read_excel('sample2.xlsx')
#print the column names
print (trainData.columns)

# In[3]:
m = len(trainData['X'])
m1 = len(testData['X'])
numParameters = 24 # number of parameters

trainq_ref  = np.zeros((6,m), dtype = np.float32)
testq_ref   = np.zeros((6,m1), dtype = np.float32)


trainq_ref[0,:] = (trainData['A1'].values)
trainq_ref[1,:] = (trainData['A2'].values) 
trainq_ref[2,:] = (trainData['A3'].values)
trainq_ref[3,:] = (trainData['A4'].values)
trainq_ref[4,:] = (trainData['A5'].values)
trainq_ref[5,:] = (trainData['A6'].values)

trainq_ref = trainq_ref*D2R      #(q_ref*PI)/180  
#print(q_ref[0,:])
#print(q_ref[:,0])
testq_ref[0,:] = testData['A1'].values
testq_ref[1,:] = testData['A2'].values
testq_ref[2,:] = testData['A3'].values
testq_ref[3,:] = testData['A4'].values
testq_ref[4,:] = testData['A5'].values
testq_ref[5,:] = testData['A6'].values

tQ_ref = D2R*testq_ref
print(testq_ref.shape)
# In[4]
P_n = Matrix([a[0],alp[0],d[1],q[1],a[1],alp[1],d[2],q[2],a[2],alp[2],d[3],q[3],
              a[3],alp[3],d[4],q[4],a[4],alp[4],d[5],q[5],a[5],alp[5],d[6],q[6]])

# In[5]
def measuredX(data):
    l = len(data['X'])
    mX = np.zeros((3*l,1), dtype = np.float32)
    countt = 0
    for j in range(0,l):
        mX[countt,0]   = (data['X'].values)[j]  #val_mX[j]
        mX[countt+1,0] = (data['Y'].values)[j]    #val_mY[j]
        mX[countt+2,0] = (data['Z'].values)[j]     #val_mZ[j]
        countt +=3
    return mX
# In[6]
#mx = measuredX(trainData)
#print(mx[0:3,:])

# In[6]
def Xr_and_Jacobian(nParam,jointConfig,theta_ref,dq,Tool):
    l      = theta_ref.shape[1] 
    count  = 0
    agrX   = np.zeros((1,1), dtype = np.float32)
    agrH   = np.zeros((1,24), dtype = np.float32)
    m      = (l*3) + 1
    for j in range (0,l):

        changeConfiguration(jointConfig,theta_ref[:,j],dq)
#        print("this is qref:" + str(q_ref[:,j]))
        xr,jq = Jacobian(nParam,jointConfig,Tool)  # joint configuration is substittuted
        
        agrH = np.concatenate((agrH,jq),axis = 0)#np.vstack((agrH,jq))
        agrX = np.concatenate((agrX,xr),axis = 0)#np.vstack((agrX,xr))
        
        
#        agrX[count,0]    = xr[0,0]
#        agrX[count+1,0]  = xr[1,0]
#        agrX[count+2,0]  = xr[2,0]
#        
##        obIndx.append(observIndex(jq))
#        agrH[count:count+3,:] = jq
#        count +=3
#    print(agrH[-1,:])
    return agrX[1:m],agrH[1:m,:]
# In[7]
#xr,Jr =Xr_and_Jacobian(nTestParam,jointConfig,tQ_ref,jointdq,"Tool_2")
pList,pol = calibConfigSelect(nTestParam,tQ_ref,'Tool_2')
print(pList)
print(pol)


lp = [x for x in range(10)]
print(lp)
#print(Jr[-1,:])
#print(xr.shape)
#plt.plot(np.squeeze(obIn))
#plt.ylabel('ObservIndx')
#plt.xlabel('test Configuration')
#plt.title('observability Index')
#plt.show() 
    
# In[8]
    
def A_and_DJ(nParam,jointConfig,theta_ref,dq,Tool):
    l      = theta_ref.shape[1]
    J_dna = np.zeros((3*l,14), dtype = np.float32)
    
    Xr,Jr = Xr_and_Jacobian(nParam,jointConfig,theta_ref,dq,Tool)
#    J_dna[:,0:20]  = Jr[:,4:24]
    
    J_dna[:,0:3]   = Jr[:,4:7]
    J_dna[:,3:5]   = Jr[:,8:10]
    J_dna[:,5:8]   = Jr[:,12:15] 
    J_dna[:,8:11]  = Jr[:,16:19] 
    J_dna[:,11:14] = Jr[:,20:23] 
#    J_dna[:,15:18] = Jr[:,20:23] 
#    J_dna[:,21:23] = Jr[:,22:24]
#    J_dna[:,6:19] = Jr[:,11:24]
#    print(J_dna)
#    for k in range(0,12):
#        
#        J_dna[:,k] = Jr[:,2*k]
    
    return Xr,J_dna
    
# In[10]
"""
This fucntion do non-linear leastsquare optimization to calibrate robot kinematic parameter
We plan to try jacobian of position w.r.t kin parameters i.e. J= dP/d(a,alpha,d,theta) 
use sysmbolic python for that purpose. In selecting the measured data point for calibiration
Observability measure is considered.

"""
def nLCalibration_NLS(num_iterations,learning_rate,epislon,print_costs=False):
    
    
    """
    mX    --- measured cartesian position using laser tracker
    n_P  --  nominal kinemartic parameter
    Xr   --- rference position to calibrate on
    qr   --  joint position corresponding to Xr
    lr --- cosntant learning rate to update the kinematics parameter based on least square error
    epislon --- convergence chrateria 
    
    """
#    m = qr.shape[1]
    numParam = 24#14
    costs = []
    detr = []

    mX  = measuredX(trainData)

    pk  = np.zeros((numParam,1), dtype = np.float32)
#    mP  = np.zeros((3,1), dtype = np.float32)
    err = 0
    
    mu = 0.01
    muI = mu*(np.identity(numParam))
    
    prevL_k = 0
    
#    ro = 0.25
#    gama = 0.75

#######################aggrigate Methods(Levenbrg Marquardt-LM)################ 
    for i in range(0,num_iterations):
        if (i==0):
            muI = muI/10.0
#        if (i>=100):
#            mu = 0.001
        startTime = timeit.default_timer()    
        Xr,idfJ= Xr_and_Jacobian(nTestParam,jointConfig,trainq_ref,jointdq,"Tool_1")
        
#        Xr,idfJ = A_and_DJ(nTestParam,jointConfig,trainq_ref[:,0:20],jointdq,"Tool_1")
        condNumber = np.linalg.cond(idfJ)
        er    = (mX - Xr)
#        print(idfJ.shape)
        gk    = -2*((idfJ.transpose()).dot(er))  # gradient of loss function evealuated @
#        print(gk.shape)
        G_k   = 2*((idfJ.transpose()).dot(idfJ) + muI)
#        print(G_k.shape)
        pk    = -1*(np.linalg.inv(G_k)).dot(gk)
#        print(pk.shape)

        updateParameters(nTestParam,jointdq,pk,learning_rate,'M1')
             
#          # compute cost and appened to costs list
        
        L_k = np.sqrt(compute_cost(Xr,mX))

        if (L_k > prevL_k):
            muI = muI*10
#        else:
#            muI = muI/10
        
        prevL_k = L_k
        endTime = timeit.default_timer()
#        print(endTime - startTime)
#        if(np.abs(L_k - prevL_k) <= 1e-3):   #ignore the mu value when solution approach convergance
#            mu = 0.0001
#         
        if print_costs and i % 10 == 0:
            
            print("Cost after iteration {}: {}".format(i, np.squeeze(L_k)))
            detr.append(condNumber)
             
        if print_costs:# and i % 10 == 0:
            
            costs.append(L_k)
            
            
        if(L_k < epislon): 
            return  
    print("Improvment in Accuracy is:"+str((costs[0] - costs[-1])*100/costs[0])+"%")
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()  
    d = {"costs": costs,
         "clibratedParameter": nTestParam, 
         "calibratedJoint" : jointdq, 
         "learning_rate" : learning_rate,
         "determinant_idfJ": detr}
    return d

# In[11]
#startTime = timeit.default_timer()
#
#rd = nLCalibration_NLS(num_iterations=500,learning_rate=0.1,epislon =1e-3,print_costs=True)
#with open('calibParameters.csv', 'w') as f:
#    [f.write('{0},{1}\n'.format(key, value)) for key, value in rd.items()]
#endTime = timeit.default_timer()
#
#print(endTime - startTime)
#
#print(rd["clibratedParameter"])
#print(rd["calibratedJoint"])

# In[12]

"""
This fucntion do linear leastsquare optimization to calibrate robot kinematic parameter
We plan to try jacobian of position w.r.t kin parameters i.e. J= dP/d(a,alpha,d) 
use sysmbolic python for that purpose. In selecting the measured data point for calibiration
Observability measure is considered.

"""
def nLCalibration_LS(num_iterations,learning_rate,epislon,full_parameters = True,print_costs=False):
    
    
    """
    mX    --- measured cartesian position using laser tracker
    n_P  --  nominal kinemartic parameter
    Xr   --- rference position to calibrate on
    qr   --  joint position corresponding to Xr
    lr --- cosntant learning rate to update the kinematics parameter based on least square error
    epislon --- convergence chrateria 
    
    """
#    m = qr.shape[1]

    costs  = []
    ObIndx = []
#    parameters = nParameters

    mX  = measuredX(trainData)
#    Xr,_ = aggregateJacobian(nTestParam,jointConfig,jointdq) # aggregate pose and jacobian
    pk  = np.zeros((numParameters,1), dtype = np.float32)
#    mP  = np.zeros((3,1), dtype = np.float32)
    err = 0
    
#######################aggrigate Methods(Levenbrg Marquardt-LM)################ 
    for i in range(0,num_iterations):
#        if (i==0):
#            muI = muI/10.0
        startTime = timeit.default_timer()
#        if (full_parameters):
        Xr,idfJ = Xr_and_Jacobian(nTestParam,jointConfig,trainq_ref,jointdq,"Tool_1") #aggregateJacobian(nTestParam,jointConfig,jointdq) #idfJ(identification jacobian)
        
        
#        Xr,idfJ = A_and_DJ(nTestParam,jointConfig,trainq_ref,jointdq,"Tool_2")

        er    = (mX - Xr)
#        print(er)

       
        gk    = np.linalg.pinv(idfJ,1e-8)#(np.linalg.inv(G_k)).dot(idfJ)
        pk    = (gk.dot(er))
        
        condNumber = np.linalg.cond(idfJ)

        updateParameters(nTestParam,jointdq,pk,learning_rate,'M1')
#              
###############################################################################         
#          # compute cost and appened to costs list
        
        L_k = np.sqrt(compute_cost(Xr,mX))

        
#        prevL_k = L_k
        endTime = timeit.default_timer()
#        print(endTime - startTime)
        
#         
        if print_costs and i % 100 == 0:
            
            print("Cost after iteration {}: {}".format(i, np.squeeze(L_k)))
#            print(pk[6])
            
        if print_costs:# and i % 10 == 0:
            
            costs.append(L_k)
            ObIndx.append(condNumber)
        if(L_k < epislon): 
            return 
    print("Improvment in Accuracy is:"+str((costs[0] - costs[-1])*100/costs[0])+"%")
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("LS_learning rate =" + str(learning_rate))
    plt.show()  
    d ={"costs":costs,"learning_rate":learning_rate,"calibratedParameters":nTestParam,"JointOffset":jointdq}
    return d

# In[13]
startTime = timeit.default_timer()

parmd = nLCalibration_LS(num_iterations=2000,learning_rate=0.001,epislon =1e-3,full_parameters = True,print_costs=True)

with open('calibParametersLS.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in parmd.items()]

endTime = timeit.default_timer()

print(endTime - startTime) 

print(parmd["calibratedParameters"])
print(parmd["JointOffset"])   

# In[14]
#startTime = timeit.default_timer()
#learning_rates = [0.1, 0.1, 0.1]
#models = {}
#for i in learning_rates:
#    print ("learning rate is: " + str(i))
#    models[str(i)] = nLCalibration_NLS(num_iterations=300,learning_rate=i,epislon =1e-3,print_costs=True)
#    print ('\n' + "-------------------------------------------------------" + '\n')
#
#for i in learning_rates:
#    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
#
#plt.ylabel('cost')
#plt.xlabel('iterations')
#
#legend = plt.legend(loc='upper center', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.show()
#
#endTime = timeit.default_timer()
#
#print(endTime - startTime)

# In[15]
#evalute the performance
cst = []
jConfig = {'q1':9.05880481e-02,
               'q2':-1.37911820e+00,
               'q3':-5.84525391e-02,
               'q4':1.92422551e-04,
               'q5':1.57268214e+00,
               'q6':1.29712690e-02}
qref = {'q1': -0.0019053210949872236, 'q2': -0.0074753436301932463, 'q3': 0.0067879131750221639,
        'q4': 0.0019570135008669587, 'q5': -0.011103828843761083, 'q6': 0.67582795831979081}
#for i in range(0,testq_ref.shape[1]):
#    
#    changeConfiguration(jConfig,testq_ref[:,i],qref)
##    print(jConfig)
#    xr,_ =  FKI(nTestParam,jConfig,"Tool_2")
#    print(xr)
#print(cst)
#print(testq_ref.shape[1])
Xr,_ =  Xr_and_Jacobian(nTestParam,jConfig,trainq_ref,jointdq,"Tool_2")
#print(Xr)
xm   = measuredX(trainData)
print(xm.shape)
cost = np.sqrt(compute_cost(xm,Xr))
#print(xr)
##print(jointdq)
cst.append(nTestParam.values())
print(cost)

#with open('aftercalib.csv', 'w') as f:
#    [f.write('{0}\n,{1}\n'.format(xr, xm))]

# In[16]