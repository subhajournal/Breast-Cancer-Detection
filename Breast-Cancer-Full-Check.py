
# coding: utf-8

# In[6]:


#Algorithm to check Logistic Regression for Breast Cancer
from statistics import mean


# In[7]:


def TruePlot(lst1,lst2,data):
    import matplotlib.pyplot as plt
    import numpy as np
    corx=lst1.index(1)
    cory=lst2[168]      #mean value list
    fig = plt.figure(figsize=(20,10))
    plt.title('MEAN-PROBABILITY(=1) PLOT')
    plt.plot(lst1[168:data],"r",label='Malignant Probability=1')
    plt.plot(lst2[168:data],"y",label='Mean value of each instance')
    plt.plot(lst2[168:data],"*m",label='Mean Peak')
    plt.legend(loc='lower right')
    plt.xlim(0,175,5)
    plt.ylim(0, 10,0.2)
    plt.xlabel('INSTANCE OR SAMPLE DATA')
    plt.ylabel('MEAN OF INSTANCE')
    plt.xticks(np.arange(0,170,5))
    plt.yticks(np.arange(0,10,0.2))
    plt.grid()
    plt.savefig("D:/As a Trainer/OGMA/Python/Study Material Folder/Problem Set/Breast Cancer/Mean_vs_Probability(1)")
    plt.show()


# In[8]:


def PrintData(data,lst):
    for i in range(data):
        print("Index no:"+str(i)+" "+"Value: "+str(lst[i])+"\n")    #lst=mn[]
 


# In[9]:


def FalseData(lst1,lst2,data):
    import matplotlib.pyplot as plt
    import numpy as np
    corx=lst1.index(1)
    cory=lst2[168]      #mean value list
    fig = plt.figure(figsize=(20,10))
    plt.title('MEAN-PROBABILITY PLOT')
    plt.plot(lst1[0:167],"g",label='Malignant Probability<1')
    plt.plot(lst2[0:167],"c",label='Mean value of each instance')
    plt.plot(lst2[0:167],"*k",label='Mean Peak')
    plt.legend(loc='lower right')
    plt.xlim(0,175,5)
    plt.ylim(0, 10,0.2)
    plt.xlabel('INSTANCE OR SAMPLE DATA')
    plt.ylabel('MEAN OF INSTANCE')
    plt.xticks(np.arange(0,170,5))
    plt.yticks(np.arange(-1,10,0.2))
    plt.grid()
    plt.savefig("D:/As a Trainer/OGMA/Python/Study Material Folder/Problem Set/Breast Cancer/Mean_vs_Probability(0)")
    plt.show()


# In[10]:


def BrcnCheck(file):
    import numpy as np
    r,c=brcn.shape
    ins = []
    insm=[]
    malig=brcn['Malignancy'].tolist()
    #print(malig)
    for i in range(r):
        part=[]
        part=brcn.ix[i].tolist()
        del part[-1]
        ins.append(part)
    mxlst=[]
    for i in range(r):
        mx=max(ins[i])
        mxlst.append(mx)
    mn=[]
    for i in range(r):
        mnval=mean(ins[i])
        mn.append(mnval)
    seek=input("Want to Print Training data?\n Press y to Show \n Press n to Skip: \n")
    if seek=='y' or seek=='Y':
        PrintData(r,mn)
    mnmn=mean(mn)
    fltrmng=[]
    fltrmng=[x for x in mn if x > mnmn]
    fltrmnl=[]
    fltrmnl=[x for x in mn if x < mnmn]
    len(fltrmng)
    len(fltrmnl)
    seek=input("Want to Visualize True Malignancy?\n Press y to Show \n Press n to Skip: \n")
    if seek=='y' or seek=='Y':
        TruePlot(malig,mn,r)
    seek=input("Want to Visualize False Malignancy?\n Press y to Show \n Press n to Skip: \n")
    if seek=='y' or seek=='Y':
        FalseData(malig,mn,r)
    if len(fltrmng)>len(fltrmnl):
        print("Malignancy Positive!!")
    else:
        print("Malignancy Negetive!!Smile!!")


# In[ ]:


import pandas as pd

brcn = pd.read_excel("D:/As a Trainer/OGMA/Python/Study Material Folder/Problem Set/Breast Cancer/breastcancer_training.xlsx")
BrcnCheck(brcn)

