import cv2 
import numpy as np
from scipy import ndimage
from scipy.spatial import distance 
from matplotlib import pyplot as plt
import functions
#PART a K-NN with Gabor Filter
#******************************************************************************
#This function creates 40 gabor filters

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 40):
        params = {'ksize':(ksize, ksize), 'sigma':20, 'theta':theta, 'lambd':10.0,
                  'gamma':0.50, 'psi':0 }
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters #filter list
    
#******************************************************************************
#This fuction applying gabor filter with convolutional

def process(img, filters):
    fourty_mean=[]
    for kern in filters:
        f_img = ndimage.convolve(img, kern, mode='constant', cval=1.0) #apply convolution
        #plt.imshow(f_img)               #see filter apply in image
        #plt.show()
        mean_img = np.mean(f_img)       #mean for each filter image
        fourty_mean.append(mean_img)
    return fourty_mean                  #1x40 vector
    
#******************************************************************************
#Here, found 1x40 vector for train image

def train_gabor(filters):
    train_forty_gabor =[]
    for a in functions.train_img_names:
        t_vector =[]
        name = a.split('\\')
        e = cv2.imread(a,0)                   #read image
        forty_vector = process(e, filters)    #process filter and return mean 40 filter 
        t_vector.append(forty_vector)         #append each image 40 vector
        t_vector.append(name[2][0:8])         #append each image category
        #print(t_vector[0])                   # see 40 vector each image
        #print(t_vector[1])                   # see category each image
        train_forty_gabor.append(t_vector)
        #print(train_forty_gabor[0][0])       # 0x40 vector
        #print(train_forty_gabor[0][1])       # label
    return train_forty_gabor

#same operation with query dataset

def query_gabor(filters):
    query_forty_gabor=[]
    for b in functions.query_img_names:
        q_vector =[]
        e = cv2.imread(b,0)	                 #read image
        forty_vector2 = process(e, filters)   #process filter and return mean 40 filter 
        q_vector.append(forty_vector2)        #append each image 40 vector
        q_vector.append(b[14:22])             #append each image category
        query_forty_gabor.append(q_vector)
    return query_forty_gabor
    

#******************************************************************************
#now I applying knn


def accuracy_gabor(train_forty_gabor,query_forty_gabor):
    total_true = 0
    
    bear_true = 0 #009
    butterfly_true = 0 #024
    coffee_true = 0 #041
    elk_true = 0 #065
    fire_true = 0 #072
    horse_true = 0 #105
    balloon_true = 0 #107
    iris_true = 0 #118
    owl_true = 0 #152
    teapot_true = 0 #212
    total_weight = {}
    for q in range(len(query_forty_gabor)):
        weight = {}
        for t in range(len(train_forty_gabor)):
            dist = distance.euclidean(query_forty_gabor[q][0], train_forty_gabor[t][0])
            weight[train_forty_gabor[t][1]] = dist
        total_weight[query_forty_gabor[q][1]] = weight
        find_label = min(weight, key=lambda k: weight[k]) #return smallest distance category
        real_label = query_forty_gabor[q][1]
        if find_label[0:3] == real_label[0:3]:
            total_true = total_true+1
            if real_label[0:3] == '009':
                bear_true += 1
            if real_label[0:3] == '024':
                butterfly_true += 1
            if real_label[0:3] == '041':
                coffee_true += 1
            if real_label[0:3] == '065':
                elk_true += 1
            if real_label[0:3] == '072':
                fire_true += 1
            if real_label[0:3] == '105':
                horse_true += 1
            if real_label[0:3] == '107':
                balloon_true += 1
            if real_label[0:3] == '118':
                iris_true += 1
            if real_label[0:3] == '152':
                owl_true += 1
            if real_label[0:3] == '212':
                teapot_true += 1
                
    #avarage accuracy
    
    avg_acc = total_true / 50
    print("Gabor Filter Average Accuracy",avg_acc)  
    
    #class-based accuracy 
     
    bear_acc =  bear_true / 5   
    print("Gabor Filter Bear Class Accuracy",bear_acc) 
    
    butterfly_acc =butterfly_true / 5
    print("Gabor Filter Butterfly Class Accuracy",butterfly_acc)
    
    coffee_acc =coffee_true / 5
    print("Gabor Filter Coffee-mug Class Accuracy",coffee_acc)
    
    elk_acc =elk_true / 5
    print("Gabor Filter Elk Class Accuracy",elk_acc)
    
    fire_acc =fire_true / 5
    print("Gabor Filter Fire-truck Class Accuracy",fire_acc)
    
    horse_acc =horse_true / 5
    print("Gabor Filter Horse Class Accuracy",horse_acc)
    
    balloon_acc =balloon_true / 5
    print("Gabor Filter Hot-air-balloon Class Accuracy",balloon_acc)
    
    iris_acc =iris_true / 5
    print("Gabor Filter Iris Class Accuracy",iris_acc)
    
    owl_acc =owl_true / 5
    print("Gabor Filter Owl Class Accuracy",owl_acc)
    
    teapot_acc =teapot_true / 5
    print("Gabor Filter Teapot Class Accuracy",teapot_acc)
    
    return total_weight
#******************************************************************************
 
 
def example_query(train_mean_desc,query_mean_desc,ex_query,total_weight):
    for x in ex_query:
        name = x.split('/')
        print("Image closest 5 image",name[2][0:8])
        min_five(total_weight[name[2][0:8]])
    
def min_five(dic):
    result =[]
    for i in range(5):
        key_min = min(dic.keys(), key=(lambda y: dic[y]))
        result.append(key_min)
        del dic[key_min]
    print (result)
    read_ex_query(result)
    

def read_ex_query(five_result):
     for i in five_result:
         for a in functions.train_img_names:
             name = a.split('\\')
             if name[2][0:8] == i:
                    e = cv2.imread(a)
                    plt.imshow(e)               
                    plt.show()   
        
          
#******************************************************************************  
if __name__ == '__main__':
    
    filters = build_filters()             #build filter
    
    param1 = train_gabor(filters)
    param2 = query_gabor(filters)
    
    total_weight =  accuracy_gabor(param1,param2)
    
    ex_query =["dataset/query/107_0041","dataset/query/072_0019","dataset/query/024_0005"]
    example_query(param1,param2,ex_query,total_weight)
    
    
    