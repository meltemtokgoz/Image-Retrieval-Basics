import cv2 
import numpy as np
from scipy import ndimage
from scipy.spatial import distance 
from matplotlib import pyplot as plt
import functions
#PART b K-NN with SIFT
#******************************************************************************

def train_sift_mean():
    train_mean_desc = []
    for a in functions.train_img_names:
        t_vector = []                          
        name = a.split('\\')
        t_img = cv2.imread(a)  
        t_gray = functions.ToGray(t_img)
        kp, desc_t, tmean_desc = functions.sift_features(t_gray)
        t_vector.append(tmean_desc)          #1x128 vector each train image
        t_vector.append(name[2][0:8])        #name each train image
        train_mean_desc.append(t_vector)     #[1x128 vector,name] for train all image
    return train_mean_desc
   
def query_sift_mean():
    query_mean_desc = []
    for b in functions.query_img_names:
        q_vector = []
        q_img = cv2.imread(b)  
        q_gray = functions.ToGray(q_img)
        kp, desc_q, qmean_desc = functions.sift_features(q_gray)    
        q_vector.append(qmean_desc)
        q_vector.append(b[14:22])
        query_mean_desc.append(q_vector)
        #functions.show_sift_features(q_gray, q_img, kp);
        #plt.show()
    return query_mean_desc

#print(query_mean_desc[0][0]) #1x128 vector
#print(query_mean_desc[0][1]) #name image

#******************************************************************************
#knn for shift

def accuracy_sift(train_mean_desc,query_mean_desc):
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
    for q in range(len(query_mean_desc)):
        weight = {}
        for t in range(len(train_mean_desc)):
            dist = distance.euclidean(query_mean_desc[q][0], train_mean_desc[t][0])
            weight[train_mean_desc[t][1]] = dist
        total_weight[query_mean_desc[q][1]] = weight
        find_label = min(weight, key=lambda k: weight[k]) #return smallest distance category
        real_label = query_mean_desc[q][1]
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
    print("SIFT Average Accuracy",avg_acc)  
    
    #class-based accuracy 
     
    bear_acc =  bear_true / 5   
    print("SIFT Bear Class Accuracy",bear_acc) 
    
    butterfly_acc =butterfly_true / 5
    print("SIFT Butterfly Class Accuracy",butterfly_acc)
    
    coffee_acc =coffee_true / 5
    print("SIFT Coffee-mug Class Accuracy",coffee_acc)
    
    elk_acc =elk_true / 5
    print("SIFT Elk Class Accuracy",elk_acc)
    
    fire_acc =fire_true / 5
    print("SIFT Fire-truck Class Accuracy",fire_acc)
    
    horse_acc =horse_true / 5
    print("SIFT Horse Class Accuracy",horse_acc)
    
    balloon_acc =balloon_true / 5
    print("SIFT Hot-air-balloon Class Accuracy",balloon_acc)
    
    iris_acc =iris_true / 5
    print("SIFT Iris Class Accuracy",iris_acc)
    
    owl_acc =owl_true / 5
    print("SIFT Owl Class Accuracy",owl_acc)
    
    teapot_acc =teapot_true / 5
    print("SIFT Teapot Class Accuracy",teapot_acc)
    
    
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
    
    param1 = train_sift_mean()
    param2 = query_sift_mean()
    
    total_weight = accuracy_sift(param1,param2)
    
    ex_query =["dataset/query/107_0041","dataset/query/072_0019","dataset/query/024_0005"]
    example_query(param1,param2,ex_query,total_weight)

    