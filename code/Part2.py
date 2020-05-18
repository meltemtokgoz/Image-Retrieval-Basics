import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans 
import functions
from matplotlib import pyplot as plt

#PART2    
###############################################################################   
#1.STEP CODEBOOK

def sift_desc_vector():
    sift_vector=[]
    for a in functions.train_img_names:
        t_img = cv2.imread(a)  
        t_gray = functions.ToGray(t_img)
        kp, desc_t, desc_vector_t = functions.sift_features(t_gray)
        sift_vector.extend(desc_t)    #nx128 shıft vector 
        #print(sift_vector)
    return sift_vector


def codebook(k, des):
    km = KMeans(n_clusters=k)
    km.fit(des)
    codebook = km.cluster_centers_
    label = km.labels_
    return codebook, label
    
  
###############################################################################
#2.STEP QUANTIZATION

def histogram_query(codebk):
    all_qHistogram = []
    for i in functions.query_img_names:
        histo_q=[]
        img = cv2.imread(i)  
        gray = functions.ToGray(img)
        kp, desc, desc_vector = functions.sift_features(gray) 
        hist = np.zeros(k, dtype = int)   # create zero array
        for d in range(len(desc)):
            weight = {}
            for m in range(len(codebk)):
                sim = distance.euclidean(desc[d],codebk[m])
                weight[m] = sim
            find = min(weight, key=lambda f: weight[f]) 
            hist[find] = hist[find] +1    #closest codebook value decrease 1
        histo_q.append(hist)              #histogram vector each query image
        histo_q.append(i[14:22])           #name each query image
        all_qHistogram.append(histo_q)    #all query histograms
    return all_qHistogram
    
def histogram_train(codebk):
    all_tHistogram = []
    for i in functions.train_img_names:
        histo_t = []
        name = i.split('\\') 
        img = cv2.imread(i)  
        gray = functions.ToGray(img)
        kp, desc2, desc_vector = functions.sift_features(gray) 
        hist2 = np.zeros(k, dtype = int) 
        for d in range(len(desc2)):
            weight = {}
            for m in range(len(codebk)):
                sim = distance.euclidean(desc2[d],codebk[m])
                weight[m] = sim
            find = min(weight, key=lambda f: weight[f]) 
            hist2[find] = hist2[find] +1    #closest codebook value decrease 1
        histo_t.append(hist2)               #histogram vector each train image
        histo_t.append(name[2][0:8])        #name each train image
        all_tHistogram.append(histo_t)      #all train histograms
    return all_tHistogram
    
###############################################################################
#3.STEP HISTOGRAM AND ACCURACY

#apply knn histograms vector between train and query

def accuracy_bow(all_qHistogram,all_tHistogram):
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
    for q in range(len(all_qHistogram)):
        weight = {}
        for t in range(len(all_tHistogram)):
            dist = distance.euclidean(all_qHistogram[q][0], all_tHistogram[t][0])
            weight[all_tHistogram[t][1]] = dist
        total_weight[all_qHistogram[q][1]] = weight
        find_label = min(weight, key=lambda k: weight[k]) #return smallest distance category
        real_label = all_qHistogram[q][1]
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
    print("BoW Average Accuracy",avg_acc)  
    
    #class-based accuracy 
     
    bear_acc =  bear_true / 5   
    print("BoW Bear Class Accuracy",bear_acc) 
    
    butterfly_acc =butterfly_true / 5
    print("BoW Butterfly Class Accuracy",butterfly_acc)
    
    coffee_acc =coffee_true / 5
    print("BoW Coffee-mug Class Accuracy",coffee_acc)
    
    elk_acc =elk_true / 5
    print("BoW Elk Class Accuracy",elk_acc)
    
    fire_acc =fire_true / 5
    print("BoW Fire-truck Class Accuracy",fire_acc)
    
    horse_acc =horse_true / 5
    print("BoW Horse Class Accuracy",horse_acc)
    
    balloon_acc =balloon_true / 5
    print("BoW Hot-air-balloon Class Accuracy",balloon_acc)
    
    iris_acc =iris_true / 5
    print("BoW Iris Class Accuracy",iris_acc)
    
    owl_acc =owl_true / 5
    print("BoW Owl Class Accuracy",owl_acc)
    
    teapot_acc =teapot_true / 5
    print("BoW Teapot Class Accuracy",teapot_acc)
    
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
    
    k = 100           # k parameter in k-means
    sift_vector = sift_desc_vector()
    codebk, label = codebook(k,sift_vector)
        
    param1 = histogram_query(codebk)
    param2 = histogram_train(codebk)
    total_weight = accuracy_bow(param1,param2)
    
    
    ex_query =["dataset/query/107_0041","dataset/query/072_0019","dataset/query/024_0005"]
    five_result = example_query(param1,param2,ex_query,total_weight)
    