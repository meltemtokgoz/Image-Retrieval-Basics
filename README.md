All my .py file codes are in the folder named code.
The layout between the .py files is as follows.

-code
	-Part1_a.py
	-Part1_b.py
	-Part2.py
	-functions.py

Dataset layout

-dataset
	-train
		-10 category file (bear...)
	-query
	
As the names indicate, each part's code is independent of each other and works separately.
With direct run can run every .py file.
The functions.py file contains all of the common functions and codes in the .py files side.
(I did this in terms of not having a repetition of code.)
When you run the code in the console, closest 5 images are shown for the sample 3 query picture each algorithm.

****************************
functions.py content

In this function, all the query and training name of pictures in the data set are added to the arrays. 
These arrays are used in other .py files.

There are 3 functions required to use SIFT.
The functions found are:
	-ToGray(color_img)
	-sift_features(gray_img)
	-show_sift_features(gray_img, color_img, kp)

These functions are used by Part1_b.py and Part2.py.

***************************

Part1_a.py content

Here, I have calculated the accuracy, by applying of knn algorithm 
for 1x40 vectors that I have created 40 gabor filters for each image.
	
The functions used are as follows:
	-build_filters()
	-process(img, filters)

Here, the funtions.py module is called from the inside.
It can be operated with direct run button without needing an extra parameter.

***************************

Part1_b.py content

Here, I have calculated the accuracy, by applying of knn algorithm  
for SIFT descriptor vector for each image.

No functions were written. 
Functions called from function.py module were used.
It can be operated with direct run button without needing an extra parameter.

***************************

Part2.py content

Accuracy was calculated using the BoW method here.

Here is the function by written:
	-codebook(k, des)

Functions called from function.py module were used.
The K parameter is defined in the module with k variable and can be changed.
It can be operated with direct run button without needing an extra parameter.

***************************

Implementation Info

I've used Python 3.5 also I wrote my code in Anaconda Spyder

I've only used built-in functions for creating a gabor filter, convolution, 
extracting SIFT identifiers, k-tool aggregation, and distance calculation. 
I've written my own functions for other steps.

I'm reading all the images in the code from a folder called  "dataset"  
and printing the results to the console.

