# Digits
OCR on digits using computer vision algorithms and SVM learning algorithm

Team members:
Youssef Hussein	Section 1
Kirolos  Lewis	Section 1
Baraa Gamal	Section 1
Mina Maged		Section 1
Aml Mohamed	Section 1
Introduction:
This is an implementation of digits OCR that reads a handwritten digit from an image and outputs the digit in the image using computer vision algorithms and SVM learning algorithm.
Algorithm:
The image is read, enhanced, segmented and then we find its contours, we get HOG features, then we pass the features to a trained SVM classifier that then outputs what the digit in the image is.
Code:
In the CD you will find digits.py which is a code to download dataset and train SVM on it and runDigits.py which takes an image, operates on it and passes it to previously trained SVM.





Sample output:
 
Notes:
1 - Digits.py was used to train the SVM and save it, if the script was run again it will download the dataset, train the SVM and save it again. We recommend that runDigits.py is the script to be run.
2 – Some sample images are present in the source code folder to test on, to change what image to be the input, put its directory in line 8 in runDigits.py