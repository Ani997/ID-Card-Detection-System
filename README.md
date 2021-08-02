# ID-Card-Detection-System
In the current scenario, a receptionist has to manually enter information of visitors from the Id Cards at the reception of buildings/offices. 
This manual approach is a time consuming and leads to long queues in front of the reception. 
To counter this problem, a visitor sign-in system should be developed to be deployed at the company’s reception. This system should be capable of extracting certain details like Face photo, Name of the Person, Unique ID, etc. from the ID card scanned by a tablet to identify the visitor. 
Furthermore, this system is expected to be robust such that it can accept a variety of ID card images like Passport, National ID card, etc. and the data extraction is expected to be an on-demand process.

General approach to this problem:-

1) Capture the image using a tablet/smartphone and pass it as an input. If the image captured by the device is titled, use the DLIB toolkit for recognizing faces in the image and for adjusting the alignment of the image using the inclination angles calculated for the eyes of the human face in the ID card.

2) Since the background of the image isn’t of much importance for the task at hand and it may intrude in the process of textual extraction using OCR, it will be removed.

3) The task of removing the background is performed by first converting the image to grayscale and performing gamma correction on the image. Gamma correction helps in accurately displaying the image.

4) Next, we will find the contours within the image and further create a crop mask. The crop mask highlights the ID card as white and rest of the background as black, this will allow us to extract the ID card from the image and subtract the background and output the resultant image.

5) Now that we have an image containing only the ID card, we will use Easy OCR, a package created by Jaided AI. Easy OCR is one of the simplest tools available in the market for optical character recognition. Since Easy OCR also supports 58 languages, extracting text from ID cards of different countries will not be a problem. Easy OCR will recognize and extract text from the image using reader.readtext() and output it to a list.

6) Using Spacy, we will extract name of the person in the ID card from the generated list and extraction of the Unique ID will be done by filtering the list for ID number.

7) This extracted information will then be stored into a CSV file for future reference/Usage.


References:
1. Rotating the image and aligning the ID card: https://programmersought.com/article/91024394860/
2. Cropping the ID card background: https://medium.com/just-ai/tesseract-ocr-on-identity-documents-f3abae0ab1fc
3. Cropping and displaying the face photo: https://stackoverflow.com/questions/53926110/extract-face-rectangle-from-id-card/53926836
4. Easy OCR for text extraction from the image: https://github.com/bhattbhavesh91/easyocr-demo
5. Spacy: https://spacy.io/api/doc
6. DLIB Toolkit: http://dlib.net/
7. Haar Cascade Face Detector: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
