# Cancer-Detection-and-Segmentation
The purpose of our project is to have an accurate detection on cancer cells. The specific cancer cells we want to detect are that of in the parotid gland, the largest of the 3 salivary glands. Although parotid gland tumors are very rare, there are not many readily available projects and information on the tumor. As such, we want to use this opportunity to practice designing a system that can effectively identify parotid tumor types by segmentation on head and neck CT scans.

Data: The dataset comprises 300 CT images annotated by experts with bounding boxes and segmentation masks.
Evaluation: Performance evaluation was conducted using Intersection over Union (IoU), accuracy, precision, recall, and Dice coefficient metrics.
Model: YOLOv8 along with PyQt's GUI
