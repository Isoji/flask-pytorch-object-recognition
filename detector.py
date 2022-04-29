# Detector class file

# Imports
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
from custom_utils import loadImage

# Class for the object detection model instantiation and inferencing
class Detector:
    # Class names that the fine-tuned model is trained on
    MY_CATEGORY_NAMES = [
    '__background__', 'vehicle', 'person']

    # Loads pre-trained model and sets it to evaluation mode
    def __init__(self):
        # Sets the device to gpu if available, else it uses cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Loads the model
        self.model = torch.load("app/model/tuned_fasterrcnn.pt")
        self.model.to(self.device).eval()

    # Method that inferences the model on the image input and returns the predictions
    def get_prediction(self, imageFile, threshold):
        image = loadImage(imageFile)
        transform = T.Compose([T.ToTensor()])# Transforms the loaded image (with normalization) into a float tensor of shape (C x H x W)
        imgTensor = transform(image).to(self.device)

        # Perform the prediction on the transformed image (tensor)
        pred = self.model([imgTensor])
        # Adding each predicted class to the pred_class array
        pred_class = [self.MY_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        # Adding each predicted bounding box to the pred_boxes array
        pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        # Adding each prediction score to the pred_score array
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        
        # If there are no predictions, or there is no score that is above the threshold
        if(len(pred_score) == 0 or max(pred_score) < threshold):
            pred_box = []
            pred_class = [] 

        # Else filter through pred_scores and get the last index where the score > threshold
        # Assign the box and class arrays with index values until pred_t index
        else:
            pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
            pred_box = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]

        # Returns a tuple containing the final arrays storing the inference results    
        return (pred_box, pred_class)

    # Method that saves an image with drawn bounding boxes and labels on it
    def create_visualization(self, imageFile, fileName, boxes, classes, rect_thickness, text_thickness, text_size):
        image = loadImage(imageFile)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Assigning the boxes and classes from the inference
        boxes_ = boxes
        classes_ = classes

        # Each box means a seperate detection
        for i in range(len(boxes)):
            # Only display data if the class == 'person'
            if(classes[i] == 'person'):
                #r,g,b,a = (255, 56, 56, 1) # red
                r,g,b,a = (255, 105, 180, 1) # pink
                # Draw box with the coordinates
                cv2.rectangle(img, boxes_[i][0], boxes_[i][1], color=(r,g,b,a), thickness=rect_thickness)
                # Put the text that goes with the box
                cv2.putText(img, classes_[i], boxes_[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (r,g,b,a), thickness=text_thickness)
            elif(classes[i] == 'car' or classes[i] == 'vehicle'):
                #r,g,b,a = (56, 56, 255, 1) # blue
                r,g,b,a = (0, 255, 255, 1) # cyan
                # Draw box with the coordinates
                cv2.rectangle(img, boxes_[i][0], boxes_[i][1], color=(r,g,b,a), thickness=rect_thickness)
                # Put the text that goes with the box
                cv2.putText(img, classes_[i], boxes_[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (r,g,b,a), thickness=text_thickness)

        # Create output image
        dpi = 96
        plt.figure(frameon=False, figsize=(352/dpi, 288/dpi), dpi=dpi) # 352 x 288
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig("C:/Users/tremb/Dev/PyTorch/app/outputs/"+fileName, dpi=dpi)
        plt.savefig("C:/Users/tremb/Dev/PyTorch/app/static/outputs/"+fileName, dpi=dpi)