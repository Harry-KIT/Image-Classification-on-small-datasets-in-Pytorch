import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image


def plot_loss(train_loss, val_loss):
    # Get num of epochs
    epochs = np.arange(0, len(train_loss))
    
    # Plot loss history
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, train_loss, "r--")
    plt.plot(epochs, val_loss, "b-")
    plt.legend(["Training Loss", "Val Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('results/loss.png')

def plot_accuracy(train_accuracy, val_accuracy):
    train_accuracy_list = [tensor.item() for tensor in train_accuracy]
    val_accuracy_list = [tensor.item() for tensor in val_accuracy]
    # Get num of epochs
    epochs = np.arange(0, len(train_accuracy_list))
    
    # Plot loss history
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, train_accuracy_list, "r--")
    plt.plot(epochs, val_accuracy_list, "b-")
    plt.legend(["Training Accuracy", "Val Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig('results/acc.png')

def plot_confusion_matrix(classes, test_labels, predictions):

    confusion = confusion_matrix(test_labels, predictions)
    cm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    axis_labels = classes
    
    ax = sns.heatmap(cm, xticklabels=axis_labels, yticklabels=axis_labels,
                        cmap='viridis', annot=True, fmt='.2f', square=True)
    
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig('results/confusion_matrix.png')

def get_classification_report(label, label_predicted, class_names):
    report = classification_report(label.tolist(), label_predicted.tolist(), target_names=class_names)
    report_path = "results/report.txt"
    text_file = open(report_path, "w")
    text_file.write(report)
    text_file.close()



def return_cam(feature_conv, weight_softmax, class_idx, model_input_size):
    # generate the class activation maps upsample to 256x256
    size_upsample = model_input_size
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        resized_cam_img = cv2.resize(cam_img, size_upsample)
        output_cam.append(resized_cam_img)
    return output_cam

def calculate_cam(model, model_input_size, features_blobs, img_read_path, classes):
    # get the softmax weight
    params = list(model.cpu().parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    
    # preprocess image
    preprocess = transforms.Compose([
                transforms.Resize(model_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_read_path)
    img_tensor = preprocess(img)
    img_variable = img_tensor.unsqueeze(0)
    
    # get model prediction
    logit = model(img_variable)
    #print(logit)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    # output the prediction
    #end = min(NUM_CLASSES, 5)
    #for i in range(0, end):
    #    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    
    # generate class activation mapping for the top1 prediction
    CAMs = return_cam(features_blobs[0], weight_softmax, [idx[0]], model_input_size)
    
    # render the CAM and output
    #print("Output CAM.jpg for the Top-1 Prediction: %s" % classes[idx[0]])
    img = cv2.resize(cv2.imread(img_read_path), model_input_size)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET) #cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite("CAM.jpg", result)
    del features_blobs[:]
    return Image.open("CAM.jpg"), classes[idx[0]]


def plot_images_cam(paths, model, input_size, features_blobs, class_names, columns=4, rows=5):
    fig = plt.figure(figsize=(15, 15))
    end = min(columns*rows+1, len(paths)+1)
    for i in range(1, end):
        img, pred = calculate_cam(model, input_size, features_blobs, paths[i-1], class_names)
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text("Predicted: {}".format(pred))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.show()

def create_path_list(base_path):
    path_list = []
    files = os.listdir(base_path)
    for _, file in enumerate(files):
        path_list.append(os.path.join(base_path, file))
    return path_list