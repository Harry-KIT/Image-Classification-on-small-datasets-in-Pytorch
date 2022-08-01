import random
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import time
import copy
from config import *
from data_loader import Dataloader
import cnn_models
import utils
import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# seed = random.randint(1, 10000)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if USE_GPU:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#         cudnn.benchmark = True
#         torch.cuda.manual_seed_all(seed)

def train_model(data_loader, model, criterion, optimizer, lr_scheduler, num_epochs=25):
    
    since_time = time.time()

    # list for tracking 
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_data in data_loader.load_data(data_set=phase):
                inputs, labels = batch_data
                if USE_GPU:
                    inputs, labels = inputs.to(device), labels.to(device)
                else:
                    inputs, labels = inputs, labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                # with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, predict = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # collect data info
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predict == labels.data)

            epoch_loss = running_loss / data_loader.data_sizes[phase]
            epoch_acc = running_corrects / data_loader.data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and train_acc_history[-1] > 0.9 and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            if phase == "val":
                # lr_scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history


def train():
    data_loader = Dataloader(data_dir='dataset', image_size=IMAGE_SIZE, batch_size=30)

    '''---Choose model---'''

    # model, params_to_update = cnn_models.initialize_model(model_name='vgg', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='densenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='resnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='squeezenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='alexnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='resnext', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='googlenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='mnasnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='regnet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='shufflenet', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='efficientnet', num_classes=4, feature_extract=True)
    model, params_to_update = cnn_models.initialize_model(model_name='ViT', num_classes=4, feature_extract=True)
    # model, params_to_update = cnn_models.initialize_model(model_name='Swin', num_classes=4, feature_extract=True)


    
    
    # print(model)
    model.to(device)

    # if USE_GPU:
    #     model = torch.nn.DataParallel(model).cuda()
    #     # or
    #     model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9) # use params_to_update instead of model.parameters()
    try:
        model, val_acc_history, val_loss_history, train_acc_history, train_loss_history = train_model(data_loader, 
                                            model, criterion, optimizer_ft, utils.exp_lr_scheduler, num_epochs=25)

        model.eval()
        label, label_predicted = utils.get_test_results(model, data_loader.load_data(data_set="val"))
        class_names = data_loader.data_classes

        metrics.plot_confusion_matrix(class_names, label.tolist(), label_predicted.tolist())
        metrics.get_classification_report(label, label_predicted, class_names)
        metrics.plot_loss(train_loss_history, val_loss_history)
        metrics.plot_accuracy(train_acc_history, val_acc_history)
        utils.save_torch_model(model, MODEL_SAVE_FILE)
        # torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print('manually interrupt, try saving model for now...')
        utils.save_torch_model(model, MODEL_SAVE_FILE)
        print('model saved.')


def main():
    train()

if __name__ == '__main__':
    main()
