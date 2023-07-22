import torch
import time
import torchvision
from torchvision.transforms import ToTensor
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import collections

# todo: write how to use notes

# todo: include example results, input data, and weights/layers/params files for each set of test classes (once I have a GPU that doesnt take a week ;D)

# todo: args parse from outside function call from user input
# todo: allow user input of new pre-trained model or fine-tuning of model (requires saving model with other method .save_pretrained(''))
# todo: allow input of other models to train function and add more model classes here, as well as optimizer and loss function
# todo: transform Chris numpy arrays directly to pixel values tensor and class tuples for input without ImageFolder images (in data prep file and then rem that line from here)
# todo: parallelize and use ensemble with different models
# todo: readme!
# if anyone has any questions about the code or (other) requests for ease of use
# please let me know - Robin (yancey5) :) 

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(str(os.getcwdb())[2:-1] + '/save_p/') #n_classes
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values = pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None


def train(ViT_results_directory, input_ims_path, epochs, n_workers = 8, valid_steps = 1000, batch_size = 10, learning_rate = 2e-5):

    isExist = os.path.exists(ViT_results_directory)
    if not isExist:
        os.mkdir(ViT_results_directory)

    version = str(time.ctime()).replace(' ', '-').replace(':', '_') + '_train'
    results_directory = ViT_results_directory + version + '/'

    os.mkdir(results_directory)

    model_save_path = results_directory + version + '_vit_weights.pt' # comment out last line of file to not use or updat


    train_ds = torchvision.datasets.ImageFolder(input_ims_path + 'train/', transform = ToTensor())
    print('TRAINING...')
    print("Number of train examples: ", len(train_ds))
    print('Train classes:', train_ds.classes)
    num_classes = len(train_ds.classes)
    print('Number of classes:', num_classes)
    train_loader = data.DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = n_workers)
    isExist = os.path.exists(input_ims_path + 'valid/')
    if isExist:
        valid_ds = torchvision.datasets.ImageFolder(input_ims_path + 'valid/', transform = ToTensor())
        print("# of validation examples: ", len(valid_ds))
        valid_loader = data.DataLoader(valid_ds, batch_size = batch_size, shuffle = True, num_workers = n_workers)
    else:
        validate = False

    # gets info from images that the network was pretrained on
    feature_extractor = ViTFeatureExtractor(resample = 2)

    tot_steps = (len(train_ds) / batch_size) * epochs
    print('Total steps:', tot_steps , 'over', epochs, 'epochs')

    model = ViTForImageClassification(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.cuda()
        print('Using GPU')
    else:
        print('Using CPU')
        n_workers = 1

    model.train()
    t1 = time.time()
    for epoch in range(epochs):
      for step, (x, y) in enumerate(train_loader):
        arr = np.squeeze(np.array(x))
        x = np.split(arr, arr.shape[0])

        for index, array in enumerate(x):
          x[index] = np.squeeze(array)

        x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis = 0))
        x, y = x.to(device), y.to(device)
        b_x = Variable(x)
        b_y = Variable(y)
        output, loss = model(b_x, None)

        if loss is None:
          loss = loss_func(output, b_y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if validate and (step % valid_steps == 0):
          print('Train Step ', step, 'of', tot_steps)
          val = next(iter(valid_loader))
          val_x = val[0] # first index is im

          # divide test batch up into list of arrays to iterate
          arr = np.squeeze(np.array(val_x))
          val_x = np.split(arr, arr.shape[0])

          # iterate through each (rm resulting/unneccesary 1-d in first dimension)
          for index, array in enumerate(val_x):
            val_x[index] = np.squeeze(array)

          # this is the format the network takes
          val_x = torch.tensor(np.stack(feature_extractor(val_x)['pixel_values'], axis=0))

          val_x = val_x.to(device)
          # the second item in each tensor from the iterator will always
          # holds the truth class
          val_y = val[1].to(device)

          val_output, loss = model(val_x, val_y)
          val_output = val_output.argmax(1) # pick class with highest accuracy

          acc = (val_output == val_y).sum().item() / len(val_output)
          print('Validation set results for epoch #:', epoch, '; batch loss: %.4f' % loss, '; batch accuracy: %.2f' % acc)


    torch.save(model, model_save_path)
    print('Model saved to:', model_save_path)

    train_time = round((time.time() - t1)/60, 2)
    print('Total training time', train_time, 'minutes')

    class_count = dict(collections.Counter(train_ds.targets))
    class_nums = dict(collections.Counter(train_ds.class_to_idx))
    d = {}
    for i in range(len(class_count)):
        d[train_ds.classes[i]] = class_count[i]

    f = open(results_directory + 'Train-details.txt', 'a')
    f.write('Training time '+str(train_time) +' minutes\n')
    f.write('Params: \nNumber of epochs: '+ str(epochs)+ ' Batch size: ' + str(batch_size)+ ' Learning rate: ' + str(learning_rate) + '\n')

    f.write("# of train examples: " + str(len(train_ds)) + '\n')
    f.write('Train classes:' + str(train_ds.classes) + '\n')
    f.write('Number of classes:' + str(num_classes) + '\n')

    f.write('Class counts used for training: ' + str(d) + '\n')
    f.close()


def test(ViT_results_directory, input_ims_path, model_load_path, UL_classes_to_predict = None, n_workers = 1):
    feature_extractor = ViTFeatureExtractor(resample = 2)
    version = str(time.ctime()).replace(' ', '-').replace(':', '_')
    results_directory = ViT_results_directory + version + '_test' + '/'

    isExist = os.path.exists(ViT_results_directory)
    if not isExist:
        os.mkdir(ViT_results_directory)
    os.mkdir(results_directory)

    test_ds = torchvision.datasets.ImageFolder(input_ims_path + 'test/', transform = ToTensor())
    test_loader = data.DataLoader(test_ds, batch_size = 1, shuffle = False, num_workers = n_workers)


    print('TESTING...')
    print("Number of test examples: ", len(test_ds))
    num_classes = len(test_ds.classes)
    print('Test classes:', test_ds.classes)
    print('Number of classes:', num_classes)
    model = torch.load(model_load_path)

    model.eval()
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.cuda()
        print('Using GPU')
    else:
        print('Using CPU')

    tot_c, all_conf_c = 0, 0
    all_conf = 0
    all_preds, all_targets = [], []

    isExist = os.path.exists(results_directory + 'Incorrect_Predictions')
    if not isExist and UL_classes_to_predict is not None:
        os.mkdir(results_directory + 'Incorrect_Predictions')

    if UL_classes_to_predict is not None:

        os.mkdir(results_directory + 'Low_Conf_Predictions/')
        os.mkdir(results_directory + 'High_Conf_Predictions/')

        for i in range(len(UL_classes_to_predict)):
            os.mkdir(results_directory + 'Low_Conf_Predictions/' + str(UL_classes_to_predict[i]).replace(' ', '_'))
            os.mkdir(results_directory + 'High_Conf_Predictions/' + str(UL_classes_to_predict[i]).replace(' ', '_'))


    t1 = time.time()

    with torch.no_grad():

        for i, arr in enumerate(test_loader):
            test_im, expected = arr
            # extract tensor value repping actual class
            target = expected[0].item()
            # put permutation of dims required for network
            test_im = test_im[0].permute(1, 2, 0)
            test_im = np.squeeze(test_im)
            # save  image to classify as numpy (after rem extra dim) for saving into appropriate output folder
            save = test_im.numpy()
            # and extract 'pixel values' feature required for input to network for testing
            test_im = torch.tensor(np.stack(feature_extractor(test_im)['pixel_values'], axis=0))
            # put in on gpu if possible
            test_im = test_im.to(device)
            expected = expected.to(device)
            # expected value won't be supplied (or will not be one of the classes input for unlabeled data testing)
            prediction, loss = model(test_im, expected)
            # take ind with highest output repping class pred
            predicted_class = np.argmax(prediction.cpu())
            # save list of preds and targets to create the confusion matrix
            all_preds.append(predicted_class.item())
            all_targets.append(target)
            # now reverse normalize to save with cv2
            save = (save * 255.0).astype("uint8")
            # calculate confidence (as % vs other classes) for info feedback to user
            #conf0 = max(sm(prediction.cpu()[0]))
            conf0 = max(F.softmax(prediction.cpu()[0], dim=0))
            all_conf += conf0.item()
            # this is the version for the image labels
            conf = '_(' + str(round(conf0.item() * 100)) + '%)'

            if UL_classes_to_predict is not None:
                if conf0 < 0.99:
                    # write low conf image to output directory
                    cv2.imwrite(results_directory + 'Low_Conf_Predictions/' + str(UL_classes_to_predict[predicted_class.item()]).replace(' ', '_')
                                + '/' + str(i) + '_predicted_' + str(UL_classes_to_predict[predicted_class.item()]).replace(' ', '_') + conf + '.jpg', save)
                else:
                    # write high conf image to output directory
                    cv2.imwrite(results_directory + 'High_Conf_Predictions/' + str(UL_classes_to_predict[predicted_class.item()]).replace(' ', '_')
                                + '/' + str(i) + '_predicted_' + str(UL_classes_to_predict[predicted_class.item()]).replace(' ', '_') + conf + '.jpg', save)
            else:
                if predicted_class.item() == target:
                    # tot_cal confidence vals for
                    tot_c += 1
                    all_conf_c += conf0
                else:
                    # write misclassified image to output directory
                    cv2.imwrite(results_directory + 'Incorrect_Predictions/' + str(i) + '_predicted_' + str(test_ds.classes[predicted_class.item()]).replace(' ','_') + conf + '_actual_' + str(test_ds.classes[target]).replace(' ','_') + '.jpg', save)
                    print('predicted_class', predicted_class.item(), 'actual', target)

    print('Test time:', round((time.time() - t1)/60, 2), 'minutes')
    print('Av confidence in ALL predictions: ', round(all_conf / len(test_ds) * 100), '%')

    if UL_classes_to_predict is None:
        print('Accuracy:', round((tot_c / len(test_ds) * 100), 2), '%')
        print('Average confidence in correct predictions', round(all_conf_c.item() / tot_c * 100), '%')

        y_true = pd.Series(all_targets, name = "Actual")
        y_pred = pd.Series(all_preds, name = "Predicted")
        df_confusion = pd.crosstab(y_true, y_pred)
        cfm = [ [ df_confusion[j][i] for i in range(len(df_confusion[j])) ] for j in range(len(df_confusion[0])) ]
        df_cfm = pd.DataFrame(cfm, index = test_ds.classes, columns = test_ds.classes)
        print('Confusion Matrix of Actaul Classes vs. Predictions:')
        print(df_cfm)


        f = open(results_directory + 'Evaluation-details.txt','a')
        f.write('Accuracy: ' + str(round((tot_c / len(test_ds) * 100), 2)) + '%\nAverage confidence in correct predictions: ' + str(round(all_conf_c.item() / tot_c * 100)) + '%\n')
        f.write('Av confidence in ALL predictions: ' + str(round(all_conf / len(test_ds) * 100)) + '%\n')

        f.write('\nConfusion Matrix of Actual Classes vs. Predictions: \n')
        f.write(str(df_cfm)+'\n')
        f.write('Test time: ' + str(round((time.time() - t1)/60, 2)) + ' minutes')
        f.close()

    else:
        print('Predicting UL data time:', round((time.time() - t1)/60, 2), 'minutes')



if __name__ == '__main__':

    # main inputs: 1; results directory (main for test and/or train), 2; path to data set with test and/or train sets, 3; epochs (for train) and model (for test)
    # datasets should be in pytorch file folder structure
    # this file should be in the same directory as save_p (pretrained ImageNet weights)
    train('/Users/yancey5/Desktop/AMH_EDA/train_test_results/', '/Users/yancey5/Desktop/AMH_EDA/AMH_55k-Binary-mini/', 1)
    ctp = ['Damage', 'Non-Damage']
    test('/Users/yancey5/Desktop/AMH_EDA/train_test_results/',  '/Users/yancey5/Desktop/AMH_EDA/AMH_55k-Binary-mini/', '/Users/yancey5/Desktop/AMH_EDA/ViT_model.pt')#, ctp)
