# GENERAL LIBS
# Data Analysis Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# from scipy.io import loadmat
from os import path

# Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

# Saving Trained Models
import pickle

# %matplotlib inline

# CNN keras libs
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D, AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

# Visualizing Model Architecture
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import pydot

# loading saved cnn model
from keras.models import load_model
from keras.utils import load_img, img_to_array


class FruitClassifierModel:

    __INPUT_SIZE = (256, 256)
    __BATCH_SIZE = 16
    # __N_EPOCHS = 40
    __METRICS = ["Accuracy", "Precision", "Recall", "ROC-AUC", "F-Measure"]

    # model names
    __FRUIT_CLASSIFIER_MODEL = 'fruits_classifier_model.h5'
    __APPLE_CLASSIFIER_MODEL = 'apple_days_classifier_model.h5'
    __BANANA_CLASSIFIER_MODEL = 'banana_days_classifier_model.h5'
    __CUCUMBER_CLASSIFIER_MODEL = 'cucumber_days_classifier_model.h5'
    __GUAVA_CLASSIFIER_MODEL = 'guava_days_classifier_model.h5'
    __LEMON_CLASSIFIER_MODEL = 'lemon_days_classifier_model.h5'
    __ORANGE_CLASSIFIER_MODEL = 'orange_days_classifier_model.h5'

    # banana_model1 = load_model(f'{MODEL_PATH}/banana_days_classifier_model.h5')
    # cucumber_model1 = load_model(f'{MODEL_PATH}/cucumber_days_classifier_model.h5')
    # guava_model1 = load_model(f'{MODEL_PATH}/guava_days_classifier_model.h5')
    # lemon_model1 = load_model(f'{MODEL_PATH}/lemon_days_classifier_model.h5')
    # orange_model1 = load_model(f'{MODEL_PATH}/orange_days_classifier_model.h5')

    def __init__(self, model_dir_or_url=None):
        """
          model_dir_or_url: the folder containing the trained models or url pointing to the location or endpoint of the model
        """
        self.default_dir = "models"
        self.model_dir = model_dir_or_url or f'{self.default_dir}'

        # Loading models
        self.trained_fruits_classifier = self.get_model(model_code_name='main')
        self.trained_fruit_days_classifier = None

    def set_model_directory(self, model_uri):
        self.model_dir = model_uri

    # def load_model(self, image_model_name):
    #     """
    #       Returns a tupple containing trained models: (textmodel, imagemodel)
    #     """
    #     try:
    #         image_model_path = path.join(self.model_dir, image_model_name)
    #         # from keras.models.load_model
    #         image_model = load_model(image_model_path)

    #         return image_model

    #     except Exception as err:
    #         print(
    #             f'Sorry, an error occured. Could not load model "{image_model_name}"...\n{err}')
    #         return None, None

    # UTILITY FUNCTIONS HERE (Private)
    def get_model(self, model_code_name='main'):

        if model_code_name.lower() == 'main':
            return load_model(path.join(self.model_dir, self.__FRUIT_CLASSIFIER_MODEL))
        elif model_code_name.lower() == 'apple':
            return load_model(path.join(self.model_dir, self.__APPLE_CLASSIFIER_MODEL))
        elif model_code_name.lower() == 'banana':
            return load_model(path.join(self.model_dir, self.__BANANA_CLASSIFIER_MODEL))
        elif model_code_name.lower() == 'cucumber':
            return load_model(path.join(self.model_dir, self.__CUCUMBER_CLASSIFIER_MODEL))
        elif model_code_name.lower() == 'guava':
            return load_model(path.join(self.model_dir, self.__GUAVA_CLASSIFIER_MODEL))
        elif model_code_name.lower() == 'lemon':
            return load_model(path.join(self.model_dir, self.__LEMON_CLASSIFIER_MODEL))
        elif model_code_name.lower() == 'orange':
            return load_model(path.join(self.model_dir, self.__ORANGE_CLASSIFIER_MODEL))
        else:
            raise Exception("Invalid model code name supplied")

    def __load_image(self, img_path, show=False):
        img = load_img(img_path, target_size=self.__INPUT_SIZE,
                       color_mode='rgb')

        # (height,width,channels)
        img_tensor = img_to_array(img)

        # img_tensor = np.vstack([img_tensor])

        # (1,height,width,channels), adds a dim coz model expects shape: (batch_size,height,width,channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # imshow expects values in range[0,1]
        img_tensor = img_tensor / 255.

        if show:
            plt.imshow(img_tensor[0])
            plt.axis('off')
            plt.show()

        return img_tensor

    def get_fruit_encodings(self, fruit=None):
        if fruit is None:
            return {
                'Apple': 0, 'Banana': 1, 'Cucumber': 2,
                'Guava': 3, 'Lemon': 4, 'Orange': 5
            }

        elif fruit.lower().strip() in {'apple', 'banana', 'cucumber', 'guava', 'lemon', 'orange'}:
            if fruit.lower().strip() == 'apple':
                return {
                    'Day0': 0,
                    'Day11': 1,
                    'Day13': 2,
                    'Day17': 3,
                    'Day21': 4,
                    'Day4': 5,
                    'Day5': 6,
                    'Day7': 7
                }

            elif fruit.lower().strip() == 'banana':
                return {
                    'Day0': 0,
                    'Day11': 1,
                    'Day13': 2,
                    'Day15': 3,
                    'Day2': 4,
                    'Day21': 5,
                    'Day3': 6,
                    'Day5': 7,
                    'Day7': 8
                }

            elif fruit.lower().strip() == 'cucumber':
                return {
                    'Day0': 0,
                    'Day11': 1,
                    'Day17': 2,
                    'Day19': 3,
                    'Day19b': 4,
                    'Day2': 5,
                    'Day21': 6,
                    'Day23': 7,
                    'Day25': 8,
                    'Day27': 9,
                    'Day3': 10,
                    'Day6': 11,
                    'Day9': 12
                }

            elif fruit.lower().strip() == 'guava':
                return {'Day0': 0,
                        'Day13': 1,
                        'Day15': 2,
                        'Day23': 3,
                        'Day4': 4,
                        'Day5': 5,
                        'Day7': 6
                        }

            elif fruit.lower().strip() == 'lemon':
                return {'Day0': 0,
                        'Day11': 1,
                        'Day15': 2,
                        'Day17': 3,
                        'Day19': 4,
                        'Day2': 5,
                        'Day21': 6,
                        'Day3': 7,
                        'Day5': 8,
                        'Day9': 9
                        }

            elif fruit.lower().strip() == 'orange':
                return {'Day0': 0,
                        'Day11': 1,
                        'Day14': 2,
                        'Day2': 3,
                        'Day20': 4,
                        'Day3': 5,
                        'Day30': 6,
                        'Day5': 7,
                        'Day7': 8
                        }

        else:
            raise Exception(
                f"Invalid value set for fruit '{fruit}'. Allowed value is any of {None, 'apple', 'banana', 'cucumber', 'guava','lemon','orange'}.")

    def get_class_name(self, x, fruit=None):
        """
        Returns the 
            fruit: 
                - For fruit days prediction, set fruit to any of 
                    {'apple', 'banana', 'cucumber', 'guava','lemon','orange'}
                - For fruit prediction, leave fruit as, None
        """
        class_coding = self.get_fruit_encodings(fruit=fruit)

        if type(x) == list:  # if list
            classes = []
            for label in x:
                if label not in class_coding.values():
                    raise Exception('Invalid label supplied')
                    return None
                else:
                    loc = list(class_coding.values()).index(label)
                    classes.append(list(class_coding.keys())[loc])
                    # for k,v in class_coding.items():
                    #     if v == label:
                    #         classes.append(k)
                    # break
            return classes

        else:  # if string
            if x not in class_coding.values():
                raise Exception('Invalid label supplied')
                return None
            else:
                loc = list(class_coding.values()).index(x)
                return list(class_coding.keys())[loc]

    def get_labels_from_path(self, paths, fruit=None):
        class_coding = self.get_fruit_encodings(fruit=fruit)

        if type(paths) == list:
            labels = []
            for path in paths:
                classes = path.split("/")[-2]
                labels.append(class_coding[classes])
            return labels

        else:
            classes = paths.split("/")[-2]
            return class_coding[classes]

    def get_class_code(self, label, fruit=None):
        class_coding = self.get_fruit_encodings(fruit=fruit)

        # class_coding = get_encoded_labels_coding(fruit=fruit)e3  zzz
        if label.capitalize() in class_coding.keys():
            return class_coding[label]
        else:
            raise Exception(
                f"Invalid label supplied. Valid labels are: {list(class_coding.keys())}")
            return None

    def get_preprocessed_aug_images(self, image_directory_path):
        # Loading Testing Set data
        datagen = ImageDataGenerator(rescale=1./255)

        aug_images = datagen.flow_from_directory(image_directory_path,
                                                 target_size=self.__INPUT_SIZE,
                                                 batch_size=self.__BATCH_SIZE,
                                                 class_mode='categorical')
        return aug_images

    def predict_cnn_with_details(self, img_path, model, fruit_name=None):
        # Preprocess Image
        results = {}
        new_image = self.__load_image(img_path, show=True)

        # Making Prediction
        predicted = model.predict(new_image, batch_size=16)

        predicted = predicted.round(3)

        # Showing Prediction details
        encoder = self.get_fruit_encodings(fruit=fruit_name)
        classes = list(encoder.keys())

        predicted_value = classes[predicted[0].argmax()]
        confidence = round(predicted[0].max() * 100, 2)

        # fruit_day = img_path.split(
        #     "/")[-2] if fruit_name is not None else img_path.split("/")[-3]
        # if fruit_day in encoder.keys():
        #     actual_value = fruit_day.capitalize()
        # else:
        #     actual_value = 'NA'
        actual_value = 'NA'

        results['Classes'] = classes
        results['Probabilities'] = predicted
        results['Prediction'] = predicted_value
        results['Confidence'] = confidence
        results['Actual'] = actual_value

        # print()
        # print(f"Probabilities: \t{predicted}")
        # print(f"Classes: {classes}")
        # print(
        #     f"Predicted Value: {predicted_value} (Confidence: {confidence}%)")
        # print(f"Actual Value: {actual_value}")

        return results

    def predict_single_image_with_details(self, img_path, model, fruit_name=None):
        # if supported image format
        if img_path.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp']:
            # Preprocessing Image
            new_image = self.__load_image(img_path, show=True)

            # Making Prediction
            predicted = model.predict(
                new_image, batch_size=self.__BATCH_SIZE)

            # Showing Prediction details
            encoder = self.get_fruit_encodings(fruit=fruit_name)

            classes = list(encoder.keys())

            pred_lbl = predicted[0].argmax()
            # pred_val = self.__get_label_string_from_code(pred_lbl)
            pred_val = classes[pred_lbl]
            confidence = round(predicted[0].max() * 100, 2)
            predicted = predicted.round(3)

            # predicted = np.array([[predicted[0][1], predicted[0][0]]])
            hybrid_pred = {'probability': predicted, 'classes': classes,
                           'prediction': pred_val, 'confidence': confidence}

            # print()
            # print(f"Prediction: \t{predicted}")
            # print(f"Encoded Classes: {classes}")
            # print(f"Predicted Value: {pred_val} ")
            # print(f"Confidence:: {confidence}%")
            return hybrid_pred

        else:
            raise Exception(
                "Unsupported file or image format. Supported image formats are ('.jgp', '.png', 'jpeg', '.bmp')")
            return None

    def __get_fruit_properties(self, fruit, fruit_day):
        properties = {}
        base_dir = path.join(self.model_dir, 'fruit_details')
        if fruit.lower() in ['apple', 'banana', 'cucumber', 'guava', 'lemon', 'orange']:
            base_dir = path.join(base_dir, fruit.capitalize())
        physico_chem_path = path.join(
            base_dir, f'{fruit.upper()}_PHYSICOCHEMICAL.csv')
        proximate_anal_path = path.join(
            base_dir, f'{fruit.upper()}_PROXIMATE_ANALYSIS.csv')
        vitamins_path = path.join(base_dir, f'{fruit.upper()}_VITAMINS.csv')

        physico_chemical_ds = pd.read_csv(physico_chem_path, index_col=0)
        proximate_anal_ds = pd.read_csv(proximate_anal_path, index_col=0)
        vitamins_ds = pd.read_csv(vitamins_path, index_col=0)

        physico_chemical_props = physico_chemical_ds.loc[fruit_day.upper(
        )].to_dict()
        proximate_anal_props = proximate_anal_ds.loc[fruit_day.upper(
        )].to_dict()
        vitamins_props = vitamins_ds.loc[fruit_day.upper()].to_dict()

        properties['Physicochemical Properties'] = physico_chemical_props
        properties['Proximate Analysis'] = proximate_anal_props
        properties['Vitamins'] = vitamins_props

        return properties

    def predict_pipeline(self, img_path):
        results = {}
        try:
            # fruit_results = self.predict_cnn_with_details(
            #     img_path=img_path, model=self.trained_fruits_classifier)
            fruit_results = self.predict_single_image_with_details(
                img_path=img_path, model=self.trained_fruits_classifier)

            if 'prediction' in fruit_results.keys():
                fruit_name = fruit_results['prediction']
                days_model = self.get_model(model_code_name=fruit_name.lower())

                # days_results = self.predict_cnn_with_details(
                #     img_path=img_path, model=days_model, fruit_name=fruit_name)

                days_results = self.predict_single_image_with_details(
                    img_path=img_path, model=days_model, fruit_name=fruit_name)

                results['category'] = fruit_results

                if 'prediction' in days_results.keys():
                    fruit_day = days_results['prediction']

                    results["day"] = days_results
                    results['properties'] = self.__get_fruit_properties(
                        fruit=fruit_name, fruit_day=fruit_day)
                else:
                    results['day'] = 'NA'
            else:
                results['category'] = 'NA'

            # return results
        except Exception as err:
            results['error'] = True
            results['error_message'] = err
        finally:
            return results
