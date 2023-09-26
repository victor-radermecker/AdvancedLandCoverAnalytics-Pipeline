########################################## Import packages ##########################################
import sys
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import datetime
import tensorflow as tf
# building new cudnn lib from: https://macjim.medium.com/loading-alternative-cudnn-library-versions-in-tensorflow-90c7472e361a
# tf.load_library('/home/ubuntu/anaconda3/envs/tf/lib/python3.9/site-packages/nvidia/cudnn/lib')
from tensorflow import keras
from tensorflow.keras import layers

# Set preferences
tf.random.set_seed(42)
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

# Training XGBoost
# import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append('../src/')
sys.path.append('../Scripts/')

from  SequenceDataLoader import SequenceDataLoader
from get_data_loader import get_data_loader

########################################## Datasets for Generating Dataloader ##########################################

def get_historical_data(path):
    TRAIN_PATH = os.path.join(path,'urbanization_train.csv') #'../../Datasets/urbanization_train.csv'
    VAL_PATH = os.path.join(path,'urbanization_valid.csv') #'../../Datasets/urbanization_valid.csv'
    TEST_PATH = os.path.join(path,'urbanization_test.csv')  #'../../Datasets/urbanization_test.csv'

    # train data, Texas region
    train_data = pd.read_csv(TRAIN_PATH) # change
    # # validation data, Georgia region
    val_data = pd.read_csv(VAL_PATH) # change
    # test data, Ohio region
    test_data = pd.read_csv(TEST_PATH) # change

    # Train
    mean_urbanization_by_tile = train_data.groupby('tile_id')['urbanization'].mean()
    train_data_f = train_data[train_data['tile_id'].map(mean_urbanization_by_tile) > 0]

    # Validation
    mean_urbanization_by_tile = val_data.groupby('tile_id')['urbanization'].mean()
    val_data_f = val_data[val_data['tile_id'].map(mean_urbanization_by_tile) > 0]

    # Test
    mean_urbanization_by_tile = test_data.groupby('tile_id')['urbanization'].mean()
    test_data_f = test_data[test_data['tile_id'].map(mean_urbanization_by_tile) > 0]
    return train_data_f, val_data_f, test_data_f


########################################## Generate Dataloaders ###############################################################
def generate_dataloaders(train_data_f, val_data_f, test_data_f, IMG_DIR, IMG_SIZE, BATCH_SIZE, N_CHANNELS, LABELS, YEAR):
    # Image Directories 
    IMG_DIR_TRAIN = os.path.join(IMG_DIR, 'Train') #'../../Images/Train/'
    IMG_DIR_VALID = os.path.join(IMG_DIR, 'Valid') #'../../Images/Valid/'
    IMG_DIR_TEST =  os.path.join(IMG_DIR, 'Test') #'../../Images/Test/'

    train_data_loader = None

    # data_wide = val_data.pivot_table(index='tile_id', columns='year', values='urbanization', aggfunc='first')
    # data_wide = data_wide.merge(right=val_data[['tile_id', 'Lat', 'Lon']], on='tile_id')
    # data_wide.drop_duplicates(inplace = True, ignore_index = True)
    # TAB_DATA = data_wide.drop(columns = ["tile_id", 2022]) #.to_dict(orient='index')

    val_data_loader = get_data_loader(val_data_f, IMG_DIR_VALID, IMG_SIZE, BATCH_SIZE, LABELS, N_CHANNELS, YEAR) #, TAB_DATA = TAB_DATA)

    X, y= val_data_loader.__getitem__(10)

    # data_wide = test_data.pivot_table(index='tile_id', columns='year', values='urbanization', aggfunc='first')
    # data_wide = data_wide.merge(right=test_data[['tile_id', 'Lat', 'Lon']], on='tile_id')
    # data_wide.drop_duplicates(inplace = True, ignore_index = True)
    # tab_data = data_wide.drop(columns = ["tile_id", 2022]) #.to_dict(orient='index')

    test_data_loader = get_data_loader(test_data_f, IMG_DIR_TEST, IMG_SIZE, BATCH_SIZE, LABELS, N_CHANNELS, YEAR) #, TAB_DATA = tab_data)
    return train_data_loader, val_data_loader, test_data_loader



###################################### CONV-LSTM Model ########################################################

def create_model(inp1_shape):
    # Construct the input layer with 4 time frames in input
    inp1 = layers.Input(shape=(4, *inp1_shape))
    # inp2 =  layers.Input(shape=inp2_shape)

    # ConvLSTM layers
    x1 = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
        recurrent_activation="relu", # hard_sigmoid
    )(inp1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ConvLSTM2D(
      filters=8,
      kernel_size=(5, 5),
      padding="same",
      return_sequences=True,
      activation="relu",
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x1)
    x1_a = layers.GlobalAveragePooling3D()(x1)  # Add Global Average Pooling to collapse spatial dimensions
    x1_b = layers.GlobalMaxPooling3D()(x1)
    x = layers.concatenate([x1_a, x1_b])

    # Feedforward Layers
    # x2 = layers.Dense(8, activation='linear')(inp2)
    # x2 = layers.Dense(16, activation='relu')(x2)
    # x2 = layers.Dense(8, activation='relu')(x2)

    # concatenate before passing to next layer
    #x = layers.concatenate([x1, x2])

    X = layers.Dense(8, activation='relu')(x)
    # Add a Dense layer with a single unit and sigmoid activation
    x = layers.Dense(1, activation="sigmoid")(x)

    # ADAPT THE FOLLOWING TO MY CODE
    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp1, x)
    loss = tf.keras.losses.MeanSquaredError()  #tf.keras.losses.MeanAbsoluteError() # tf.keras.losses.MeanSquaredError() #keras.losses.binary_crossentropy
    optim = keras.optimizers.Adam()#learning_rate=0.1)
    model.compile(optimizer='adam', loss = loss, metrics = ['mae'])#weighted_mse_with_class)

    return model


######################################## Run Model ##################################################
def train(X, train_data_loader, val_data_loader):
    # Define input shapes
    inp1_shape = X.shape[2:]
    # inp2_shape = X[0].shape[1:]

    # Train the model from scratch
    model = create_model(inp1_shape) #inp2_shape

    # TENSORBOARD CALLBACKS
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # CALLBACKS
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  # Save the whole model, not just weights
        monitor='val_mean_squared_error',  # Use mean squared error for validation monitoring
        mode='min',  # Consider lower metric values as better
        save_best_only=True  # Save only the best model based on validation MSE
    )
    class ClearSessionCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):

            tf.keras.backend.clear_session()
    clear_session_callback = ClearSessionCallback()

    # PARAMS
    epochs = 20
    model.fit(
        train_data_loader,
        epochs=epochs,
        validation_data = val_data_loader,
        callbacks=[tensorboard_callback, reduce_lr, clear_session_callback],
    )
    return model

#################################### Saving Model ##########################################

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # params
    historical_data_path = '../../Datasets'
    IMG_DIR = '../../Images'
    IMG_SIZE = (40, 44)
    BATCH_SIZE = 64
    N_CHANNELS = 1
    LABELS = [2016, 2017, 2018, 2019]
    YEAR = 2020

    # train models for 2 year and 3 year prediction
    for lag in range(1, 3):
        year = YEAR+lag
        train_data_f, val_data_f, test_data_f = get_historical_data(historical_data_path)
        # train dataloader is None for now - training on validation and test set
        print('Generating Data for target year {}'.format(year))
        train_data_loader, val_data_loader, test_data_loader = generate_dataloaders(train_data_f, val_data_f, test_data_f, IMG_DIR, IMG_SIZE, BATCH_SIZE, N_CHANNELS, LABELS, year)
        X, y = test_data_loader[0]
        # training 
        print('Training...')
        model = train(X, val_data_loader, test_data_loader)
        print('Saving model...')
        model.save('../Models/conv-lstm-DW-20-{}-{}'.format(year, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
