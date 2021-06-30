import torch
import torch.nn as nn
import torchvision.models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

MODEL_LIST = ["resnet18"]

class CNN(nn.Module):
    """
    Class which implements backbone cnn model

    Arguments:
    - model_name (str): cnn model architecture (currently only resnet18 is available)
    - num_class (int): It should be None, when cnn is used as the backbone network for either LRCN or ODCNN
    - dropout (float): a probability for the last dropout layer
    - pretrained (bool): It specifies whether to use the pretrained model or nor
    """
    def __init__(self, model_name, num_class=None, dropout = 0.0, pretrained = True):
        super(CNN, self).__init__()

        self.model_name = model_name

        if model_name == "resnet18":
            input_size = (3, 224, 224)
            ft_extractor = torchvision.models.resnet18(pretrained=pretrained) #spatial feature extraction layers of CNN
            ft_size = ft_extractor.fc.in_features
            ft_extractor.fc = nn.Identity() #Nullify the last linear layer
        else:
            raise Exception(("The model '%s' is not supported. Choose the model in" % model_name) + ','.join( model for model in MODEL_LIST )) 

        self.input_size = input_size
        self.ft_extractor = ft_extractor
        self.ft_size = ft_size
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout) # The last dropout layer
        else:
            self.dropout = nn.Identity()
        self.output_size =  num_class if num_class is not None else ft_size
        
        if num_class is not None:
            # When num_class is not None, this cnn model is used as a stand-alone model to classify each frame separately
            self.classifier = nn.Linear(ft_size, num_class) # Contstruct the last linear layer
            self.ex_ft_mode = False # It is not just the feature extraction model
        else:
            # This cnn model is used as a backbone network for LRCN or ODCNN
            self.classifier = None
            self.ex_ft_mode = True # It is just the feature extraction model

        print(self.model_name, "used")


    def forward(self, inputs): #inputs: T(B, input_size) --> outputs: T(B, output_size)
        #Return the final output of the model, if ex_ft_model is True, then this method is equivalent to the method extract_feature()
        outputs = self.ft_extractor(inputs)
        outputs = self.dropout(outputs)
        if not self.ex_ft_mode:
            outputs = self.classifier( outputs ) 
        return outputs


    def extract_feature(self, inputs): #inputs: T(B, input_size) --> outputs: T(B, ft_size)
        #Return only spatial features
        return self.dropout( self.ft_extractor(inputs) )


    def extract_feature_mode(self, bool_):
        # Chagne the feature extraction model flag to bool_ and return the flag that the model previously had
        org_mode = self.ex_ft_mode
        if self.classifier is not None:
            self.ex_ft_mode = bool_
        return org_mode

class TempCNN(nn.Module):
    """
    Temporal 1-dimensional CNN kernel for the model, ODCNN
    Arguments:
    - in_feature_size (int)
    - out_feature_size (int)
    - kernel_size (int): 1D kernel size
    """
    def __init__(self, in_feature_size, out_feature_size, kernel_size):
        super(TempCNN, self).__init__()
        self.in_ftrs = in_feature_size
        self.out_ftrs = out_feature_size
        self.ks = kernel_size
        self.tconv = nn.Conv1d(in_channels = in_feature_size, out_channels = out_feature_size, kernel_size = kernel_size) #1D convolutional layer
        self.relu = nn.ReLU(True)

    def forward(self, inputs): #inputs T(B, in_feature_size, n_frame)
        outputs = self.relu( self.tconv(inputs) ) #T(B, out_channels, n_frame - kernel_size + 1)
        outputs, _ = torch.max( outputs, dim = -1 ) #T(B, out_channels) , max pooling over time
        return outputs

class LRCN(nn.Module):
    """
    LRCN model

    Arguments:
    - hidden_size (int): the hidden dimension for LSTM
    - action_size (int): the number of action categories
    - seq_len (int): unit video clip length
    - cnn_model (str): model architecture for a backbone cnn model
    - ft_size (int): the dimension of spatial features generated from the backbone network
    - lstm_dropout (float): a probability for the dropout layer which is applied to the output of LSTM 
    - cnn_dropout (float): a probability for the last dropout layer of the backbone cnn model
    """
    def __init__(self, hidden_size, action_size, seq_len = 16, cnn_model=None, ft_size = None, lstm_dropout=0.0, cnn_dropout=0.0):
        super(LRCN, self).__init__()

        self.action_size = action_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        cnn = None
        input_size = -1

        if cnn_model is not None:
            cnn = CNN(cnn_model, dropout = cnn_dropout) # Construct the backbone CNN model
        else:
            raise Exception("CNN model should be specified")

        ft_size = cnn.ft_size
        input_size = cnn.input_size

        self.cnn = cnn
        self.lstm = nn.LSTM( input_size=ft_size, hidden_size=hidden_size, batch_first=True )
        if lstm_dropout > 0.0:
            self.dropout = nn.Dropout( p = lstm_dropout )
        else:
            self.dropout = nn.Identity()
        self.linear_out = nn.Linear( hidden_size, action_size, bias=True )
        self.softmax = nn.Softmax( dim=2 )

        self.input_size = input_size
        self.ft_size = ft_size


    def forward(self, inputs):# inputs: T(B, seq_len, input_size) --> outputs: T(B, seq_len, Action)
        '''
        Take video clips and return logits for each frame in video clips
        '''
        inputs = inputs.view(-1, *self.cnn.input_size)
        inputs = self.cnn.extract_feature(inputs) # Extract spatial feature
        inputs = inputs.view(-1, self.seq_len, self.ft_size)

        return self.forward_lstm( inputs ) # Process through LSTM


    def forward_lstm(self, inputs):# inputs: T(B, seq_len, cnn_output_size) --> outputs: T(B, seq_len, Action)
        '''
        Take spatial features and return logits for each frame in video clips
        '''
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm( inputs )
        lstm_outputs = self.dropout( lstm_outputs )
        outputs = self.linear_out( lstm_outputs )

        return outputs

class ODCNN(nn.Module):
    """
    1dcnn model

    Arguments:
    - hidden_size (int): the hidden dimension for temporal features
    - action_size (int): the number of action categories
    - seq_len (int): unit video clip length
    - cnn_model (str): model architecture for a backbone cnn model
    - ft_size (int): the dimension of spatial features generated from the backbone network
    - dropout2 (float): a probability for the dropout layer which is applied to the output of a 1d cnn
    - dropout1 (float): a probability for the last dropout layer of the backbone cnn model
    - t_kernel_size (int): a temporal kernel size
    """    
    def __init__(self, hidden_size, action_size, seq_len = 16, cnn_model=None, dropout2=0.0, dropout1=0.0, t_kernel_size = 4):
        super(ODCNN, self).__init__()

        self.action_size = action_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.temp_kernel = t_kernel_size

        cnn = None
        input_size = -1

        if cnn_model is not None:
            cnn = CNN(cnn_model, dropout = dropout1) # Construct the backbone CNN model
        else:
            raise Exception("CNN model should be specified")

        ft_size = cnn.ft_size
        input_size = cnn.input_size

        self.cnn = cnn
        self.tcnn = TempCNN(ft_size, hidden_size, t_kernel_size) # Construct the temporal convolutional layer

        self.dropout = nn.Dropout( p = dropout2 ) if dropout2 > 0.0 else nn.Identity()

        self.linear_out = nn.Linear( hidden_size, action_size, bias=True )

        self.input_size = input_size
        self.ft_size = ft_size

    def forward(self, inputs):# inputs: T(B, seq_len, input_size) --> outputs: T(B, seq_len, Action)
        '''
        Take video clips and return logits for each frame in video clips
        '''
        inputs = inputs.view(-1, *self.cnn.input_size) #T(B * seq_len, *IMG_SIZE)
        inputs = self.cnn.extract_feature(inputs) #T(B * seq_len, ft_size), Exctract spatial features
        return self.process_feature(inputs) # Process through the temporal kernel


    def process_feature(self, inputs):# inputs: T(B, seq_len, cnn_output_size) --> outputs: T(B, Action)
        '''
        Take spatial features and return logits for each frame in video clips
        '''
        inputs = torch.transpose( inputs.view(-1, self.seq_len, self.ft_size), 1, 2 ) #T(B, cnn_output_size, seq_len)
        inputs = self.tcnn(inputs) #T(B, hidden_size)

        return self.linear_out( self.dropout( inputs ) ) #T(B, action_size)
