from tensorflow import keras
from keras.models import Model
from keras import layers
import tensorflow as tf
import pyttsx3

# import h5py

def get_cnn_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')

    new_input = image_model.input                    #write code here to get the input of the image_model
    hidden_layer = image_model.layers[-1].output     #write code here to get the output of the image_model

    image_features_extract_model = keras.Model(new_input, hidden_layer)   #build the final model using both input & output layer
    return image_features_extract_model

class Encoder(Model):
    def __init__(self,embedding_dim):
        super(Encoder, self).__init__()
        self.dense = layers.Dense(embedding_dim)       #build your Dense layer with relu activation
        
    def call(self, features, training=False):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = self.dense(features)
        features = tf.nn.relu(features)
        return features

class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = layers.Dense(units)       #build Dense layer
        self.W2 = layers.Dense(units)       #build Dense layer
        self.V = layers.Dense(1)            #build final Dense layer with unit 1
        self.units=units

    def call(self, features, hidden):
        #features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        
        # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis =  tf.expand_dims(hidden, 1)  
        
        # build the score funciton to shape: (batch_size, 8*8, units)
        score = keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) 
        
        # extract the attention weights with shape: (batch_size, 8*8, 1)
        attention_weights = keras.activations.softmax(self.V(score), axis=1)
        
        #shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = attention_weights * features 
        
        # reduce the shape to (batch_size, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units)                # iniitalise the Attention model with units
        self.embed = layers.Embedding(vocab_size, embedding_dim)    # build the Embedding layer     
        self.gru = layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = layers.Dense(self.units)              # build the Dense layer    
        self.d2 = layers.Dense(vocab_size)              # build the Dense layer
        
    def call(self,x,features, hidden):
        #create the context vector & attention weights from attention model
        context_vector, attention_weights = self.attention(features, hidden)
        # embed the input to shape: (batch_size, 1, embedding_dim)
        embed = self.embed(x) 
        # Concatenate the input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1) 
        # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output,state = self.gru(embed)
        output = self.d1(output)
        # shape : (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2])) 
        # shape : (batch_size * max_length, vocab_size)
        output = self.d2(output) 
        
        return output, state, attention_weights
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def text_to_speech(text):
    """
    Function to convert text to speech
    :param text: text
    :param gender: gender
    :return: None
    """

    engine = pyttsx3.init()

    # Setting up voice rate
    engine.setProperty('rate', 125)

    # Setting up volume level  between 0 and 1
    engine.setProperty('volume', 0.8)

    # Change voices: 0 for male and 1 for female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', 1)

    engine.say(text)
    engine.runAndWait()
