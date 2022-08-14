from models import Decoder, Encoder, get_cnn_model
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import Input, layers
import pickle


embedding_dim = 256 
units = 512
#question
vocab_size = 20001   #top 5,000 words +1
max_length_caption = 39

decoder = Decoder(embedding_dim, units, vocab_size)
decoder.load_weights('weights/decoder/dec')

encoder = Encoder(embedding_dim)
encoder.load_weights('weights/encoder/enc')

with open('weights/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def load_image(image_path):
    #write your pre-processing steps here
    preprocessed_img = tf.io.read_file(image_path)
    preprocessed_img = tf.image.decode_jpeg(preprocessed_img, channels=3)
    preprocessed_img = tf.image.resize(preprocessed_img, (299, 299))
    # tf.keras.applications.inception_v3.preprocess_input will normalize input to range of -1 to 1
    preprocessed_img = tf.keras.applications.inception_v3.preprocess_input(preprocessed_img)
    return preprocessed_img, image_path

    
def infer_caption(image):
    
    # decoder = keras.models.load_model('decoder')
    # Create a new model instance
    hidden = decoder.init_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)  # process the input image to desired format before extracting features
    image_features_extract_model = get_cnn_model()
    img_tensor_val = image_features_extract_model(temp_input)        # Extracting features using feature extraction model
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)       # extracting the features by passing the input to encoder

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length_caption):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)    # getting the output from decoder

        
        # extract the predicted id(embedded value) which carries the max value
        predicted_id = tf.argmax(predictions[0]).numpy()     
        # map the id to the word from tokenizer and append the value to the result list
        result.append(tokenizer.index_word[predicted_id])
        
        if tokenizer.index_word[predicted_id] == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)


    
    pred_caption=' '.join(result).rsplit(' ', 1)[0]
    
    # we make use of Google Text to Speech API (online), which will convert the caption to audio
    # speech = gTTS('Predicted Caption : ' + pred_caption, lang = 'en', slow = False)
    # speech.save('voice.mp3')
    # audio_file = 'voice.mp3'

    # display.display(display.Audio(audio_file, rate = None, autoplay = autoplay))
    
    return pred_caption
