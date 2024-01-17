import speech_recognition as sr
import nltk
import string
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pygame

# Load the fine-tuned model and tokenizer
model_path = 'model.h5'
tokenizer_path = 'tokenizer.json'

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Download the Punkt tokenizer for Marathi
nltk.download('punkt')

# Initialize pygame for playing the beep sound
pygame.init()
beep_sound = pygame.mixer.Sound('censor_beep.mp3')

def tokenize_marathi_sentence(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)

    # Remove punctuation from the words
    words = [word for word in words if word not in string.punctuation]

    return words

def play_beep():
    pygame.mixer.Channel(0).play(beep_sound)

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source,phrase_time_limit=5)

    try:
        text = recognizer.recognize_google(audio, language="mr-IN")  # Change language to Marathi (mr-IN)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Google Speech Recognition request failed: {e}")
        return None

if __name__ == "__main__":
    while True:
        spoken_text = recognize_speech()

        if spoken_text:
            marathi_words = tokenize_marathi_sentence(spoken_text)
            for word in marathi_words:
                # Tokenize and process the input text
                inputs = tokenizer.encode_plus(word, return_tensors='pt', padding=True, truncation=True)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                # Ensure the input is on the same device as the model
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

                # Make the prediction
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)

                # Get the predicted class probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Get the predicted label (class with the highest probability)
                predicted_label = torch.argmax(probs).item()

                # Check if the predicted label corresponds to an offensive word
                if predicted_label == 1:  # Assuming 1 corresponds to the offensive class
                    print(f"Offensive word detected: {word}")
                    play_beep()

                # Print the results
                print(f"Input Text: {word}")
                print(f"Predicted Label: {predicted_label}")
                print(f"Class Probabilities: {probs}")
