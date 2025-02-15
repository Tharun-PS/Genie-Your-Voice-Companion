from flask import Flask, render_template, jsonify, request
import os
import speech_recognition as sr
# import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

app = Flask(__name__)

# Folder for temporary files
TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

ai_assistant = AIVoiceAssistant()

# Route to render the UI
@app.route('/')
def index():
    return render_template('index.html')


# Endpoint for speech-to-text (from frontend)
@app.route('/process_text', methods=['POST'])
def process_audio():
    """Process the audio input, transcribe, and interact with AI assistant."""
    audio_data = request.get_json()
    transcript = audio_data.get('text', '')

    try:
        # Transcribed audio text
        # print(f"User said: {transcript}")

        # Process customer input and get response from AI assistant
        output = ai_assistant.interact_with_llm(transcript)
        if output:
            output = output.lstrip()
            # vs.play_text_to_speech(output) 
            bot_reply = output
        else:
            bot_reply = "I'm sorry, I didn't understand that."

        return jsonify({'transcript': transcript, 'bot_reply': bot_reply})
    
    except sr.UnknownValueError:
        return jsonify({'error': 'Speech recognition could not understand audio.'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f"Could not request results from Google Speech Recognition service; {e}"}), 500


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
