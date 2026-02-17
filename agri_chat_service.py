import google.generativeai as genai
from config import Config
import os

class AgriChatService:
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        
        if not self.api_key or self.api_key.startswith('your-'):
            print("Warning: Gemini API Key not configured correctly.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                # Use gemini-2.5-flash as verified in debug
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"Error configuring Gemini: {e}")
                self.model = None

    def get_chat_response(self, user_message, context=None):
        """
        Generates a response from Google Gemini API.
        """
        if not self.model:
            return "Error: Gemini API is not configured. Please check config.py."

        # System Prompt / Persona
        system_instructions = (
            "You are an expert agricultural advisor for the 'SmaCro' platform. "
            "Help farmers with crop recommendations, disease identification, and farming advice. "
            "Be practical, concise, and empathetic. "
            "ALWAYS include a disclaimer: 'Please consult a local agriculture officer for critical advice.'"
        )

        # Construct the full prompt
        full_prompt = f"{system_instructions}\n"
        if context:
            full_prompt += f"Context: {context}\n"
        full_prompt += f"\nUser: {user_message}"

        try:
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
                
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "I'm having trouble connecting to the Knowledge Base right now. Please check your internet connection."
