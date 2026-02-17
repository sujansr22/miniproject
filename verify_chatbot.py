import unittest
from unittest.mock import patch, MagicMock
from app import app
from agri_chat_service import AgriChatService

class TestGeminiChatbot(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        app.config['SECRET_KEY'] = 'test-secret'

    def test_service_initialization(self):
        """Test that service init works."""
        # Mock genai.configure to avoid real network calls during init
        with patch('agri_chat_service.genai.configure'):
            service = AgriChatService()
            self.assertIsNotNone(service)

    @patch('agri_chat_service.genai.GenerativeModel')
    def test_chat_response_success(self, mock_model_class):
        """Test a successful API call."""
        # Mock the model instance and its generate_content method
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Use nitrogen fertilizer."
        mock_model_instance.generate_content.return_value = mock_response
        
        mock_model_class.return_value = mock_model_instance

        # Initialize service (mocking configure again)
        with patch('agri_chat_service.genai.configure'):
            service = AgriChatService()
            # Manually set the model because we mocked the class that creates it
            service.model = mock_model_instance
            
            response = service.get_chat_response("What should I do?")
            self.assertEqual(response, "Use nitrogen fertilizer.")

    def test_missing_api_key(self):
        """Test behavior when API Key is missing."""
        # Temporarily unset key in config (mocking Config is harder, easier to just mock the service attr)
        with patch('agri_chat_service.genai.configure'):
             service = AgriChatService()
             service.model = None # Simons missing config
             
             response = service.get_chat_response("Hello")
             self.assertIn("Error", response)

if __name__ == '__main__':
    unittest.main()
