"""
TTS语音合成API测试
"""
import pytest
import requests
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.slow
class TestTTSAPI:
    """TTS speech synthesis API test class"""
    
    def test_tts_basic_synthesis(self, api_client_instance, test_audio_file, test_text):
        """Test basic speech synthesis functionality"""
        response = api_client_instance.tts_synthesis(
            text=test_text,
            ref_audio=test_audio_file,
            ref_text=test_text
        )
        
        assert response.status_code == 200
        # API returns WAV audio file directly
        assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
        
        # Check that we got audio data
        assert len(response.content) > 0
        # Check for WAV header
        assert response.content[:4] == b'RIFF'
    
    def test_tts_with_all_parameters(self, api_client_instance, test_audio_file, test_text):
        """Test speech synthesis with all parameters"""
        response = api_client_instance.tts_synthesis(
            text=test_text,
            ref_audio=test_audio_file,
            ref_text="Reference text",
            language="chinese",
            x_vector_only_mode=False,
            remove_silence=True,
            postprocess=True,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            max_new_tokens=1024
        )
        
        assert response.status_code == 200
        # API returns WAV audio file directly
        assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
        assert len(response.content) > 0
        assert response.content[:4] == b'RIFF'
    
    def test_tts_different_languages(self, api_client_instance, test_audio_file):
        """Test speech synthesis in different languages"""
        test_cases = [
            ("Hello World", "english"),
            ("Bonjour le monde", "french"),
            ("Hola mundo", "spanish"),
        ]
        
        for text, language in test_cases:
            response = api_client_instance.tts_synthesis(
                text=text,
                ref_audio=test_audio_file,
                ref_text=text,
                language=language
            )
            
            assert response.status_code == 200, f"Language {language} test failed"
            # API returns WAV audio file directly
            assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
            assert len(response.content) > 0
            assert response.content[:4] == b'RIFF'
    
    def test_tts_x_vector_only_mode(self, api_client_instance, test_audio_file, test_text):
        """Test X-vector only mode"""
        response = api_client_instance.tts_synthesis(
            text=test_text,
            ref_audio=test_audio_file,
            ref_text=test_text,
            x_vector_only_mode=True
        )
        
        assert response.status_code == 200
        # API returns WAV audio file directly
        assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
        assert len(response.content) > 0
        assert response.content[:4] == b'RIFF'
    
    def test_tts_remove_silence(self, api_client_instance, test_audio_file, test_text):
        """Test silence removal functionality"""
        response = api_client_instance.tts_synthesis(
            text=test_text,
            ref_audio=test_audio_file,
            ref_text=test_text,
            remove_silence=True
        )
        
        assert response.status_code == 200
        # API returns WAV audio file directly
        assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
        assert len(response.content) > 0
        assert response.content[:4] == b'RIFF'
    
    def test_tts_postprocess_disabled(self, api_client_instance, test_audio_file, test_text):
        """Test disabled post-processing"""
        response = api_client_instance.tts_synthesis(
            text=test_text,
            ref_audio=test_audio_file,
            ref_text=test_text,
            postprocess=False
        )
        
        assert response.status_code == 200
        # API returns WAV audio file directly
        assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
        assert len(response.content) > 0
        assert response.content[:4] == b'RIFF'

@pytest.mark.integration
class TestTTSParameters:
    """TTS parameter validation test class"""
    
    def test_tts_parameter_validation(self, api_client_instance, test_audio_file):
        """Test parameter validation"""
        # Test missing required parameters
        try:
            # Missing text parameter
            response = api_client_instance.post("/api/tts", files={
                'ref_audio': open(test_audio_file, 'rb')
            })
            assert response.status_code == 422
        finally:
            pass
        
        # Test missing ref_audio parameter
        response = api_client_instance.post("/api/tts", data={
            'text': 'Test text'
        })
        assert response.status_code == 422
    
    def test_tts_temperature_bounds(self, api_client_instance, test_audio_file, test_text):
        """Test temperature parameter bounds"""
        # Test valid range
        valid_temperatures = [0.1, 0.5, 0.9, 1.0]
        for temp in valid_temperatures:
            response = api_client_instance.tts_synthesis(
                text=test_text,
                ref_audio=test_audio_file,
                temperature=temp
            )
            assert response.status_code == 200
    
    def test_tts_top_p_bounds(self, api_client_instance, test_audio_file, test_text):
        """Test top_p parameter bounds"""
        valid_top_p = [0.1, 0.5, 0.9, 1.0]
        for top_p in valid_top_p:
            response = api_client_instance.tts_synthesis(
                text=test_text,
                ref_audio=test_audio_file,
                top_p=top_p
            )
            assert response.status_code == 200
    
    def test_tts_top_k_bounds(self, api_client_instance, test_audio_file, test_text):
        """Test top_k parameter bounds"""
        valid_top_k = [1, 25, 50, 100]
        for top_k in valid_top_k:
            response = api_client_instance.tts_synthesis(
                text=test_text,
                ref_audio=test_audio_file,
                top_k=top_k
            )
            assert response.status_code == 200
    
    def test_tts_max_tokens_bounds(self, api_client_instance, test_audio_file, test_text):
        """Test max_new_tokens parameter bounds"""
        valid_tokens = [100, 512, 1024, 2048]
        for tokens in valid_tokens:
            response = api_client_instance.tts_synthesis(
                text=test_text,
                ref_audio=test_audio_file,
                max_new_tokens=tokens
            )
            assert response.status_code == 200

@pytest.mark.integration
class TestTTSErrorHandling:
    """TTS error handling test class"""
    
    def test_tts_invalid_audio_format(self, api_client_instance, test_text):
        """Test invalid audio format"""
        # Create a non-audio file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not an audio file")
            invalid_audio = Path(f.name)
        
        try:
            response = api_client_instance.tts_synthesis(
                text=test_text,
                ref_audio=invalid_audio
            )
            # Should return error or handle invalid file
            assert response.status_code in [400, 422, 500]
        finally:
            invalid_audio.unlink()
    
    def test_tts_empty_text(self, api_client_instance, test_audio_file):
        """Test empty text"""
        response = api_client_instance.tts_synthesis(
            text="",
            ref_audio=test_audio_file
        )
        
        # Empty text should be handled or return error
        assert response.status_code in [200, 400, 422]
    
    def test_tts_very_long_text(self, api_client_instance, test_audio_file):
        """Test very long text"""
        long_text = "This is a very long test text. " * 1000  # Repeat 1000 times
        
        response = api_client_instance.tts_synthesis(
            text=long_text,
            ref_audio=test_audio_file
        )
        
        # Very long text should be handled or return error
        assert response.status_code in [200, 400, 413, 422]
    
    def test_tts_unicode_text(self, api_client_instance, test_audio_file):
        """Test Unicode text"""
        unicode_text = "Test Unicode: music microphone expression"
        
        response = api_client_instance.tts_synthesis(
            text=unicode_text,
            ref_audio=test_audio_file,
            ref_text=unicode_text
        )
        
        assert response.status_code == 200
        # API returns WAV audio file directly
        assert response.headers.get('content-type') in ['audio/wav', 'audio/x-wav']
        assert len(response.content) > 0
        assert response.content[:4] == b'RIFF'

@pytest.mark.integration
@pytest.mark.slow
class TestTTSPerformance:
    """TTS performance test class"""
    
    def test_tts_response_time(self, api_client_instance, test_audio_file, test_text):
        """Test TTS response time"""
        import time
        start_time = time.time()
        
        response = api_client_instance.tts_synthesis(
            text=test_text,
            ref_audio=test_audio_file
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # TTS processing time should be within reasonable range (less than 60 seconds)
        assert response_time < 60.0, f"TTS processing time too long: {response_time:.2f} seconds"
    
    def test_tts_concurrent_requests(self, api_client_instance, test_audio_file, test_text):
        """Test concurrent requests"""
        import threading
        import time
        results = []
        
        def make_request():
            response = api_client_instance.tts_synthesis(
                text=test_text,
                ref_audio=test_audio_file
            )
            results.append(response.status_code)
        
        # Create 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join(timeout=120)  # 2 minute timeout
        
        # Check all requests succeeded
        assert len(results) == 5
        assert all(status == 200 for status in results), f"Some requests failed: {results}"
