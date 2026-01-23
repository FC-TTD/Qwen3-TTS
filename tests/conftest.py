"""
pytest configuration file
Provides test fixtures and configuration
"""
import pytest
import os
import tempfile
from pathlib import Path
import requests
from typing import Dict, Any

@pytest.fixture(scope="session")
def api_base_url() -> str:
    """API base URL, configurable via environment variable"""
    return os.getenv("API_BASE_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def test_audio_file() -> Path:
    """Create test audio file"""
    # Create a simple test audio file (WAV format)
    import numpy as np
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Generate a 440Hz sine wave for 1 second
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        sf.write(f.name, audio_data, sample_rate)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()

@pytest.fixture(scope="session")
def test_text() -> str:
    """Test text"""
    return "This is a test text for verifying speech synthesis functionality."

@pytest.fixture(scope="session")
def api_client():
    """API client fixture"""
    class APIClient:
        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip('/')
            self.session = requests.Session()
        
        def get(self, endpoint: str, **kwargs) -> requests.Response:
            return self.session.get(f"{self.base_url}{endpoint}", **kwargs)
        
        def post(self, endpoint: str, **kwargs) -> requests.Response:
            return self.session.post(f"{self.base_url}{endpoint}", **kwargs)
        
        def health_check(self) -> requests.Response:
            return self.get("/health")
        
        def health_trace(self) -> requests.Response:
            return self.session.request("TRACE", f"{self.base_url}/health")
        
        def get_languages(self) -> requests.Response:
            return self.get("/languages")
        
        def tts_synthesis(self, text: str, ref_audio: Path, **kwargs) -> requests.Response:
            ref_audio_file = open(ref_audio, 'rb')
            files = {
                'text': (None, text),
                'ref_audio': ref_audio_file
            }
            
            # Add optional parameters
            for key, value in kwargs.items():
                if value is not None:
                    files[key] = (None, str(value))
            
            try:
                response = self.post("/api/tts", files=files)
                return response
            finally:
                ref_audio_file.close()
    
    return APIClient

@pytest.fixture
def api_client_instance(api_client, api_base_url):
    """API client instance"""
    return api_client(api_base_url)

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration"""
    return {
        "timeout": 30,  # Request timeout (seconds)
        "max_tokens": 2048,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
    }

# Test markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: Smoke tests, verify basic functionality"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests, require complete API service"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests, involve heavy computation"
    )
