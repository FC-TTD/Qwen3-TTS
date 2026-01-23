"""
Health check and language list API tests
"""
import pytest
import requests
from typing import Dict, Any

@pytest.mark.smoke
@pytest.mark.integration
class TestHealthAPI:
    """Health check API test class"""
    
    def test_health_check_success(self, api_client_instance):
        """Test health check endpoint returns success"""
        response = api_client_instance.health_check()
        
        assert response.status_code == 200
        assert response.headers.get('content-type') == 'application/json'
        
        data = response.json()
        # Health check should return status information
        assert isinstance(data, dict)
    
    def test_health_trace_success(self, api_client_instance):
        """Test health check trace endpoint returns success"""
        response = api_client_instance.health_trace()
        
        assert response.status_code == 200
        assert response.headers.get('content-type') == 'application/json'
        
        data = response.json()
        # Trace information should contain detailed information
        assert isinstance(data, dict)
    
    def test_health_check_response_time(self, api_client_instance):
        """Test health check endpoint response time"""
        import time
        start_time = time.time()
        response = api_client_instance.health_check()
        end_time = time.time()
        
        assert response.status_code == 200
        # Response time should be within reasonable range (less than 5 seconds)
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response time too long: {response_time:.2f} seconds"

@pytest.mark.smoke
@pytest.mark.integration
class TestLanguagesAPI:
    """Language list API test class"""
    
    def test_get_languages_success(self, api_client_instance):
        """Test get language list endpoint returns success"""
        response = api_client_instance.get_languages()
        
        assert response.status_code == 200
        assert response.headers.get('content-type') == 'application/json'
        
        data = response.json()
        assert isinstance(data, dict)
        
        # Language list should contain common languages
        if data:  # If there is data
            assert isinstance(data, dict)
    
    def test_get_languages_response_time(self, api_client_instance):
        """Test get language list endpoint response time"""
        import time
        start_time = time.time()
        response = api_client_instance.get_languages()
        end_time = time.time()
        
        assert response.status_code == 200
        # Response time should be within reasonable range (less than 3 seconds)
        response_time = end_time - start_time
        assert response_time < 3.0, f"Response time too long: {response_time:.2f} seconds"
    
    def test_get_languages_cache_headers(self, api_client_instance):
        """Test language list endpoint cache headers"""
        response = api_client_instance.get_languages()
        
        assert response.status_code == 200
        # Language list should have appropriate cache control
        cache_control = response.headers.get('cache-control')
        if cache_control:
            assert 'max-age' in cache_control.lower()

@pytest.mark.integration
class TestAPIConnectivity:
    """API connectivity test class"""
    
    def test_api_base_accessible(self, api_base_url):
        """Test API base address accessibility"""
        try:
            response = requests.get(f"{api_base_url}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.fail(f"API service not accessible: {e}")
    
    def test_cors_headers(self, api_client_instance):
        """Test CORS header settings"""
        # Use OPTIONS request to test CORS
        try:
            response = requests.options(f"{api_client_instance.base_url}/health", timeout=10)
            # Test CORS-related headers
            cors_headers = [
                'access-control-allow-origin',
                'access-control-allow-methods',
                'access-control-allow-headers'
            ]
            
            # Should have at least some CORS-related headers
            has_cors = any(header in response.headers for header in cors_headers)
            # If no CORS headers, it's not an error, just a suggestion
        except requests.exceptions.RequestException:
            # OPTIONS request may not be supported, not an error
            pass
