# Qwen3-TTS API Tests

This directory contains comprehensive test cases for the Qwen3-TTS API service.

## Test Structure

```fs
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Pytest configuration and fixtures
├── pytest.ini            # Pytest settings
├── test_health_api.py    # Health check and language API tests
├── test_tts_api.py       # TTS synthesis API tests
└── README.md             # This file
```

## Test Categories

### Smoke Tests (`@pytest.mark.smoke`)

Basic functionality tests that verify the API is working correctly.

- Health check endpoints
- Language list endpoint
- Basic TTS synthesis

### Integration Tests (`@pytest.mark.integration`)

Full API tests that require the complete service to be running.

- All TTS endpoints with various parameters
- Error handling and validation
- Performance tests

### Slow Tests (`@pytest.mark.slow`)

Tests that involve heavy computation or take longer to run.

- Large text synthesis
- Concurrent request testing
- Performance benchmarks

## Running Tests

### Quick Start

```bash
# Run all smoke tests (recommended for quick verification)
./run_tests.sh -t smoke

# Run all tests (excluding slow tests)
./run_tests.sh

# Run all tests including slow tests
./run_tests.sh -t all
```

### Advanced Usage

```bash
# Test against different API endpoint
./run_tests.sh -u http://api.example.com

# Run with verbose output
./run_tests.sh -v

# Run without coverage (faster)
./run_tests.sh -c false

# Run specific test file
cd tests
pytest test_health_api.py -v

# Run specific test method
pytest test_tts_api.py::TestTTSAPI::test_tts_basic_synthesis -v

# Run with coverage report
pytest --cov=../qwen_tts --cov-report=html
```

### Environment Variables

- `API_BASE_URL`: Base URL of the API service (default: `http://localhost:8000`)

## Test Configuration

### pytest.ini

Contains pytest configuration including:

- Minimum pytest version
- Coverage settings
- Test markers
- Timeout settings
- Warning filters

### conftest.py

Provides test fixtures:

- `api_base_url`: Configurable API base URL
- `test_audio_file`: Generated test audio file
- `test_text`: Sample text for testing
- `api_client`: HTTP client for API requests
- `test_config`: Test configuration parameters

## Test Cases

### Health API Tests (`test_health_api.py`)

1. **Health Check Tests**
   - Basic health check endpoint
   - Health trace endpoint
   - Response time validation

2. **Language API Tests**
   - Language list retrieval
   - Response time validation
   - Cache header validation

3. **Connectivity Tests**
   - API accessibility
   - CORS header validation

### TTS API Tests (`test_tts_api.py`)

1. **Basic Synthesis Tests**
   - Simple text-to-speech synthesis
   - All parameters synthesis
   - Different language support

2. **Parameter Validation Tests**
   - Temperature bounds
   - Top-p and Top-k bounds
   - Max tokens validation
   - Required parameter validation

3. **Error Handling Tests**
   - Invalid audio format
   - Empty text handling
   - Very long text handling
   - Unicode text support

4. **Performance Tests**
   - Response time measurement
   - Concurrent request handling

## API Endpoints Tested

### GET /health

- Status: 200
- Content-Type: application/json
- Response time < 5s

### GET /trace

- Status: 200
- Content-Type: application/json

### GET /languages

- Status: 200
- Content-Type: application/json
- Response time < 3s
- Cache headers validation

### POST /api/tts

- Required parameters: `text`, `ref_audio`
- Optional parameters: `ref_text`, `language`, `x_vector_only_mode`, `remove_silence`, `postprocess`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens`
- Status: 200 (success), 422 (validation error), 400/500 (other errors)
- Content-Type: application/json
- Response time < 60s

## Coverage

Tests aim for >80% code coverage. Coverage reports are generated in:

- Terminal output (summary)
- HTML report: `tests/htmlcov/index.html`

## Troubleshooting

### Common Issues

1. **API Not Accessible**

   ```shell
   ✗ API is not accessible at http://localhost:8000
   ```

   - Ensure the API server is running
   - Check the URL is correct
   - Verify network connectivity

2. **Missing Dependencies**

   ```shell
   ModuleNotFoundError: No module named 'pytest'
   ```

   - Install dependencies: `pip install -r requirements.txt`

3. **Timeout Issues**
   - Increase timeout in pytest.ini
   - Run tests on faster hardware
   - Skip slow tests: `-t smoke`

4. **Coverage Failures**
   - Ensure source code is accessible
   - Check coverage paths in pytest.ini
   - Run without coverage: `-c false`

### Debug Mode

For debugging, run with verbose output and without timeout:

```bash
cd tests
pytest -v --tb=long --timeout=0
```

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use appropriate markers (@pytest.mark.smoke, @pytest.mark.integration, @pytest.mark.slow)
3. Add fixtures to conftest.py if needed
4. Update this README
5. Ensure >80% coverage

## License

These tests are part of the Qwen3-TTS project and follow the same license terms.
