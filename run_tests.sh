#!/bin/bash

# Qwen3-TTS API Test Runner
# This script runs different test suites for the Qwen3-TTS API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
API_BASE_URL="http://qwen-api"
TEST_TYPE="all"
COVERAGE="true"
VERBOSE="false"

# Help function
show_help() {
    echo "Qwen3-TTS API Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --url URL        API base URL (default: http://localhost:8000)"
    echo "  -t, --type TYPE     Test type: smoke, integration, slow, or all (default: all)"
    echo "  -c, --coverage       Enable coverage report (default: true)"
    echo "  -v, --verbose       Verbose output"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests"
    echo "  $0 -t smoke                  # Run smoke tests only"
    echo "  $0 -u http://api.example.com -t integration"
    echo "  $0 -c false -v               # Run tests without coverage, verbose"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            API_BASE_URL="$2"
            shift 2
            ;;
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}Qwen3-TTS API Test Runner${NC}"
echo "=================================="
echo "API URL: $API_BASE_URL"
echo "Test Type: $TEST_TYPE"
echo "Coverage: $COVERAGE"
echo "Verbose: $VERBOSE"
echo ""

# Check if API is accessible
echo -e "${YELLOW}Checking API accessibility...${NC}"
if curl -s --max-time 10 "$API_BASE_URL/health" > /dev/null; then
    echo -e "${GREEN}✓ API is accessible${NC}"
else
    echo -e "${RED}✗ API is not accessible at $API_BASE_URL${NC}"
    echo "Please make sure the API server is running before running tests."
    exit 1
fi

# Install test dependencies if not already installed
echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -q pytest pytest-cov pytest-asyncio pytest-mock requests httpx

# Set environment variables
export API_BASE_URL="$API_BASE_URL"

# Build pytest command
PYTEST_CMD="pytest"

# Add test type filter
case $TEST_TYPE in
    smoke)
        PYTEST_CMD="$PYTEST_CMD -m smoke"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD -m integration"
        ;;
    slow)
        PYTEST_CMD="$PYTEST_CMD -m slow"
        ;;
    all)
        # Run all tests except slow by default
        PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
        ;;
esac

# Add coverage options
if [[ "$COVERAGE" == "true" ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=../qwen_tts --cov-report=term-missing --cov-report=html:htmlcov"
fi

# Add verbosity
if [[ "$VERBOSE" == "true" ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add timeout and other options
PYTEST_CMD="$PYTEST_CMD --timeout=300 --tb=short"

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

# Change to tests directory
cd "$(dirname "$0")/tests"

# Execute tests
if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    
    # Show coverage report if generated
    if [[ "$COVERAGE" == "true" ]] && [[ -f "htmlcov/index.html" ]]; then
        echo ""
        echo -e "${BLUE}Coverage report generated: htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi
