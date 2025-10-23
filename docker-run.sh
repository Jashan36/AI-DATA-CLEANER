#!/bin/bash

# AI Data Cleaner - Docker Run Script
# This script provides easy commands to run the AI Data Cleaner with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data output notebooks
    print_success "Directories created successfully"
}

# Build and run with Docker Compose
run_with_compose() {
    print_status "Building and running AI Data Cleaner with Docker Compose..."
    docker-compose up --build
}

# Run with Jupyter Lab
run_with_jupyter() {
    print_status "Building and running AI Data Cleaner with Jupyter Lab..."
    docker-compose --profile jupyter up --build
}

# Build Docker image manually
build_image() {
    print_status "Building Docker image..."
    docker build -t ai-data-cleaner .
    print_success "Docker image built successfully"
}

# Run container manually
run_container() {
    print_status "Running Docker container..."
    docker run -p 8501:8501 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/output:/app/output" \
        ai-data-cleaner
}

# Stop containers
stop_containers() {
    print_status "Stopping all containers..."
    docker-compose down
    print_success "Containers stopped successfully"
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes --remove-orphans
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "AI Data Cleaner - Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Build and run with Docker Compose (default)"
    echo "  jupyter     Run with Jupyter Lab service"
    echo "  build       Build Docker image manually"
    echo "  run         Run container manually"
    echo "  stop        Stop all containers"
    echo "  cleanup     Stop containers and remove images/volumes"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Quick start with Docker Compose"
    echo "  $0 jupyter  # Run with Jupyter Lab at http://localhost:8888"
    echo "  $0 build    # Build image only"
    echo "  $0 cleanup  # Clean up everything"
}

# Main script logic
main() {
    check_docker
    create_directories
    
    case "${1:-start}" in
        "start")
            run_with_compose
            ;;
        "jupyter")
            run_with_jupyter
            ;;
        "build")
            build_image
            ;;
        "run")
            run_container
            ;;
        "stop")
            stop_containers
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"

