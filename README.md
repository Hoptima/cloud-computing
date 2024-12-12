# HOPTIMA Cloud Computing
## Overview
This document provides a comprehensive overview of the backend server for our project, detailing the technologies and services used, as well as the deployment and development environment setup.
## Technologies and Framework
### Python
**- Programming Language:** The backend server is developed using Python 3.9 (Docker Base Image: python:3.9-slim, Local Development Environment: Python 3.12.2) \
**- Framework:** We use Flask as the framework for building the server-side application.
### Google Cloud Services
**- Cloud Storage (Bucket) :** We store machine learning models to bucket storage \
**- Firestore :** All of user searching history will store to firestore \
**- Cloud Run :** Our backend server is continuously deployed to Google Cloud Run, for scalability, and cost-effecient deployment.
### API and Libraries
**- TensorFlow.js:** Used for running machine learning models on the server.
### Containerization
**- Docker :** We use Docker for containerizing our backend server, ensuring consistent environments across development, testing, and production.
