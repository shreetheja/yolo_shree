# Ubuntu 18.04 Software
FROM ubuntu:18.04

# Copy the current Repo to directory
COPY . /app

# Change directory to App
WORKDIR /app

# Update the Packages
RUN apt update

# Install Make For Installing Dependencies
RUN apt install python

# INstall Dependencies
RUN make

# Download Weights
RUN wget https://pjreddie.com/media/files/yolov3.weights