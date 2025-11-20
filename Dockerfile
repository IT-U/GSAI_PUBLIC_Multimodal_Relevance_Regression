# Small Python 3.12 image
FROM python:3.12-slim AS runtime

# Working directoryW
WORKDIR /workspace

# Install Python dependencies from requirements.txt file
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The code to run when container is started:
# Common practice to keep the Docker container running without performing any significant action. 
ENTRYPOINT ["tail", "-f", "/dev/null"]