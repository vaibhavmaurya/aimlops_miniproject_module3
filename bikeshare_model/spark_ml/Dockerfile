FROM apache/spark-py

COPY requirements.txt .

ARG USERNAME=myuser
ARG USER_UID=1000
ARG USER_GID=1000

# Create the non-root user and group
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

# Switch to the non-root user
USER $USERNAME


# Switch to the non-root user
USER $USERNAME

RUN mkdir /.local && chmod -R 777 /.local

RUN pip install -r requirements.txt

COPY data_processing.py .