FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Required by Hugging Face Spaces and OpenEnv graders
EXPOSE 7860

# This boots the OpenEnv API server so the grader can ping reset()
CMD ["openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]
