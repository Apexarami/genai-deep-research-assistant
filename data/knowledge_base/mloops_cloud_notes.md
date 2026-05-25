# MLOps and Cloud Deployment Notes

MLOps helps teams move machine learning and AI systems from experiments to usable software. Important parts include version control, reproducible environments, testing, deployment, monitoring, and model or prompt evaluation.

FastAPI is useful for exposing AI workflows as backend services. Docker makes deployment more consistent because the same environment can run locally and in the cloud. A CI pipeline can run tests automatically when code changes.

For cloud deployment, teams often use platforms such as AWS, Azure, or Google Cloud. A small AI service can be containerized and deployed behind an API endpoint. In enterprise settings, secret management and environment variables should be used instead of hard coded keys.

Monitoring is especially important for AI systems. Teams should track latency, failures, input patterns, output quality, and cost. These signals help improve the system over time.
