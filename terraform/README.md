# Terraform Infrastructure for Computer-Vision-Pipeline

This Terraform configuration deploys the Computer Vision Pipeline application to AWS using ECS Fargate, RDS PostgreSQL, and Qdrant.

## Architecture

- **VPC**: Custom VPC with public and private subnets across multiple AZs
- **ECR**: Container registry for Docker images
- **ECS Fargate**: Container orchestration for API and Qdrant
- **RDS PostgreSQL**: Managed database for metadata storage
- **Qdrant**: Vector search engine (ECS Fargate with service discovery)
- **Application Load Balancer**: HTTP/HTTPS load balancing
- **S3**: Storage for ML models
- **CloudWatch**: Logging and monitoring

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Docker installed (for building images)

## Quick Start

### 1. Initialize Terraform

```bash
cd terraform
terraform init
```

### 2. Create terraform.tfvars

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region         = "us-east-1"
environment        = "dev"
db_password        = "your_secure_password_here"
domain_name        = "api.example.com"  # Optional, for HTTPS
```

### 3. Validate Configuration

```bash
terraform validate
terraform plan
```

### 4. Deploy Infrastructure

```bash
terraform apply
```

### 5. Build and Push Docker Image

After infrastructure is created:

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_REPO_URL>

# Build image
docker build -f Dockerfile.api -t cv-pipeline-api:latest .

# Tag image
docker tag cv-pipeline-api:latest <ECR_REPO_URL>:latest

# Push image
docker push <ECR_REPO_URL>:latest
```

### 6. Access the API

The ALB DNS name will be in the outputs:

```bash
terraform output api_endpoint
```

Then access: `http://<alb_dns_name>/health`

## Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-east-1` |
| `environment` | Environment (dev/staging/prod) | `dev` |
| `vpc_cidr` | VPC CIDR block | `10.0.0.0/16` |
| `image_tag` | Docker image tag | `latest` |
| `api_cpu` | CPU units for API (1024 = 1 vCPU) | `1024` |
| `api_memory` | Memory for API (in MB) | `2048` |
| `api_desired_count` | Number of API tasks | `2` |
| `db_password` | PostgreSQL password | **required** |
| `rds_instance_class` | RDS instance class | `db.t3.medium` |
| `domain_name` | Domain for HTTPS (optional) | `""` |

## Outputs

- `ecr_repository_url`: ECR repository URL for pushing images
- `alb_dns_name`: Application Load Balancer DNS name
- `api_endpoint`: Full API endpoint URL
- `rds_endpoint`: RDS PostgreSQL endpoint
- `qdrant_service_dns`: Qdrant service DNS name
- `models_bucket_name`: S3 bucket for ML models

## Services

### API Service
- FastAPI application on port 8000
- Connects to PostgreSQL and Qdrant
- Health check: `/health`

### Qdrant Service
- Vector search engine on port 6333
- Accessible via service discovery: `qdrant.cv-pipeline.dev.local`
- Single instance (can scale if needed)

### PostgreSQL
- Managed RDS PostgreSQL 16
- Encrypted storage
- Automated backups

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Note**: RDS final snapshot will be created in production environment.

## Notes

- Models should be uploaded to the S3 bucket before deployment
- Qdrant data is ephemeral (consider EFS for persistence if needed)
- Health check endpoint: `/health`
- Application runs on port 8000
