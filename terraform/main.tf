terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Computer-Vision-Pipeline"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

module "vpc" {
  source = "./modules/vpc"

  project_name       = var.project_name
  environment        = var.environment
  availability_zones = data.aws_availability_zones.available.names
  cidr_block         = var.vpc_cidr
}

# ECR Repository for API
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api-${var.environment}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# S3 Bucket for models and data
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.project_name}-models-${var.environment}"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}-api-${var.environment}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "qdrant" {
  name              = "/ecs/${var.project_name}-qdrant-${var.environment}"
  retention_in_days = 7
}

# RDS PostgreSQL Database
module "rds" {
  source = "./modules/rds"

  project_name    = var.project_name
  environment     = var.environment
  vpc_id          = module.vpc.vpc_id
  subnets         = module.vpc.private_subnet_ids
  security_groups = [module.vpc.rds_security_group_id]
  db_name         = var.db_name
  db_username     = var.db_username
  db_password     = var.db_password
  instance_class  = var.rds_instance_class
  allocated_storage = var.rds_allocated_storage
}

# Qdrant Service (ECS Fargate)
module "qdrant" {
  source = "./modules/qdrant"

  project_name    = var.project_name
  environment     = var.environment
  ecs_cluster_id  = aws_ecs_cluster.main.id
  vpc_id          = module.vpc.vpc_id
  subnets         = module.vpc.private_subnet_ids
  security_groups = [module.vpc.ecs_security_group_id]
  log_group_name  = aws_cloudwatch_log_group.qdrant.name
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"

  project_name    = var.project_name
  environment     = var.environment
  vpc_id          = module.vpc.vpc_id
  subnets         = module.vpc.public_subnet_ids
  security_groups = [module.vpc.alb_security_group_id]
  certificate_arn = var.certificate_arn != "" ? var.certificate_arn : (var.domain_name != "" ? data.aws_acm_certificate.main[0].arn : "")
}

# ACM Certificate (if domain_name provided)
data "aws_acm_certificate" "main" {
  count       = var.domain_name != "" && var.certificate_arn == "" ? 1 : 0
  domain      = var.domain_name
  statuses    = ["ISSUED"]
  most_recent = true
}

# API Service (ECS Fargate)
module "api" {
  source = "./modules/ecs"

  project_name         = var.project_name
  environment          = var.environment
  service_name         = "api"
  ecs_cluster_id       = aws_ecs_cluster.main.id
  ecr_repository_url   = aws_ecr_repository.api.repository_url
  image_tag            = var.image_tag
  vpc_id               = module.vpc.vpc_id
  subnets              = module.vpc.private_subnet_ids
  security_groups      = [module.vpc.ecs_security_group_id]
  alb_target_group_arn = module.alb.target_group_arn
  alb_listener_arn     = module.alb.listener_arn
  log_group_name       = aws_cloudwatch_log_group.api.name
  models_bucket_name   = aws_s3_bucket.models.bucket
  cpu                  = var.api_cpu
  memory               = var.api_memory
  desired_count        = var.api_desired_count
  container_port       = 8000

  environment_variables = {
    QDRANT_URL        = "http://${module.qdrant.service_dns}:6333"
    POSTGRES_HOST     = module.rds.address
    POSTGRES_PORT     = tostring(module.rds.port)
    POSTGRES_DB       = var.db_name
    POSTGRES_USER     = var.db_username
    POSTGRES_PASSWORD = var.db_password
    QDRANT_COLLECTION = var.qdrant_collection
    MODELS_DIR        = "/data/models"
    DATA_DIR          = "/data/data"
    OUTPUT_DIR        = "/data/outputs"
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "api_high_error_rate" {
  alarm_name          = "${var.project_name}-${var.environment}-api-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Alert when API 5xx error rate exceeds threshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = module.alb.arn_suffix
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-api-high-error-rate"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_high_latency" {
  alarm_name          = "${var.project_name}-${var.environment}-api-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 2.0
  alarm_description   = "Alert when API response time exceeds 2 seconds"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = module.alb.arn_suffix
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-api-high-latency"
  }
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "${var.project_name}-${var.environment}-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Alert when RDS CPU utilization exceeds 80%"
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = module.rds.instance_id
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-rds-cpu-high"
  }
}
