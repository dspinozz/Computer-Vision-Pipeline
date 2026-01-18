variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "service_name" {
  description = "Service name (e.g., 'api')"
  type        = string
}

variable "ecs_cluster_id" {
  description = "ECS cluster ID"
  type        = string
}

variable "ecr_repository_url" {
  description = "ECR repository URL"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnets" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}

variable "security_groups" {
  description = "Security group IDs for ECS tasks"
  type        = list(string)
}

variable "alb_target_group_arn" {
  description = "ALB target group ARN"
  type        = string
}

variable "alb_listener_arn" {
  description = "ALB listener ARN"
  type        = string
}

variable "log_group_name" {
  description = "CloudWatch log group name"
  type        = string
}

variable "models_bucket_name" {
  description = "S3 bucket name for models"
  type        = string
}

variable "cpu" {
  description = "CPU units for task"
  type        = number
}

variable "memory" {
  description = "Memory for task (MB)"
  type        = number
}

variable "desired_count" {
  description = "Desired number of tasks"
  type        = number
}

variable "container_port" {
  description = "Container port"
  type        = number
  default     = 8000
}

variable "environment_variables" {
  description = "Environment variables for container"
  type        = map(string)
  default     = {}
}
