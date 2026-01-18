variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "ecs_cluster_id" {
  description = "ECS cluster ID"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnets" {
  description = "Subnet IDs for Qdrant"
  type        = list(string)
}

variable "security_groups" {
  description = "Security group IDs for Qdrant"
  type        = list(string)
}

variable "log_group_name" {
  description = "CloudWatch log group name"
  type        = string
}
