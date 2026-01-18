output "ecr_repository_url" {
  description = "ECR repository URL for API"
  value       = aws_ecr_repository.api.repository_url
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = module.alb.dns_name
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = var.certificate_arn != "" || var.domain_name != "" ? "https://${module.alb.dns_name}" : "http://${module.alb.dns_name}"
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "qdrant_service_dns" {
  description = "Qdrant service DNS name"
  value       = module.qdrant.service_dns
}

output "models_bucket_name" {
  description = "S3 bucket name for models"
  value       = aws_s3_bucket.models.bucket
}

output "cloudwatch_log_group_api" {
  description = "CloudWatch log group name for API"
  value       = aws_cloudwatch_log_group.api.name
}
