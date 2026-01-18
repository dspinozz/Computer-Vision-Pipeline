output "dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "zone_id" {
  description = "ALB zone ID"
  value       = aws_lb.main.zone_id
}

output "arn_suffix" {
  description = "ALB ARN suffix for CloudWatch dimensions"
  value       = aws_lb.main.arn_suffix
}

output "target_group_arn" {
  description = "Target group ARN"
  value       = aws_lb_target_group.app.arn
}

output "listener_arn" {
  description = "Listener ARN"
  value       = aws_lb_listener.http.arn
}

output "https_listener_arn" {
  description = "HTTPS Listener ARN (if enabled)"
  value       = var.certificate_arn != "" ? aws_lb_listener.https[0].arn : null
}
