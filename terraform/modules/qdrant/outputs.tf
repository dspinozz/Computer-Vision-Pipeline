output "service_dns" {
  description = "Qdrant service DNS name"
  value       = "${aws_service_discovery_service.qdrant.name}.${aws_service_discovery_private_dns_namespace.main.name}"
}

output "service_id" {
  description = "ECS service ID"
  value       = aws_ecs_service.qdrant.id
}
