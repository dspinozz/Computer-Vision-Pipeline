# ECS Service Discovery
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${var.project_name}.${var.environment}.local"
  description = "Service discovery namespace for ${var.project_name}"
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "qdrant" {
  name = "qdrant"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }
  }

  health_check_grace_period_seconds = 30
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-${var.environment}-qdrant-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-${var.environment}-qdrant-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Definition for Qdrant
resource "aws_ecs_task_definition" "qdrant" {
  family                   = "${var.project_name}-${var.environment}-qdrant"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name  = "qdrant"
    image = "qdrant/qdrant:v1.12.1"

    portMappings = [{
      containerPort = 6333
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "QDRANT__SERVICE__GRPC_PORT"
        value = "6334"
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group_name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "qdrant"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:6333/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = {
    Name = "${var.project_name}-${var.environment}-qdrant-task"
  }
}

data "aws_region" "current" {}

# ECS Service for Qdrant
resource "aws_ecs_service" "qdrant" {
  name            = "${var.project_name}-${var.environment}-qdrant-service"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.qdrant.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnets
    security_groups = var.security_groups
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.qdrant.arn
  }

  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-qdrant-service"
  }
}
