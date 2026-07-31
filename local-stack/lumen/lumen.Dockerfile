# lumen overlay 镜像：预编译的 linux/amd64 静态二进制（collector + observation），
# 与 services.Dockerfile 同一纪律——目标机不拉私有 Go 模块、不访问公网，
# base 镜像 alpine 随离线包/既有镜像库提供。
FROM alpine:3.20

RUN apk add --no-cache curl ca-certificates tzdata
WORKDIR /app
COPY artifacts/bin/odyss-trace-collector /app/odyss-trace-collector
COPY artifacts/bin/odyss-lumen-observation /app/odyss-lumen-observation
COPY config/collector.yaml /app/collector.yaml
COPY config/observation.yaml /app/observation.yaml
