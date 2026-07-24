# 本地闭环镜像：直接装入预编译的 linux/amd64 静态二进制（CGO 关闭），
# 目标机不拉私有 Go 模块、不需要 gh_token、不访问公网。
# base 镜像 alpine 在「联网准备阶段」随离线包一并 docker save 带入。
FROM alpine:3.20

RUN apk add --no-cache curl ca-certificates tzdata
WORKDIR /app
COPY artifacts/bin/odyss-services /app/odyss-services
COPY artifacts/bin/odyss-migrate /app/odyss-migrate
COPY artifacts/bin/mockllm /app/mockllm
COPY config/runtime-config.yaml /app/runtime-config.yaml
EXPOSE 8080
CMD ["/app/odyss-services", "-config", "/app/runtime-config.yaml"]
