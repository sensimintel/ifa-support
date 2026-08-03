# llm-switch actuator：docker CLI（重启 services 容器）+ python3 标准库 HTTP 服务。
FROM docker:27-cli
RUN apk add --no-cache python3
COPY server.py /app/server.py
EXPOSE 8095
CMD ["python3", "/app/server.py"]
