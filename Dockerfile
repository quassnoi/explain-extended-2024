FROM ankane/pgvector:v0.5.1
COPY postgresql.explainextended2024.conf /etc/postgresql.explainextended2024.conf
COPY create.sql /docker-entrypoint-initdb.d/
CMD ["postgres", "-c", "config_file=/etc/postgresql.explainextended2024.conf"]
