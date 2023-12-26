## Explain Extended New Year 2024: GPT in 500 lines of SQL

https://explainextended.com/2023/12/31/happy-new-year-15/

### About

This repository contains the code for the article [Explain Extended New Year 2024: GPT in 500 lines of SQL](https://explainextended.com/2023/12/31/happy-new-year-15/).

The SQL code described in the article runs on PostgreSQL 15.0 and uses [pgvector](https://github.com/pgvector/pgvector). It depends on tables populated with the parameters of the GPT-2 language model, which are not included in this repository.

To create the database and the tables, you need to build and run the Docker image `explainextended2024`. Alternatively, you can manually create the database, install pgvector, and run the script `create.sql`.

To populate the tables, you need to run the script `populate.py`. This script will download the GPT-2 parameters from OpenAI's website and populate the tables.

### Requirements

* Docker
* Python 3.8 or later

### Installation

1. Build and run the database image:

   ```
   docker build --tag explainextended2024 .
   docker run -d --name pgvector -e POSTGRES_DB=explainextended2024 -e POSTGRES_PASSWORD=MySecretPassword -e POSTGRES_USER=explainextended2024 -p 5432:5432 -v "explainextended2024:/var/lib/postgresql/data" explainextended2024
   ```

   Consider changing the password if your machine is exposed to the Internet.

1. Install Python dependencies:

   ```
   pip install -r requirements.txt
   ```

1. Populate the tables:

   ```
   python populate.py --connection_string="postgresql://explainextended2024:MySecretPassword@localhost:5432/explainextended2024"
   ```

   Make sure you're using the same password as in the `docker run` command.

1. The scripts mentioned in the article are located in the [scripts](./scripts) directory.

**Enjoy!**