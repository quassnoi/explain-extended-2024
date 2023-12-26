from itertools import chain
import orjson
import os
import numpy as np
import psycopg
import requests
import tensorflow as tf
import fire
from tqdm import tqdm


# Copied from https://github.com/jaymody/picoGPT/blob/817292baea75f194fb0bb8ba2aa5f947af4e45ee/utils.py#L13-L41
def download_gpt2_files(model_size, model_dir):
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta"
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def model_block_get(n_layer, tf_ckpt_path, name, transpose):
    chunks = [(count, layer, iterator) for layer in range(n_layer) for count, iterator in [model_get(
        tf_ckpt_path, f"h{layer}/{name}", transpose)]]
    return sum(chunk[0] for chunk in chunks), chain.from_iterable(((layer, *record) for record in iterator) for (_, layer, iterator) in chunks)


def model_get(tf_ckpt_path, name, transpose):
    params = np.squeeze(tf.train.load_variable(
        tf_ckpt_path, f"model/{name}"))
    if transpose:
        params = np.transpose(params)
    shape = params.shape[:-1]
    return (np.prod(shape), ((*index, orjson.dumps(params[index].tolist(), option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")) for index in np.ndindex(shape)))


model_tables = [
    ("c_attn_w", "attn/c_attn/w", True, True),
    ("c_attn_b", "attn/c_attn/b", True, False),
    ("c_proj_w", "attn/c_proj/w", True, True),
    ("c_proj_b", "attn/c_proj/b", True, False),
    ("ln_1_b", "ln_1/b", True, False),
    ("ln_1_g", "ln_1/g", True, False),
    ("ln_2_b", "ln_2/b", True, False),
    ("ln_2_g", "ln_2/g", True, False),
    ("mlp_c_fc_w", "mlp/c_fc/w", True, True),
    ("mlp_c_fc_b", "mlp/c_fc/b", True, False),
    ("mlp_c_proj_w", "mlp/c_proj/w", True, True),
    ("mlp_c_proj_b", "mlp/c_proj/b", True, False),
    ("ln_f_b", "ln_f/b", False, False),
    ("ln_f_g", "ln_f/g", False, False),
    ("wpe", "wpe", False, False),
    ("wte", "wte", False, False),
]


def save_to_db(connection, table, count, iterator):
    command = f"COPY {table} FROM STDIN"
    with connection.cursor() as cursor:
        with cursor.copy(command) as copy:
            progress = tqdm(iterator, total=count, desc=table)
            for value in progress:
                copy.write_row(value)
    connection.commit()


def main(connection_string="", models_dir="models"):
    connection = psycopg.connect(connection_string)

    model_size = "124M"
    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with open(os.path.join(model_dir, "hparams.json")) as file:
        hparams = orjson.loads(file.read())

    json_tables = [
        ("tokenizer", os.path.join(model_dir, "encoder.json")),
        ("encoder", "encoder.json"),
    ]

    def get(tf_ckpt_path, name, is_block, transpose):
        if is_block:
            return model_block_get(hparams["n_layer"], tf_ckpt_path, name, transpose)
        else:
            return model_get(tf_ckpt_path, name, transpose)

    def json_get(filename):
        with open(filename) as file:
            items = orjson.loads(file.read()).items()
            return len(items), ((value, key) for key, value in items)

    factories = chain(
        ((table, *get(tf_ckpt_path, name, is_block, transpose))
         for table, name, is_block, transpose in model_tables),
        ((table, *json_get(filename)) for table, filename in json_tables))

    for table, count, iterator in factories:
        save_to_db(connection, table, count, iterator)


if __name__ == "__main__":
    fire.Fire(main)
