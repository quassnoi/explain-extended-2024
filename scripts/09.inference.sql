SELECT SETSEED(0.20231231);

WITH    RECURSIVE
        input AS
        (
        SELECT  'Happy New Year! I wish you' AS prompt,
                10 AS threshold,
                2 AS temperature,
                1 AS top_n
        ),
        clusters AS
        (
        SELECT  part_position, bpe.*
        FROM    input
        CROSS JOIN LATERAL
                REGEXP_MATCHES(prompt, '''s|''t|''re|''ve|''m|''ll|''d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+', 'g') WITH ORDINALITY AS rm (part, part_position)
        CROSS JOIN LATERAL
                (
                WITH    RECURSIVE
                        bpe AS
                        (
                        SELECT  (n + 1)::BIGINT AS position, character, TRUE AS continue
                        FROM    CONVERT_TO(part[1], 'UTF-8') AS bytes
                        CROSS JOIN LATERAL
                                GENERATE_SERIES(0, LENGTH(bytes) - 1) AS n
                        JOIN    encoder
                        ON      byte = GET_BYTE(bytes, n)
                        UNION ALL
                        (
                        WITH    RECURSIVE
                                base AS
                                (
                                SELECT  *
                                FROM    bpe
                                WHERE   continue
                                ),
                                bn AS
                                (
                                SELECT  ROW_NUMBER() OVER (ORDER BY position) AS position,
                                        continue,
                                        character,
                                        character || LEAD(character) OVER (ORDER BY position) AS cluster
                                FROM    base
                                ),
                                top_rank AS
                                (
                                SELECT  tokenizer.*
                                FROM    bn
                                CROSS JOIN LATERAL
                                        (
                                        SELECT  *
                                        FROM    tokenizer
                                        WHERE   tokenizer.cluster = bn.cluster
                                        LIMIT   1
                                        ) tokenizer
                                ORDER BY
                                        token
                                LIMIT   1
                                ),
                                breaks AS
                                (
                                SELECT  0::BIGINT AS position, 1 AS length
                                UNION ALL
                                SELECT  bn.position,
                                        CASE WHEN token IS NULL THEN 1 ELSE 2 END
                                FROM    breaks
                                JOIN    bn
                                ON      bn.position = breaks.position + length
                                LEFT JOIN
                                        top_rank
                                USING   (cluster)
                                )
                        SELECT  position, character, token IS NOT NULL
                        FROM    breaks
                        LEFT JOIN
                                top_rank
                        ON      1 = 1
                        CROSS JOIN LATERAL
                                (
                                SELECT  STRING_AGG(character, '' ORDER BY position) AS character
                                FROM    bn
                                WHERE   bn.position >= breaks.position
                                        AND bn.position < breaks.position + length
                                ) bn
                        WHERE   position > 0
                        )
                        )
                SELECT  position, character AS cluster
                FROM    bpe
                WHERE   NOT continue
                ) bpe
        ),
        tokens AS
        (
        SELECT  ARRAY_AGG(token ORDER BY part_position, position) AS input
        FROM    clusters
        JOIN    tokenizer
        USING   (cluster)
        ),
        gpt AS
        (
        SELECT  input, ARRAY_LENGTH(input, 1) AS original_length
        FROM    tokens
        UNION ALL
        SELECT  input || next_token.token, original_length
        FROM    gpt
        CROSS JOIN
                input
        CROSS JOIN LATERAL
                (
                WITH    RECURSIVE
                        hparams AS
                        (
                        SELECT  ARRAY_LENGTH(input, 1) AS n_seq,
                                12 AS n_block
                        ),
                        embeddings AS
                        (
                        SELECT  place, values
                        FROM    hparams
                        CROSS JOIN LATERAL
                                UNNEST(input) WITH ORDINALITY AS tokens (token, ordinality)
                        CROSS JOIN LATERAL
                                (
                                SELECT  ordinality - 1 AS place
                                ) o
                        CROSS JOIN LATERAL
                                (
                                SELECT  wte.values + wpe.values AS values
                                FROM    wte
                                CROSS JOIN
                                        wpe
                                WHERE   wte.token = tokens.token
                                        AND wpe.place = o.place
                                ) embedding
                        ),
                        transform AS
                        (
                        SELECT  0 AS block, place, values
                        FROM    embeddings
                        UNION ALL
                        (
                        WITH    previous AS
                                (
                                SELECT  *
                                FROM    transform
                                )
                        SELECT  block + 1 AS block, transformed_layer.*
                        FROM    hparams
                        CROSS JOIN LATERAL
                                (
                                SELECT  block
                                FROM    previous
                                WHERE   block < 12
                                LIMIT   1
                                ) q
                        CROSS JOIN LATERAL
                                (
                                WITH    ln_2_b AS
                                        (
                                        SELECT  *
                                        FROM    ln_2_b
                                        WHERE   block = q.block
                                        ),
                                        ln_2_g AS
                                        (
                                        SELECT  *
                                        FROM    ln_2_g
                                        WHERE   block = q.block
                                        ),
                                        c_proj_w AS
                                        (
                                        SELECT  *
                                        FROM    c_proj_w
                                        WHERE   block = q.block
                                        ),
                                        c_proj_b AS
                                        (
                                        SELECT  *
                                        FROM    c_proj_b
                                        WHERE   block = q.block
                                        ),
                                        mlp_c_fc_w AS
                                        (
                                        SELECT  *
                                        FROM    mlp_c_fc_w
                                        WHERE   block = q.block
                                        ),
                                        mlp_c_fc_b AS
                                        (
                                        SELECT  *
                                        FROM    mlp_c_fc_b
                                        WHERE   block = q.block
                                        ),
                                        mlp_c_proj_w AS
                                        (
                                        SELECT  *
                                        FROM    mlp_c_proj_w
                                        WHERE   block = q.block
                                        ),
                                        mlp_c_proj_b AS
                                        (
                                        SELECT  *
                                        FROM    mlp_c_proj_b
                                        WHERE   block = q.block
                                        ),
                                        c_attn_w AS
                                        (
                                        SELECT  *
                                        FROM    c_attn_w
                                        WHERE   block = q.block
                                        ),
                                        c_attn_b AS
                                        (
                                        SELECT  *
                                        FROM    c_attn_b
                                        WHERE   block = q.block
                                        ),
                                        ln_1_g AS
                                        (
                                        SELECT  *
                                        FROM    ln_1_g
                                        WHERE   block = q.block
                                        ),
                                        ln_1_b AS
                                        (
                                        SELECT  *
                                        FROM    ln_1_b
                                        WHERE   block = q.block
                                        ),
                                        mha_norm AS
                                        (
                                        SELECT  place, mm.values + c_attn_b.values AS values
                                        FROM    (
                                                SELECT  place, ARRAY_AGG(INNER_PRODUCT(c_attn_w.values, layer_norm.values) ORDER BY y)::VECTOR(2304) AS values
                                                FROM    (
                                                        SELECT  place, agg.values * ln_1_g.values + ln_1_b.values AS values
                                                        FROM    (
                                                                SELECT  place, norm.values
                                                                FROM    previous
                                                                CROSS JOIN LATERAL
                                                                        (
                                                                        SELECT  AVG(value) AS mean,
                                                                                VAR_POP(value) AS variance
                                                                        FROM    UNNEST(values::REAL[]) value
                                                                        ) agg
                                                                CROSS JOIN LATERAL
                                                                        (
                                                                        SELECT  ARRAY_AGG((value - mean) / SQRT(variance + 1E-5) ORDER BY ordinality)::VECTOR(768) AS values
                                                                        FROM    UNNEST(values::REAL[]) WITH ORDINALITY AS n(value, ordinality)
                                                                        ) norm
                                                                ) agg
                                                        CROSS JOIN
                                                                ln_1_b
                                                        CROSS JOIN
                                                                ln_1_g
                                                        ) layer_norm
                                                CROSS JOIN
                                                        c_attn_w
                                                GROUP BY
                                                        place
                                                ) mm
                                        CROSS JOIN
                                                c_attn_b
                                        ),
                                        heads AS
                                        (
                                        SELECT  place, head,
                                                (values::REAL[])[(head * 64 + 1):(head * 64 + 64)]::VECTOR(64) AS q,
                                                (values::REAL[])[(head * 64 + 1 + 768):(head * 64 + 64 + 768)]::VECTOR(64) AS k,
                                                (values::REAL[])[(head * 64 + 1 + 1536):(head * 64 + 64 + 1536)]::VECTOR(64) AS v
                                        FROM    mha_norm
                                        CROSS JOIN
                                                GENERATE_SERIES(0, 11) head
                                        ),
                                        sm_input AS
                                        (
                                        SELECT  head, h1.place AS x, h2.place AS y, INNER_PRODUCT(h1.q, h2.k) / 8 + CASE WHEN h2.place > h1.place THEN -1E10 ELSE 0 END AS value
                                        FROM    heads h1
                                        JOIN    heads h2
                                        USING   (head)
                                        ),
                                        sm_diff AS
                                        (
                                        SELECT  head, x, y, value - MAX(value) OVER (PARTITION BY head, x) AS diff
                                        FROM    sm_input
                                        ),
                                        sm_exp AS
                                        (
                                        SELECT  head, x, y, CASE WHEN diff < -745.13 THEN 0 ELSE EXP(diff) END AS e
                                        FROM    sm_diff
                                        ),
                                        softmax AS
                                        (
                                        SELECT  head, x, y AS place, e / SUM(e) OVER (PARTITION BY head, x) AS value
                                        FROM    sm_exp
                                        ),
                                        attention AS
                                        (
                                        SELECT  place, ARRAY_AGG(value ORDER BY head * 64 + ordinality)::VECTOR(768) AS values
                                        FROM    (
                                                SELECT  head, x AS place, SUM(ARRAY_FILL(softmax.value, ARRAY[64])::VECTOR(64) * heads.v) AS values
                                                FROM    softmax
                                                JOIN    heads
                                                USING   (head, place)
                                                GROUP BY
                                                        head, x
                                                ) q
                                        CROSS JOIN LATERAL
                                                UNNEST(values::REAL[]) WITH ORDINALITY v (value, ordinality)
                                        GROUP BY
                                                place
                                        ),
                                        mha AS
                                        (
                                        SELECT  place, w.values + c_proj_b.values + previous.values AS values
                                        FROM    (
                                                SELECT  attention.place, ARRAY_AGG(INNER_PRODUCT(attention.values, c_proj_w.values) ORDER BY c_proj_w.place)::VECTOR(768) AS values
                                                FROM    attention
                                                CROSS JOIN
                                                        c_proj_w
                                                GROUP BY
                                                        attention.place
                                                ) w
                                        CROSS JOIN
                                                c_proj_b
                                        JOIN    previous
                                        USING   (place)
                                        ),
                                        ffn_norm AS
                                        (
                                        SELECT  place, agg.values * ln_2_g.values + ln_2_b.values AS values
                                        FROM    (
                                                SELECT  place, norm.values
                                                FROM    mha
                                                CROSS JOIN LATERAL
                                                        (
                                                        SELECT  AVG(value) AS mean,
                                                                VAR_POP(value) AS variance
                                                        FROM    UNNEST(values::REAL[]) value
                                                        ) agg
                                                CROSS JOIN LATERAL
                                                        (
                                                        SELECT  ARRAY_AGG((value - mean) / SQRT(variance + 1E-5) ORDER BY ordinality)::VECTOR(768) AS values
                                                        FROM    UNNEST(values::REAL[]) WITH ORDINALITY AS n(value, ordinality)
                                                        ) norm
                                                ) agg
                                        CROSS JOIN
                                                ln_2_b
                                        CROSS JOIN
                                                ln_2_g
                                        ),
                                        ffn_a AS
                                        (
                                        SELECT  gelu.place, gelu.values
                                        FROM    (
                                                SELECT  place, w.values + mlp_c_fc_b.values AS values
                                                FROM    (
                                                        SELECT  ffn_norm.place, ARRAY_AGG(INNER_PRODUCT(ffn_norm.values, mlp_c_fc_w.values) ORDER BY mlp_c_fc_w.place)::VECTOR(3072) AS values
                                                        FROM    ffn_norm
                                                        CROSS JOIN
                                                                mlp_c_fc_w
                                                        GROUP BY
                                                                ffn_norm.place
                                                        ) w
                                                CROSS JOIN
                                                        mlp_c_fc_b
                                                ) v
                                        CROSS JOIN LATERAL
                                                (
                                                SELECT  place, ARRAY_AGG(0.5 * value * (1 + TANH(0.797884560802 * (value + 0.044715 * value*value*value))) ORDER BY ordinality)::VECTOR(3072) AS values
                                                FROM    UNNEST(values::REAL[]) WITH ORDINALITY n (value, ordinality)
                                                GROUP BY
                                                        place
                                                ) gelu
                                        ),
                                        ffn AS
                                        (
                                        SELECT  place, w.values + mlp_c_proj_b.values + mha.values AS values
                                        FROM    (
                                                SELECT  ffn_a.place, ARRAY_AGG(INNER_PRODUCT(ffn_a.values, mlp_c_proj_w.values) ORDER BY mlp_c_proj_w.place)::VECTOR(768) AS values
                                                FROM    ffn_a
                                                CROSS JOIN
                                                        mlp_c_proj_w
                                                GROUP BY
                                                        ffn_a.place
                                                ) w
                                        CROSS JOIN
                                                mlp_c_proj_b
                                        JOIN    mha
                                        USING   (place)
                                        )
                                SELECT  *
                                FROM    ffn
                                ) transformed_layer
                        )
                        ),
                        block_output AS
                        (
                        SELECT  *
                        FROM    hparams
                        JOIN    transform
                        ON      transform.block = n_block
                        ),
                        ln_f AS
                        (
                        SELECT  place, norm.values * ln_f_g.values + ln_f_b.values AS values
                        FROM    block_output
                        CROSS JOIN LATERAL
                                (
                                SELECT  AVG(value) AS mean,
                                        VAR_POP(value) AS variance
                                FROM    UNNEST(values::REAL[]) AS n(value)
                                ) agg
                        CROSS JOIN LATERAL
                                (
                                SELECT  ARRAY_AGG((value - mean) / SQRT(variance + 1E-5) ORDER BY ordinality)::VECTOR(768) AS values
                                FROM    UNNEST(values::REAL[]) WITH ORDINALITY AS n (value, ordinality)
                                ) norm
                        CROSS JOIN
                                ln_f_b
                        CROSS JOIN
                                ln_f_g
                        ),
                        logits AS
                        (
                        SELECT  token, INNER_PRODUCT(ln_f.values, wte.values) AS value
                        FROM    hparams
                        JOIN    ln_f
                        ON      ln_f.place = n_seq - 1
                        CROSS JOIN
                                wte
                        ORDER BY
                                value DESC
                        LIMIT   (top_n)
                        ),
                        tokens AS
                        (
                        SELECT  token,
                                high - softmax AS low,
                                high
                        FROM    (
                                SELECT  *,
                                        SUM(softmax) OVER (ORDER BY softmax) AS high
                                FROM    (
                                        SELECT  *, (e / SUM(e) OVER ()) AS softmax
                                        FROM    (
                                                SELECT  *,
                                                        (value - MAX(value) OVER ()) / temperature AS diff
                                                FROM    logits
                                                ) exp_x
                                        CROSS JOIN LATERAL
                                                (
                                                SELECT  CASE WHEN diff < -745.13 THEN 0 ELSE EXP(diff) END AS e
                                                ) exp
                                        ) q
                                ) q
                        ),
                        next_token AS
                        (
                        SELECT  *
                        FROM    (
                                SELECT  RANDOM() AS rnd
                                ) r
                        CROSS JOIN LATERAL
                                (
                                SELECT  *
                                FROM    tokens
                                WHERE   rnd >= low
                                        AND rnd < high
                                )
                        )
                SELECT  *
                FROM    next_token
                ) next_token
        WHERE   ARRAY_LENGTH(input, 1) < original_length + threshold
                AND next_token.token <> 50256
        ),
        output AS
        (
        SELECT  CONVERT_FROM(STRING_AGG(SET_BYTE('\x00', 0, byte), '' ORDER BY position), 'UTF8') AS response
        FROM    (
                SELECT  STRING_AGG(cluster, '' ORDER BY ordinality) AS response
                FROM    input
                JOIN    gpt
                ON      ARRAY_LENGTH(input, 1) = original_length + threshold
                CROSS JOIN LATERAL
                        UNNEST(input) WITH ORDINALITY n (token, ordinality)
                JOIN    tokenizer
                USING   (token)
                ) q
        CROSS JOIN LATERAL
                STRING_TO_TABLE(response, NULL) WITH ORDINALITY n (character, position)
        JOIN    encoder
        USING   (character)
        )
SELECT  *
FROM    output
