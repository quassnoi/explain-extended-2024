WITH    RECURSIVE
        initial AS
        (
        SELECT  ARRAY[6307, 47701, 318, 1049] AS input
        ),
        hparams AS
        (
        SELECT  12 AS n_block
        ),
        embeddings AS
        (
        SELECT  place, values
        FROM    initial
        CROSS JOIN
                hparams
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
        )
SELECT  place,
        (SELECT STRING_AGG(TO_CHAR(n, 'S0.000'), ' ') || ' …' FROM UNNEST((values::REAL[])[:10]) AS n) AS q
FROM    ln_f
ORDER BY
        place


