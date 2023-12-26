WITH    embeddings AS
        (
        SELECT  place, values
        FROM    UNNEST(ARRAY[6307, 47701, 318, 1049]) WITH ORDINALITY AS tokens (token, ordinality)
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
        c_attn_w AS
        (
        SELECT  *
        FROM    c_attn_w
        WHERE   block = 0
        ),
        c_attn_b AS
        (
        SELECT  *
        FROM    c_attn_b
        WHERE   block = 0
        ),
        ln_1_g AS
        (
        SELECT  *
        FROM    ln_1_g
        WHERE   block = 0
        ),
        ln_1_b AS
        (
        SELECT  *
        FROM    ln_1_b
        WHERE   block = 0
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
                                FROM    embeddings
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
        head AS
        (
        SELECT  place,
                (values::REAL[])[1:64]::VECTOR(64) AS q,
                (values::REAL[])[1 + 768:64 + 768]::VECTOR(64) AS k,
                (values::REAL[])[1 + 1536:64 + 1536]::VECTOR(64) AS v
        FROM    mha_norm
        ),
        sm_input AS
        (
        SELECT  h1.place AS x, h2.place AS y, INNER_PRODUCT(h1.q, h2.k) / 8 + CASE WHEN h2.place > h1.place THEN -1E10 ELSE 0 END AS value
        FROM    head h1
        CROSS JOIN
                head h2
        ),
        sm_diff AS
        (
        SELECT  x, y, value - MAX(value) OVER (PARTITION BY x) AS diff
        FROM    sm_input
        ),
        sm_exp AS
        (
        SELECT  x, y, CASE WHEN diff < -745.13 THEN 0 ELSE EXP(diff) END AS e
        FROM    sm_diff
        ),
        softmax AS
        (
        SELECT  x, y AS place, e / SUM(e) OVER (PARTITION BY x) AS value
        FROM    sm_exp
        ),
        attention AS
        (
        SELECT  place, (ARRAY_AGG(value ORDER BY ordinality))[:3] AS values
        FROM    (
                SELECT  x AS place, SUM(ARRAY_FILL(softmax.value, ARRAY[64])::VECTOR(64) * head.v) AS values
                FROM    softmax
                JOIN    head
                USING   (place)
                GROUP BY
                        x
                ) q
        CROSS JOIN LATERAL
                UNNEST(values::REAL[]) WITH ORDINALITY v (value, ordinality)
        GROUP BY
                place
        )
SELECT  place,
        (SELECT STRING_AGG(TO_CHAR(n, 'S0.000'), ' ') || ' …' FROM UNNEST((q::REAL[])[:3]) AS n) AS q,
        (SELECT STRING_AGG(TO_CHAR(n, 'S0.000'), ' ') || ' …' FROM UNNEST((k::REAL[])[:3]) AS n) AS k,
        (SELECT STRING_AGG(TO_CHAR(n, 'S0.000'), ' ') || ' …' FROM UNNEST((v::REAL[])[:3]) AS n) AS v,
        matrix,
        (SELECT STRING_AGG(TO_CHAR(n, 'S0.000'), ' ') || ' …' FROM UNNEST((values::REAL[])[:3]) AS n) AS attention
FROM    head
JOIN    attention
USING   (place)
JOIN    (
        SELECT  x AS place, STRING_AGG(CASE WHEN value > 0 THEN TO_CHAR(value, '0.00') ELSE '    0' END, ' ' ORDER BY place) AS matrix
        FROM    softmax
        GROUP BY
                x
        ) softmax_grouped
USING   (place)
