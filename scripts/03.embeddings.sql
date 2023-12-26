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
        )
SELECT  place, (values::REAL[])[0:5]
FROM    embeddings