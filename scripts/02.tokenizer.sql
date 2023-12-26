WITH    input AS
        (
        SELECT  'PostgreSQL is great' AS prompt
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
        SELECT  token, cluster
        FROM    clusters
        JOIN    tokenizer
        USING   (cluster)
        )
SELECT  *
FROM    tokens