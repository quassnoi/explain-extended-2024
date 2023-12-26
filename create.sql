CREATE EXTENSION vector;

CREATE TABLE c_attn_w
        (
        block INT NOT NULL,
        y INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE c_attn_b
        (
        block INT NOT NULL,
        values VECTOR(2304)
        );

CREATE TABLE ln_1_b
        (
        block INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE ln_1_g
        (
        block INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE ln_2_b
        (
        block INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE ln_2_g
        (
        block INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE c_proj_b
        (
        block INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE c_proj_w
        (
        block INT NOT NULL,
        place INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE mlp_c_fc_w
	(
	block INT NOT NULL,
	place INT NOT NULL,
	values VECTOR(768)
	);

CREATE TABLE mlp_c_fc_b
	(
	block INT NOT NULL,
	values VECTOR(3072)
	);

CREATE TABLE mlp_c_proj_w
        (
        block INT NOT NULL,
        place INT NOT NULL,
        values VECTOR(3072) NOT NULL
        );

CREATE TABLE mlp_c_proj_b
	(
	block INT NOT NULL,
	values VECTOR(768) NOT NULL
	);

CREATE TABLE ln_f_b
        (
        values VECTOR(768) NOT NULL
        );

CREATE TABLE ln_f_g
        (
        values VECTOR(768) NOT NULL
        );

CREATE TABLE wte
        (
        token INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE wpe
        (
        place INT NOT NULL,
        values VECTOR(768) NOT NULL
        );

CREATE TABLE tokenizer
	(
	token INT NOT NULL,
	cluster TEXT NOT NULL
	);

CREATE TABLE encoder
	(
	character TEXT NOT NULL,
	byte INT NOT NULL
	);

CREATE UNIQUE INDEX ix_c_attn_w_block_y ON c_attn_w (block, y);

CREATE UNIQUE INDEX ix_c_attn_b_block ON c_attn_b (block);

CREATE UNIQUE INDEX ix_ln_1_b_block ON ln_1_b (block);

CREATE UNIQUE INDEX ix_ln_1_g_block ON ln_1_g (block);

CREATE UNIQUE INDEX ix_ln_2_b_block ON ln_2_b (block);

CREATE UNIQUE INDEX ix_ln_2_g_block ON ln_2_g (block);

CREATE UNIQUE INDEX ix_c_proj_b_block ON c_proj_b (block);

CREATE UNIQUE INDEX ix_c_proj_w_block ON c_proj_w (block, place);

CREATE UNIQUE INDEX ix_mlp_c_fc_w_block_place ON mlp_c_fc_w (block, place);

CREATE UNIQUE INDEX ix_mlp_c_fc_b_block ON mlp_c_fc_b (block);

CREATE UNIQUE INDEX ix_mlp_c_proj_w_block_x_y ON mlp_c_proj_w (block, place);

CREATE UNIQUE INDEX ix_mlp_c_proj_b_block ON mlp_c_proj_b (block);

CREATE UNIQUE INDEX ix_tokenizer_token ON tokenizer (token);

CREATE UNIQUE INDEX ix_tokenizer_cluster ON tokenizer (cluster);

CREATE UNIQUE INDEX ix_encoder_character ON encoder (character);

CREATE UNIQUE INDEX ix_encoder_byte ON encoder (byte);

CREATE UNIQUE INDEX ix_wte_token ON wte (token);

CREATE UNIQUE INDEX ix_wpe_place ON wpe (place);
