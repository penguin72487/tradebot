create table Price (
    exchange VARCHAR(50) NOT NULL,
    -- binance, bybit, okx, etc.
    product VARCHAR(50) NOT NULL,
    -- spot, future, perpetual, etc.
    symbol VARCHAR(50) NOT NULL,
    -- BTCUSDT, ETHUSDT, etc.
    interval VARCHAR(10) NOT NULL,
    -- 1m, 5m, 1h, 1d, etc.
    open NUMERIC(18, 8) NOT NULL,
    high NUMERIC(18, 8) NOT NULL,
    low NUMERIC(18, 8) NOT NULL,
    close NUMERIC(18, 8) NOT NULL,
    volume NUMERIC(18, 8) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (exchange, product, symbol, interval, timestamp)
);
