CREATE DATABASE binanceKline;
USE binanceKline;


CREATE TABLE binanceKline(
    symbol VARCHAR(20) NOT NULL PRIMARY KEY,
    startTime BIGINT NOT NULL,  -- UNIX 時間戳格式
    endTime BIGINT NOT NULL    -- UNIX 時間戳格式

);

CREATE TABLE binanceKline_By_Symbol(
    symbol VARCHAR(20) NOT NULL,
    openTime BIGINT NOT NULL,  -- UNIX 時間戳格式
    openPrice NUMERIC(20,8) NOT NULL,
    highPrice NUMERIC(20,8) NOT NULL,
    lowPrice NUMERIC(20,8) NOT NULL,
    closePrice NUMERIC(20,8) NOT NULL,
    volume NUMERIC(20,8) NOT NULL,
    closeTime BIGINT NOT NULL,  -- UNIX 時間戳格式
    quoteAssetVolume NUMERIC(20,8) NOT NULL,
    numberOfTrades INT NOT NULL,
    takerBuyBaseAssetVolume NUMERIC(20,8) NOT NULL,
    takerBuyQuoteAssetVolume NUMERIC(20,8) NOT NULL,
    bin_Ignore NUMERIC(20,8) NOT NULL,
    PRIMARY KEY (symbol, openTime),
    FOREIGN KEY (symbol) REFERENCES binanceKline(symbol)
);



CREATE TABLE customer_Index(
    symbol VARCHAR(20) NOT NULL,
    openTime BIGINT NOT NULL,   -- UNIX 時間戳格式
    ATR210 NUMERIC(20,8) NOT NULL,
    volume15 NUMERIC(20,8) NOT NULL,
    avgVolume15 NUMERIC(20,8) NOT NULL,
    PRIMARY KEY (symbol, openTime),
    FOREIGN KEY (symbol, openTime) REFERENCES binanceKline_By_Symbol(symbol, openTime)
);
