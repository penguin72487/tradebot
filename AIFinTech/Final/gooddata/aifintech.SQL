-- 建議先建立資料庫
-- CREATE DATABASE aifintech;

-- ========================
-- 1) 股票主檔
-- ========================
CREATE TABLE stocks (
    stock_id     BIGINT NOT NULL PRIMARY KEY,       -- 股票代號整數，例如 2330
    market       VARCHAR(10) NOT NULL DEFAULT 'TW', -- 市場別
    name         VARCHAR(64) NOT NULL,              -- 股票名稱
    yf_symbol    VARCHAR(32) NOT NULL,              -- yfinance 用：'2330.TW'
    UNIQUE (stock_id, market),
    UNIQUE (yf_symbol)
);

-- ========================
-- 2) 年度財報
-- ========================
-- annual_financials 改成小寫欄位名
CREATE TABLE annual_financials (
    stock_id                 BIGINT NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    year                     INT    NOT NULL,
    營收_億                  NUMERIC,
    營益_億                  NUMERIC,
    淨利率_pct               NUMERIC,
    淨利增減_pct             NUMERIC,
    roa_pct                  NUMERIC,
    發行量_萬張               NUMERIC,
    股東權益總額_億           NUMERIC,
    負債總額_億               NUMERIC,
    流動資產對流動負債_pct    NUMERIC,
    速動資產對流動負債_pct    NUMERIC,
    存貨週轉率                NUMERIC,
    應收帳款週轉率            NUMERIC,
    PRIMARY KEY (stock_id, year)
);

-- ========================
-- 3) 日價表
-- ========================
CREATE TABLE daily_prices (
    stock_id   BIGINT NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    dt         DATE   NOT NULL,
    open_p     NUMERIC,
    high_p     NUMERIC,
    low_p      NUMERIC,
    close_p    NUMERIC,
    adj_close  NUMERIC,
    volume     BIGINT,
    PRIMARY KEY (stock_id, dt)
);
CREATE INDEX idx_daily_prices_stock_dt ON daily_prices(stock_id, dt DESC);

-- ========================
-- 4) 年末價格 View
-- ========================
CREATE OR REPLACE VIEW v_year_end_close AS
SELECT
    dp.stock_id,
    EXTRACT(YEAR FROM dp.dt)::INT AS year,
    FIRST_VALUE(dp.close_p) OVER (
        PARTITION BY dp.stock_id, EXTRACT(YEAR FROM dp.dt)
        ORDER BY dp.dt DESC
    ) AS closing_price_year
FROM daily_prices dp;

-- ========================
-- 5) financial_features (物化 View)
-- ========================
-- financial_features 用小寫
CREATE MATERIALIZED VIEW financial_features AS
SELECT
    a.stock_id,
    a.year,
    v.closing_price_year,
    v.closing_price_year * a.發行量_萬張 * 1000 AS market_cap_mil,
    v.closing_price_year / NULLIF((a.股東權益總額_億 * 100 / a.發行量_萬張), 0) AS pb_ratio,
    (v.closing_price_year * a.發行量_萬張 * 1000) / NULLIF((a.營收_億 * 100), 0) AS ps_ratio,
    (a.淨利率_pct/100 * a.營收_億) / NULLIF(a.股東權益總額_億, 0) AS roe_m,
    a.營益_億 / NULLIF(a.營收_億, 0) AS opm,
    a.淨利率_pct/100 AS npm,
    a.roa_pct/100 AS roa,
    a.負債總額_億 / NULLIF(a.股東權益總額_億, 0) AS debt_to_equity,
    a.流動資產對流動負債_pct/100 AS current_ratio_m,
    a.速動資產對流動負債_pct/100 AS quick_ratio_m,
    a.存貨週轉率 AS inventory_turnover_m,
    a.應收帳款週轉率 AS ar_turnover_m,
    (a.營益_億 - LAG(a.營益_億) OVER w) / NULLIF(LAG(a.營益_億) OVER w, 0) AS op_growth_m,
    (a.淨利率_pct - LAG(a.淨利率_pct) OVER w) / NULLIF(LAG(a.淨利率_pct) OVER w, 0) AS net_income_growth_m,
    LEAD(v.closing_price_year) OVER w AS future_price,
    (LEAD(v.closing_price_year) OVER w / NULLIF(v.closing_price_year, 0)) - 1 AS return_1y,
    CASE WHEN (LEAD(v.closing_price_year) OVER w / NULLIF(v.closing_price_year, 0)) - 1 > 0 THEN 1 ELSE 0 END AS return_label
FROM annual_financials a
LEFT JOIN v_year_end_close v
    ON a.stock_id = v.stock_id AND a.year = v.year
WINDOW w AS (PARTITION BY a.stock_id ORDER BY a.year)
WITH DATA;

-- 唯一索引讓 CONCURRENTLY 可用
CREATE UNIQUE INDEX idx_fin_features_unique ON financial_features(stock_id, year);

-- ========================
-- 6) Trigger：價格更新後自動刷新 financial_features
-- ========================
CREATE OR REPLACE FUNCTION refresh_financial_features()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY financial_features;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_refresh_fin_features
AFTER INSERT OR UPDATE OR DELETE ON daily_prices
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_financial_features();
