-- =========================================
-- 建議先建立資料庫（如需）
-- CREATE DATABASE aifintech;
-- =========================================

-- 為了重建方便，先清掉可能存在的物件（安全 drop）
DROP TRIGGER IF EXISTS trg_refresh_fin_features ON daily_prices;
DROP FUNCTION IF EXISTS refresh_financial_features();
DROP INDEX IF EXISTS idx_fin_features_unique;
DROP MATERIALIZED VIEW IF EXISTS financial_features;
DROP VIEW IF EXISTS v_year_open_close;

-- ========================
-- 1) 股票主檔
-- ========================
CREATE TABLE IF NOT EXISTS stocks (
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
CREATE TABLE IF NOT EXISTS annual_financials (
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
CREATE TABLE IF NOT EXISTS daily_prices (
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
CREATE INDEX IF NOT EXISTS idx_daily_prices_stock_dt ON daily_prices(stock_id, dt DESC);

-- ========================
-- 4) 每年第一/最後價格 View（同時帶出 close 與 adj_close）
-- ========================
CREATE OR REPLACE VIEW v_year_open_close AS
WITH base AS (
  SELECT stock_id,
         EXTRACT(YEAR FROM dt)::int AS yr,
         dt,
         close_p,
         adj_close
  FROM daily_prices
  WHERE close_p IS NOT NULL OR adj_close IS NOT NULL
),
firsts AS (
  SELECT DISTINCT ON (stock_id, yr)
         stock_id, yr,
         close_p   AS first_close,
         adj_close AS first_adj_close
  FROM base
  ORDER BY stock_id, yr, dt ASC
),
lasts AS (
  SELECT DISTINCT ON (stock_id, yr)
         stock_id, yr,
         close_p   AS last_close,
         adj_close AS last_adj_close
  FROM base
  ORDER BY stock_id, yr, dt DESC
)
SELECT f.stock_id, f.yr AS year,
       f.first_close,  l.last_close,
       f.first_adj_close, l.last_adj_close
FROM firsts f
JOIN lasts  l USING (stock_id, yr);

-- ========================
-- 5) financial_features (物化 View)
--    指標：用 close_p（年收）
--    報酬：用 adj_close（年開/年收）
--    - current_return = (年末 adj / 年初 adj) - 1
--    - current_return_label：>0 → 1，=0 → 0，<0 → -1
--    - 僅針對 current_return 檢查 NULL/NaN/±Inf
-- ========================
if EXISTS financial_features THEN
    DROP MATERIALIZED VIEW financial_features;
END IF;
CREATE MATERIALIZED VIEW financial_features AS
WITH oc AS (
  SELECT stock_id, year,
         first_close,  last_close,
         first_adj_close, last_adj_close
  FROM v_year_open_close
),
raw AS (
  SELECT
      a.stock_id,
      a.year,

      -- 年末收盤價（用 close_p 的 year-end，供估值/指標計算與檢視）
      oc.last_close AS closing_price_year,

      -- 估值與財務指標（全部用 close_p）
      (oc.last_close * a.發行量_萬張 * 1000)                                         AS market_cap_mil,
      (oc.last_close / NULLIF((a.股東權益總額_億 * 100 / NULLIF(a.發行量_萬張,0)),0)) AS pb_ratio,
      ((oc.last_close * a.發行量_萬張 * 1000) / NULLIF((a.營收_億 * 100),0))         AS ps_ratio,

      -- 獲利能力等（原樣保留）
      ((a.淨利率_pct/100.0 * a.營收_億) / NULLIF(a.股東權益總額_億,0))               AS roe_m,
      (a.營益_億 / NULLIF(a.營收_億,0))                                               AS opm,
      (a.淨利率_pct/100.0)                                                            AS npm,
      (a.roa_pct/100.0)                                                                AS roa,
      (a.負債總額_億 / NULLIF(a.股東權益總額_億,0))                                   AS debt_to_equity,
      (a.流動資產對流動負債_pct/100.0)                                                AS current_ratio_m,
      (a.速動資產對流動負債_pct/100.0)                                                AS quick_ratio_m,
      (a.存貨週轉率)                                                                    AS inventory_turnover_m,
      (a.應收帳款週轉率)                                                                AS ar_turnover_m,

      -- 成長率（同股內依年份排序）
      ((a.營益_億 - LAG(a.營益_億) OVER w) / NULLIF(LAG(a.營益_億) OVER w,0))         AS op_growth_m,
      ((a.淨利率_pct - LAG(a.淨利率_pct) OVER w) / NULLIF(LAG(a.淨利率_pct) OVER w,0)) AS net_income_growth_m,

      -- 年內報酬（用 adj_close）
      ((oc.last_adj_close / NULLIF(oc.first_adj_close,0)) - 1)::double precision       AS current_return
  FROM annual_financials a
  LEFT JOIN oc
    ON a.stock_id = oc.stock_id AND a.year = oc.year
  WINDOW w AS (PARTITION BY a.stock_id ORDER BY a.year)
),
enriched AS (
  SELECT
      r.*,
      CASE
        WHEN r.current_return > 0 THEN 1
        WHEN r.current_return < 0 THEN -1
        ELSE 0
      END AS current_return_label
  FROM raw r
)
SELECT
    stock_id, year, closing_price_year,
    market_cap_mil, pb_ratio, ps_ratio, roe_m, opm, npm, roa,
    debt_to_equity, current_ratio_m, quick_ratio_m,
    inventory_turnover_m, ar_turnover_m,
    op_growth_m, net_income_growth_m,
    current_return, current_return_label
FROM enriched
WHERE
    current_return IS NOT NULL
    AND (current_return = current_return)          -- 非 NaN
    AND current_return NOT IN ('Infinity'::float8, '-Infinity'::float8)
WITH DATA;

-- 為 CONCURRENTLY refresh 預留的唯一索引（手動使用時可用）
CREATE UNIQUE INDEX idx_fin_features_unique ON financial_features(stock_id, year);

-- ========================
-- 6) Trigger：價格更新後自動刷新 financial_features
--    注意：觸發器中不能用 CONCURRENTLY
-- ========================
CREATE OR REPLACE FUNCTION refresh_financial_features()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW financial_features;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_refresh_fin_features
AFTER INSERT OR UPDATE OR DELETE ON daily_prices
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_financial_features();

-- ========================
-- 7) 小檢查（可選）
-- ========================
-- SELECT current_return_label, COUNT(*) FROM financial_features GROUP BY 1 ORDER BY 1;
