-- Multi-asset migration: Add category, point_value, quote_ccy to symbols
-- Run: psql -d ict_trading -f migrate_multi_asset.sql

-- 1) Add new columns to symbols (idempotent with IF NOT EXISTS workaround)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbols' AND column_name='category') THEN
        ALTER TABLE symbols ADD COLUMN category VARCHAR(10) DEFAULT 'forex';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbols' AND column_name='point_value') THEN
        ALTER TABLE symbols ADD COLUMN point_value NUMERIC(16,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbols' AND column_name='quote_ccy') THEN
        ALTER TABLE symbols ADD COLUMN quote_ccy VARCHAR(3) DEFAULT 'USD';
    END IF;
    -- Make sl_buffer nullable for non-ICT symbols
    ALTER TABLE symbols ALTER COLUMN sl_buffer DROP NOT NULL;
END$$;

-- 2) Backfill existing forex symbols
UPDATE symbols SET category = 'forex' WHERE category IS NULL;
UPDATE symbols SET quote_ccy = 'JPY' WHERE name LIKE '%JPY' AND (quote_ccy IS NULL OR quote_ccy = 'USD');
UPDATE symbols SET quote_ccy = 'CHF' WHERE name IN ('USDCHF', 'EURCHF') AND quote_ccy = 'USD';
UPDATE symbols SET quote_ccy = 'GBP' WHERE name = 'EURGBP' AND quote_ccy = 'USD';
UPDATE symbols SET quote_ccy = 'CAD' WHERE name = 'USDCAD' AND quote_ccy = 'USD';

-- 3) Insert new symbols
INSERT INTO symbols (name, pip_size, spread_pips, commission_pips, category, point_value, quote_ccy)
VALUES
    ('SP500',    0.25, 0.5, 0, 'index',  50,     'USD'),
    ('KOSPI200', 0.05, 1.0, 0, 'index',  250000, 'KRW'),
    ('BTCUSD',   0.01, 0,   0, 'crypto', 1,      'USD')
ON CONFLICT (name) DO UPDATE SET
    category = EXCLUDED.category,
    point_value = EXCLUDED.point_value,
    quote_ccy = EXCLUDED.quote_ccy;

-- 4) Increase ohlcv_m1 numeric precision for BTC prices
ALTER TABLE ohlcv_m1 ALTER COLUMN open  TYPE NUMERIC(16,6);
ALTER TABLE ohlcv_m1 ALTER COLUMN high  TYPE NUMERIC(16,6);
ALTER TABLE ohlcv_m1 ALTER COLUMN low   TYPE NUMERIC(16,6);
ALTER TABLE ohlcv_m1 ALTER COLUMN close TYPE NUMERIC(16,6);

-- Verify
SELECT name, category, pip_size, point_value, quote_ccy FROM symbols ORDER BY category, name;
