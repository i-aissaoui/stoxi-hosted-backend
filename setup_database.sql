-- ===================================================================
-- COMPLETE DATABASE SETUP SCRIPT FOR HOSTED BACKEND
-- Configures timezone, creates all tables, and sets up initial data
-- ===================================================================

-- Set timezone to Algeria (GMT+1) PERMANENTLY
SET timezone = 'Africa/Algiers';
ALTER DATABASE shiakati_jszg SET timezone = 'Africa/Algiers';

-- Verify timezone is set
SELECT 'Current timezone: ' || current_setting('timezone') as timezone_status;
SELECT 'Current Algeria time: ' || NOW() as current_time;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;

-- ===================================================================
-- DROP EXISTING TABLES (for clean recreation)
-- ===================================================================

DROP TABLE IF EXISTS public.order_items CASCADE;
DROP TABLE IF EXISTS public.orders CASCADE;
DROP TABLE IF EXISTS public.variants CASCADE;
DROP TABLE IF EXISTS public.products CASCADE;
DROP TABLE IF EXISTS public.categories CASCADE;
DROP TABLE IF EXISTS public.customers CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP TABLE IF EXISTS public.delivery_fees CASCADE;

-- ===================================================================
-- CREATE TABLES
-- ===================================================================

-- Categories table
CREATE TABLE IF NOT EXISTS public.categories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Products table
CREATE TABLE IF NOT EXISTS public.products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category_id INTEGER REFERENCES public.categories(id),
    image_url TEXT,
    show_on_website INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Variants table
CREATE TABLE IF NOT EXISTS public.variants (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES public.products(id) ON DELETE CASCADE,
    barcode TEXT,
    price NUMERIC(10, 2) NOT NULL,
    quantity NUMERIC(10, 3) DEFAULT 0,
    width TEXT,
    height TEXT,
    color TEXT,
    stopped BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customers table
CREATE TABLE IF NOT EXISTS public.customers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    phone_number TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Orders table
CREATE TABLE IF NOT EXISTS public.orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES public.customers(id),
    customer_name TEXT,
    customer_phone TEXT,
    total NUMERIC(10, 2) NOT NULL,
    status TEXT DEFAULT 'pending',
    delivery_method TEXT,
    wilaya TEXT,
    commune TEXT,
    address TEXT,
    delivery_fee NUMERIC(10, 2),
    notes TEXT,
    order_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    synced_to_local BOOLEAN DEFAULT FALSE,
    local_order_id INTEGER
);

-- Order items table
CREATE TABLE IF NOT EXISTS public.order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES public.orders(id) ON DELETE CASCADE,
    variant_id INTEGER,
    product_name TEXT,
    variant_details TEXT,
    quantity INTEGER NOT NULL,
    price NUMERIC(10, 2) NOT NULL
);

-- Users table (for API authentication)
CREATE TABLE IF NOT EXISTS public.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Delivery fees table (58 Algerian wilayas)
CREATE TABLE IF NOT EXISTS public.delivery_fees (
    id SERIAL PRIMARY KEY,
    wilaya_code INTEGER NOT NULL,
    wilaya_name VARCHAR NOT NULL,
    home_delivery_fee NUMERIC(10,2) NOT NULL,
    office_delivery_fee NUMERIC(10,2),
    CONSTRAINT delivery_fees_wilaya_code_key UNIQUE (wilaya_code)
);

-- ===================================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ===================================================================

-- Safety migration for existing databases (idempotent)
ALTER TABLE IF EXISTS public.orders
    ADD COLUMN IF NOT EXISTS delivery_fee NUMERIC(10,2);

-- Categories indexes
CREATE INDEX IF NOT EXISTS idx_categories_name ON public.categories(name);

-- Products indexes
CREATE INDEX IF NOT EXISTS idx_products_name ON public.products(name);
CREATE INDEX IF NOT EXISTS idx_products_category_id ON public.products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_show_on_website ON public.products(show_on_website);

-- Variants indexes
CREATE INDEX IF NOT EXISTS idx_variants_product_id ON public.variants(product_id);
CREATE INDEX IF NOT EXISTS idx_variants_barcode ON public.variants(barcode);
CREATE INDEX IF NOT EXISTS idx_variants_stopped ON public.variants(stopped);

-- Customers indexes
CREATE INDEX IF NOT EXISTS idx_customers_phone ON public.customers(phone_number);

-- Orders indexes
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON public.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON public.orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_synced_to_local ON public.orders(synced_to_local);
CREATE INDEX IF NOT EXISTS idx_orders_order_time ON public.orders(order_time);

-- Order items indexes
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON public.order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_variant_id ON public.order_items(variant_id);

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);

-- ===================================================================
-- INSERT ONLY OWNER USER (password: 123)
-- ===================================================================

-- Insert ONLY owner user (username: owner, password: 123)
INSERT INTO public.users (username, email, hashed_password, is_active, is_superuser) VALUES
('owner', 'owner@shiakati.com', '$2b$12$8CyiYsfvCfbiQ0YtfO1VfOf6GtCUVuAhOy.2tqK6zud3Z2tzcvWEO', TRUE, TRUE)
ON CONFLICT (username) DO NOTHING;

-- Seed delivery fees for all 58 wilayas (can be updated later via sync)
INSERT INTO public.delivery_fees (wilaya_code, wilaya_name, home_delivery_fee, office_delivery_fee) VALUES
(1, 'ADRAR', 1000.00, 900.00),
(2, 'CHLEF', 750.00, 450.00),
(3, 'LAGHOUAT', 950.00, 600.00),
(4, 'OUM EL BOUAGHI', 800.00, 450.00),
(5, 'BATNA', 800.00, 450.00),
(6, 'BEJAIA', 800.00, 450.00),
(7, 'BISKRA', 950.00, 600.00),
(8, 'BECHAR', 1100.00, 650.00),
(9, 'BLIDA', 400.00, 300.00),
(10, 'BOUIRA', 750.00, 450.00),
(11, 'TAMENRASSET', 1600.00, 1050.00),
(12, 'TEBESSA', 850.00, 450.00),
(13, 'TLEMCEN', 850.00, 500.00),
(14, 'TIARET', 800.00, 450.00),
(15, 'TIZI OUZOU', 750.00, 450.00),
(16, 'ALGER', 500.00, 350.00),
(17, 'DJELFA', 950.00, 600.00),
(18, 'JIJEL', 800.00, 450.00),
(19, 'SETIF', 750.00, 450.00),
(20, 'SAIDA', 800.00, 450.00),
(21, 'SKIKDA', 800.00, 450.00),
(22, 'SIDI BEL ABBAS', 800.00, 450.00),
(23, 'ANNABA', 800.00, 450.00),
(24, 'GUELMA', 800.00, 450.00),
(25, 'CONSTANTINE', 800.00, 450.00),
(26, 'MEDEA', 750.00, 450.00),
(27, 'MOSTAGANEM', 800.00, 450.00),
(28, 'M SILA', 850.00, 500.00),
(29, 'MASCARA', 800.00, 450.00),
(30, 'OUARGLA', 950.00, 600.00),
(31, 'ORAN', 800.00, 450.00),
(32, 'EL BAYADH', 1100.00, 650.00),
(33, 'ILLIZI', 1600.00, 1050.00),
(34, 'BORDJ BOU ARRERIDJ', 750.00, 450.00),
(35, 'BOUMERDESS', 750.00, 450.00),
(36, 'EL TARF', 800.00, 450.00),
(37, 'TINDOUF', 1600.00, 1050.00),
(38, 'TISSEMSILT', 800.00, 450.00),
(39, 'EL OUED', 950.00, 600.00),
(40, 'KHENCHELA', 800.00, 450.00),
(41, 'SOUK AHRAS', 800.00, 450.00),
(42, 'TIPAZA', 750.00, 450.00),
(43, 'MILA', 800.00, 450.00),
(44, 'AIN DEFLA', 750.00, 450.00),
(45, 'NAAMA', 1100.00, 650.00),
(46, 'AIN TEMOUCHENT', 800.00, 450.00),
(47, 'GHARDAIA', 950.00, 600.00),
(48, 'RELIZANE', 800.00, 450.00),
(49, 'TIMIMOUN', 1400.00, 900.00),
(50, 'BORDJ BADJI MOKHTAR', 1600.00, 1050.00),
(51, 'OULED DJELLAL', 950.00, 550.00),
(52, 'BENI ABBES', 1200.00, 800.00),
(53, 'IN SALAH', 1600.00, 1050.00),
(54, 'IN GUEZZAM', 1600.00, 1050.00),
(55, 'TOUGGOURT', 950.00, 600.00),
(56, 'DJANET', 1600.00, 1050.00),
(57, 'EL M GHAIR', 950.00, 600.00),
(58, 'EL MENIA', 1000.00, 650.00)
ON CONFLICT (wilaya_code) DO NOTHING;

-- ALL OTHER TABLES REMAIN EMPTY - will be populated via sync from local backend

-- ===================================================================
-- VERIFY SETUP
-- ===================================================================

-- Show current timezone
SELECT 'Database timezone: ' || current_setting('timezone') as timezone_info;

-- Show current Algeria time
SELECT 'Current Algeria time: ' || NOW() as algeria_time;

-- Show table counts (should be empty except users)
SELECT 'Categories: ' || COUNT(*) || ' (empty - will sync from local)' FROM public.categories
UNION ALL
SELECT 'Products: ' || COUNT(*) || ' (empty - will sync from local)' FROM public.products
UNION ALL
SELECT 'Variants: ' || COUNT(*) || ' (empty - will sync from local)' FROM public.variants
UNION ALL
SELECT 'Customers: ' || COUNT(*) || ' (empty - will sync from local)' FROM public.customers
UNION ALL
SELECT 'Orders: ' || COUNT(*) || ' (empty - will receive from website)' FROM public.orders
UNION ALL
SELECT 'Users: ' || COUNT(*) || ' (owner user only)' FROM public.users
UNION ALL
SELECT 'Delivery fees: ' || COUNT(*) || ' (58 rows expected)' FROM public.delivery_fees;

-- Show the owner user
SELECT 'Owner user created: ' || username || ' (password: 123)' FROM public.users WHERE username = 'owner';

-- ===================================================================
-- SETUP COMPLETE MESSAGE
-- ===================================================================

SELECT '🎉 DATABASE SETUP COMPLETE! 🎉' as status;

-- Final verification
SELECT 'Database timezone: ' || current_setting('timezone') as final_timezone_check;
SELECT 'Current Algeria time: ' || NOW() as final_time_check;

-- Table summary
SELECT 'Tables created: 8 (delivery_fees pre-seeded, others empty except users)' as tables_status;
SELECT 'Owner user: created (password: 123)' as user_status;
SELECT 'Timezone: Africa/Algiers (GMT+1) - PERMANENT' as timezone_status;
SELECT 'Ready for sync from local backend!' as sync_status;
SELECT 'Ready for hosted backend deployment!' as deployment_status;
