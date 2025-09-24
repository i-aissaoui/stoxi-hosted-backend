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

-- ===================================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ===================================================================

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
SELECT 'Users: ' || COUNT(*) || ' (owner user only)' FROM public.users;

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
SELECT 'Tables created: 7 (all empty except users)' as tables_status;
SELECT 'Owner user: created (password: 123)' as user_status;
SELECT 'Timezone: Africa/Algiers (GMT+1) - PERMANENT' as timezone_status;
SELECT 'Ready for sync from local backend!' as sync_status;
SELECT 'Ready for hosted backend deployment!' as deployment_status;
