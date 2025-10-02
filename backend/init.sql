-- Database initialization script
-- This file is run when the PostgreSQL container starts for the first time

-- Create extensions that might be needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- The application will handle table creation via SQLAlchemy migrations
-- This file is mainly for extensions and initial setup