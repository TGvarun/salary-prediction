-- ============================================================
-- MySQL Setup Script for Salary Prediction System
-- Run once before starting the Flask app with DB_BACKEND=mysql
-- ============================================================

-- 1. Create the database
CREATE DATABASE IF NOT EXISTS salary_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 2. Create a dedicated user (optional but recommended)
-- Replace 'yourpassword' with a strong password
CREATE USER IF NOT EXISTS 'salary_user'@'localhost' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON salary_db.* TO 'salary_user'@'localhost';
FLUSH PRIVILEGES;

-- 3. Use the database
USE salary_db;

-- 4. Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    input_features   JSON         NOT NULL COMMENT 'Raw input JSON sent by user',
    predicted_salary FLOAT        NOT NULL COMMENT 'Model prediction in INR',
    timestamp        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'UTC timestamp'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Verify
DESCRIBE predictions;

-- ============================================================
-- To query stored predictions:
--   SELECT id, JSON_PRETTY(input_features), predicted_salary, timestamp
--   FROM predictions ORDER BY id DESC LIMIT 20;
-- ============================================================
