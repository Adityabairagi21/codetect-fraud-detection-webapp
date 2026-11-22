-- CoDetect - one-time MySQL setup. Run as root in MySQL Workbench or:
--   "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p < database\setup_mysql.sql
-- Replace CHANGE_ME with a strong password, then put the same password in .env (DATABASE_URL).
CREATE DATABASE IF NOT EXISTS codetect CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'codetect_app'@'localhost' IDENTIFIED BY 'CHANGE_ME';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, DROP, REFERENCES ON codetect.* TO 'codetect_app'@'localhost';
-- If the user already existed with another password:
-- ALTER USER 'codetect_app'@'localhost' IDENTIFIED BY 'CHANGE_ME';
FLUSH PRIVILEGES;
-- Tables are created by:  flask --app app init-db
