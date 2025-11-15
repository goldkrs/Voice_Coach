-- create a user (replace myuser/mypass)
CREATE USER myuser WITH PASSWORD 'mypass';

-- create a database owned by that user
CREATE DATABASE mydb OWNER myuser;

-- give the user basic privileges (optional if owner)
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;