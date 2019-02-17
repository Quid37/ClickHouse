CREATE DATABASE IF NOT EXISTS test;
DROP TABLE IF EXISTS test.defaults;
CREATE TABLE IF NOT EXISTS test.defaults
(
    param1 Float64, param2 Float64, param3 Float64
) ENGINE = Memory;

insert into test.defaults values
(0.0, 0.1, 2.2);

DROP TABLE IF EXISTS test.model;
select IncrementalClustering(10)(param1, param2, param3) from test.defaults;
