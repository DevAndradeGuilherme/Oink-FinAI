SELECT 'database' AS object_kind, current_database() AS schema_name,
       current_database() AS object_name, pg_get_userbyid(datdba) AS owner
FROM pg_database
WHERE datname = current_database()
UNION ALL
SELECT 'schema', nspname, nspname, pg_get_userbyid(nspowner)
FROM pg_namespace
WHERE nspname = :'app_schema'
UNION ALL
SELECT CASE c.relkind
           WHEN 'r' THEN 'table'
           WHEN 'p' THEN 'partitioned_table'
           WHEN 'S' THEN 'sequence'
           WHEN 'v' THEN 'view'
           WHEN 'm' THEN 'materialized_view'
           WHEN 'f' THEN 'foreign_table'
           WHEN 'i' THEN 'index'
           WHEN 'I' THEN 'partitioned_index'
       END,
       n.nspname, c.relname, pg_get_userbyid(c.relowner)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = :'app_schema' AND c.relkind IN ('r', 'p', 'S', 'v', 'm', 'f', 'i', 'I')
UNION ALL
SELECT CASE t.typtype WHEN 'd' THEN 'domain' ELSE 'type' END,
       n.nspname, t.typname, pg_get_userbyid(t.typowner)
FROM pg_type t
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = :'app_schema' AND t.typrelid = 0 AND t.typtype IN ('c', 'd', 'e', 'm', 'r')
UNION ALL
SELECT 'routine', n.nspname, p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')',
       pg_get_userbyid(p.proowner)
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = :'app_schema'
ORDER BY 1, 2, 3;
