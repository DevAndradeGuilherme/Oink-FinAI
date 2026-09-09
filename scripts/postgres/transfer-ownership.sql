ALTER DATABASE :"app_database" OWNER TO :"migrator_role";
ALTER SCHEMA :"app_schema" OWNER TO :"migrator_role";

SELECT format(
    'ALTER %s %I.%I OWNER TO %I',
    CASE c.relkind
        WHEN 'r' THEN 'TABLE'
        WHEN 'p' THEN 'TABLE'
        WHEN 'S' THEN 'SEQUENCE'
        WHEN 'v' THEN 'VIEW'
        WHEN 'm' THEN 'MATERIALIZED VIEW'
        WHEN 'f' THEN 'FOREIGN TABLE'
    END,
    n.nspname,
    c.relname,
    :'migrator_role'
)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = :'app_schema'
  AND c.relkind IN ('r', 'p', 'S', 'v', 'm', 'f')
  AND pg_get_userbyid(c.relowner) <> :'migrator_role'
ORDER BY c.relkind, c.relname
\gexec

SELECT format(
    'ALTER %s %I.%I OWNER TO %I',
    CASE t.typtype WHEN 'd' THEN 'DOMAIN' ELSE 'TYPE' END,
    n.nspname,
    t.typname,
    :'migrator_role'
)
FROM pg_type t
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = :'app_schema'
  AND t.typrelid = 0
  AND t.typtype IN ('c', 'd', 'e', 'm', 'r')
  AND pg_get_userbyid(t.typowner) <> :'migrator_role'
ORDER BY t.typname
\gexec

SELECT format(
    'ALTER %s %I.%I(%s) OWNER TO %I',
    CASE p.prokind WHEN 'p' THEN 'PROCEDURE' ELSE 'FUNCTION' END,
    n.nspname,
    p.proname,
    pg_get_function_identity_arguments(p.oid),
    :'migrator_role'
)
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = :'app_schema'
  AND pg_get_userbyid(p.proowner) <> :'migrator_role'
ORDER BY p.proname, pg_get_function_identity_arguments(p.oid)
\gexec
