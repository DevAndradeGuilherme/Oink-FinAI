WITH expected_roles(role_name, can_create_objects, can_write) AS (
    VALUES
        (:'migrator_role'::name, true, true),
        (:'runtime_role'::name, false, true),
        (:'backup_role'::name, false, false)
), role_checks AS (
    SELECT e.role_name,
           r.rolcanlogin
           AND NOT r.rolsuper
           AND NOT r.rolcreatedb
           AND NOT r.rolcreaterole
           AND NOT r.rolreplication
           AND NOT r.rolbypassrls AS valid
    FROM expected_roles e
    LEFT JOIN pg_roles r ON r.rolname = e.role_name
), membership_checks AS (
    SELECT r.rolname AS role_name, count(m.*) = 0 AS valid
    FROM pg_roles r
    LEFT JOIN pg_auth_members m ON r.oid IN (m.member, m.roleid)
    WHERE r.rolname IN (:'migrator_role', :'runtime_role', :'backup_role')
    GROUP BY r.rolname
), ownership_checks AS (
    SELECT :'migrator_role'::name AS role_name,
           NOT EXISTS (
               SELECT 1 FROM pg_class c
               JOIN pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname = :'app_schema'
                 AND c.relkind IN ('r','p','S','v','m','f','i','I')
                 AND pg_get_userbyid(c.relowner) <> :'migrator_role'
           ) AS valid
), privilege_checks AS (
    SELECT :'runtime_role'::name AS role_name,
           has_database_privilege(:'runtime_role', current_database(), 'CONNECT')
           AND has_schema_privilege(:'runtime_role', :'app_schema', 'USAGE')
           AND NOT has_schema_privilege(:'runtime_role', :'app_schema', 'CREATE')
           AND has_table_privilege(
               :'runtime_role', format('%I.alembic_version', :'app_schema'), 'SELECT'
           )
           AND NOT has_table_privilege(
               :'runtime_role',
               format('%I.alembic_version', :'app_schema'),
               'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'
           ) AS valid
    UNION ALL
    SELECT :'backup_role'::name,
           has_database_privilege(:'backup_role', current_database(), 'CONNECT')
           AND has_schema_privilege(:'backup_role', :'app_schema', 'USAGE')
           AND NOT has_schema_privilege(:'backup_role', :'app_schema', 'CREATE')
), checks AS (
    SELECT 'attributes' AS check_name, role_name, valid FROM role_checks
    UNION ALL SELECT 'memberships', role_name, valid FROM membership_checks
    UNION ALL SELECT 'ownership', role_name, valid FROM ownership_checks
    UNION ALL SELECT 'connect_and_schema', role_name, valid FROM privilege_checks
)
SELECT check_name, role_name, valid FROM checks ORDER BY check_name, role_name;
